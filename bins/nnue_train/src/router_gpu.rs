//! `--bucket-mode router` の GPU-resident 学習 (M step)。
//!
//! `nnue-trainer` が GPU (既定の `cuda-oxide` backend) 有効でビルドされた場合、
//! `shogi_features::router_kpabs::RouterKPAbsWeights::train_oracle_batch` /
//! `train_backprop_batch` (CPU 実装) の代わりに本 module の
//! [`RouterGpuState`] が router の forward / backward / Adam を GPU 上で行う。
//! CPU 実装は GPU 未使用時 (`--gpu` 無し build や `RouterGpuState` 初期化
//! 失敗時) のフォールバックとして引き続き有効 — 呼び出し側
//! (`crates/nnue-train::trainer::run`、`TrainerBackend::router_train_*`
//! 経由) は両方に対応する。
//!
//! ## 設計: sparse な forward/backward + Adam のみを GPU 化
//!
//! router の重みテーブルは `num_features (= 125,388) × num_buckets` 要素
//! (既定 9 bucket で ~9MB、f64)。1 position あたりの active index 数は
//! ~76 個 (KP-absolute) で、これに対する forward (sparse 重み和) / backward
//! (atomic scatter) / Adam (全重み更新) が「大きい」側の計算。一方、
//! softmax cross entropy (または backprop モードの Top-K 期待損失) と負荷
//! 分散補助損失の勾配計算は `batch × num_buckets` 要素の dense な計算で
//! num_buckets が小さい (既定 9、上限も数十程度) ため、CPU でも十分速い。
//!
//! そのため本 module は次のように分担する:
//!
//! 1. **GPU**: `router_forward_f64` (`bins/nnue_train/src/kernels/layerstack.rs`) で sparse index → logits
//!    (`batch × num_buckets`) を計算し host へ read back
//! 2. **host**: `RouterKPAbsWeights::train_oracle_batch` /
//!    `train_backprop_batch` と **全く同じ式** で softmax・cross entropy
//!    (or Top-K 期待損失)・負荷分散勾配を計算し `d_logits` (batch size で
//!    平均済) を得る — GPU から取得した `logits` を使うだけで、それ以外の
//!    計算式は CPU 経路と一字一句同じにすることで数値的な同等性を保証する
//! 3. **GPU**: `d_logits` を upload し `router_backward_scatter_f64`
//!    で `grad_weight` (呼出前に 0 初期化) へ atomic scatter-add
//! 4. **GPU**: `router_adam_step_f64` で router 専用の plain Adam
//!    (`shogi_features::router_kpabs` 内部の private `adam_step` と同じ式)
//!    を重み全体に適用
//!
//! ## host ⇔ GPU 同期
//!
//! [`RouterGpuState`] は router の重み / Adam moment (`m`/`v`) / step 数
//! `t` を GPU 上に保持し続ける (毎 step H2D/D2H しない)。しかし:
//!
//! - dataloader worker スレッドの bucket 割当 (`RouterKPAbs::bucket_board`
//!   等、CPU forward) は process-global `RwLock<RouterKPAbsWeights>`
//!   (`RouterKPAbs::state()`) を読む
//! - checkpoint 保存 (`RouterKPAbs::save_full`) も同じ global を読む
//!
//! ため、[`RouterGpuState::sync_to_host`] で GPU 上の最新の重み / Adam 状態を
//! `RouterKPAbs::overwrite_weights` 経由で process-global へ書き戻し、
//! かつ呼び出し側が保持する `RouterAdamState` (checkpoint 保存に使う) も
//! 更新する。呼び出し側 (`crates/nnue-train::trainer::run`) は
//! `--router-refresh-interval` ごと、および checkpoint 保存の直前に必ずこれを
//! 呼ぶ。
//!
//! ## 性能: grow-only workspace で per-step `cudaMalloc`/`cudaFree` を避ける
//!
//! `--router-refresh-interval` の値によっては router の M step が数 batch に 1
//! 回、場合によっては毎 batch 走る。そのため forward/backward が使う一時
//! device buffer (indices / logits / d_logits) は `GrowBuffer` で保持し、
//! 必要サイズが既存 capacity を超えたときだけ再確保する (以後は同じ
//! allocation を使い回す、`Vec::reserve` と同じ発想)。`w`/`m`/`v`/`grad`
//! (router の全重みテーブルサイズ、`--num-buckets` が決まれば学習中不変) は
//! そもそも [`RouterGpuState::new`] で 1 回だけ確保する。
//!
//! また、position ごとの active index 列 (`indices_batch`) の host→device
//! flatten + upload は **1 training step あたり 1 回だけ**
//! ([`RouterGpuState::upload_indices`]) 行い、forward と backward の両方で
//! その device buffer を使い回す (indices は forward/backward で全く同じ値
//! のため、以前の実装がそれぞれで別々に flatten + upload していたのは無駄な
//! 重複だった)。

use std::sync::Arc;

use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, Result as GpuResult, cuda_launch};
use shogi_features::router_kpabs::{RouterAdamState, RouterKPAbsWeights, RouterTrainStats, top_k_error_ranked_indices};

// `cuda_launch!` は `#[kernel]` marker 型 (`__<name>_CudaKernel`) を
// 呼出側 module から bare 名で解決する (`kernels` module 内で `pub(crate) use
// layerstack::*;` 経由で見える名前を、他の呼出側 (`trainer_layerstack.rs` 等)
// と同じく `use crate::*;` で取り込む — path 修飾 (`kernels::router_forward_f64`)
// ではなく bare 名で揃えるのが本 codebase の既存 kernel 呼出しと同じ規約)。
use crate::*;
use crate::trainer_common::{cfg_1d, memset_zero};

/// plain Adam のハイパーパラメータ (`shogi_features::router_kpabs` 内部の
/// private `adam_step` と同じ既定値)。router 側に `--router-beta1` 等の CLI
/// オプションは無いため固定値。
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;

/// 要求サイズが既存 capacity を超えたときだけ再確保する grow-only な
/// `DeviceBuffer<T>` ラッパー。router 学習は batch サイズ (端数 batch)・
/// batch 内の実 active index 数の最大値 (`max_active`) が呼出のたび変わり
/// うるが、大半の呼出では前回以下のサイズに収まるため、一度確保した device
/// allocation を使い回すことで per-step の `cudaMalloc`/`cudaFree` を無くす
/// (`--router-refresh-interval` が小さい設定では router の M step が毎 batch
/// 走りうるため、この allocator churn は無視できない)。
struct GrowBuffer<T: Copy> {
    buf: DeviceBuffer<T>,
    capacity: usize,
}

impl<T: Copy> GrowBuffer<T> {
    fn with_capacity(stream: &CudaStream, capacity: usize) -> GpuResult<Self> {
        let capacity = capacity.max(1);
        Ok(Self { buf: DeviceBuffer::<T>::zeroed(stream, capacity)?, capacity })
    }

    /// `needed` が現在の capacity を超える場合のみ再確保する (2 倍付けで
    /// 償却 O(1) 化、`Vec::reserve` と同じ発想)。既存内容は破棄される —
    /// 呼び出し側は本 fn の直後に必ず [`Self::upload_sync`] 等で上書きする
    /// こと。
    fn ensure_capacity(&mut self, stream: &CudaStream, needed: usize) -> GpuResult<()> {
        if needed > self.capacity {
            let new_capacity = needed.max(self.capacity.saturating_mul(2)).max(1);
            self.buf = DeviceBuffer::<T>::zeroed(stream, new_capacity)?;
            self.capacity = new_capacity;
        }
        Ok(())
    }

    /// `values` (`values.len() <= capacity` 必須、呼出前に [`Self::ensure_capacity`]
    /// すること) を先頭から H2D upload し、転送完了まで `stream` を同期する。
    /// `gpu_runtime::memcpy_htod_async` の「呼出元は転送完了まで `values` を
    /// 生存させる」という unsafe 契約を、この関数内の同期で閉じ込めて安全な
    /// API として公開する (host 側の一時 `Vec` を呼出直後に自由に drop できる)。
    fn upload_sync(&self, stream: &CudaStream, values: &[T]) -> GpuResult<()> {
        debug_assert!(values.len() <= self.capacity);
        // SAFETY: `values` はこの関数呼び出しのスコープ内 (`stream.synchronize()`
        // を待つ行) を通じて生存しており、転送完了を待ってから返るため、呼出元は
        // 戻り値を受け取った時点で `values` を自由に破棄してよい。
        unsafe { gpu_runtime::memcpy_htod_async(&self.buf, values, stream)? };
        stream.synchronize()?;
        Ok(())
    }
}

/// router の重み / Adam moment を GPU 上に保持し、forward / backward / Adam
/// step を GPU 上で行う。[module doc](self) 参照。
pub(crate) struct RouterGpuState {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    num_buckets: usize,
    num_features: usize,
    /// index-major (`w[idx * num_buckets + k]`)、`RouterKPAbsWeights::w` と同レイアウト。
    w: DeviceBuffer<f64>,
    m: DeviceBuffer<f64>,
    v: DeviceBuffer<f64>,
    /// 呼出のたび `memset_zero` してから backward scatter で埋める
    /// (重みテーブルと同サイズで固定、grow 不要)。
    grad: DeviceBuffer<f64>,
    /// `batch × max_active` の `-1` padding 付き flat index (grow-only,
    /// forward/backward で共有 — 1 step あたり 1 回だけ upload する)。
    indices_buf: GrowBuffer<i32>,
    /// `batch × num_buckets` の router logits (forward の出力、host read back 用)。
    logits_buf: GrowBuffer<f64>,
    /// `batch × num_buckets` の d_logits (host で計算した勾配の upload 先)。
    d_logits_buf: GrowBuffer<f64>,
    /// Adam の step 数 (host 側で管理、bias correction の事前計算に使う)。
    t: u64,
}

impl RouterGpuState {
    /// `weights`/`adam` (通常は学習開始直後の初期状態、または checkpoint から
    /// 復元した状態) を H2D して GPU-resident state を作る。
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        weights: &RouterKPAbsWeights,
        adam: &RouterAdamState,
    ) -> GpuResult<Self> {
        let _ = ctx; // stream/module が同じ context 由来である前提 (呼び出し側契約)。
        let num_buckets = weights.num_buckets();
        let n = weights.w.len();
        debug_assert_eq!(n % num_buckets.max(1), 0);
        let num_features = if num_buckets == 0 { 0 } else { n / num_buckets };

        let w = DeviceBuffer::from_host(&stream, &weights.w)?;
        let m = DeviceBuffer::from_host(&stream, adam.m_w())?;
        let v = DeviceBuffer::from_host(&stream, adam.v_w())?;
        let grad = DeviceBuffer::<f64>::zeroed(&stream, n)?;
        let indices_buf = GrowBuffer::with_capacity(&stream, 1)?;
        let logits_buf = GrowBuffer::with_capacity(&stream, 1)?;
        let d_logits_buf = GrowBuffer::with_capacity(&stream, 1)?;

        Ok(Self {
            stream,
            module,
            num_buckets,
            num_features,
            w,
            m,
            v,
            grad,
            indices_buf,
            logits_buf,
            d_logits_buf,
            t: adam.t(),
        })
    }

    /// `indices_batch` (position ごとの active KP-absolute index 列、可変長) を
    /// `-1` padding 付き固定幅 (batch 内の実 active 数の最大値) の flat i32
    /// 配列へ変換する。空 batch (`max_active == 0`) は 1 に底上げする (0 幅
    /// buffer の alloc / launch を避けるだけの安全策で、その場合は forward
    /// が全 position 0 logits を返す)。
    fn flatten_indices(indices_batch: &[Vec<u32>]) -> (Vec<i32>, usize) {
        let batch = indices_batch.len();
        let max_active = indices_batch.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
        let mut flat = vec![-1_i32; batch * max_active];
        for (bi, idxs) in indices_batch.iter().enumerate() {
            for (ni, &idx) in idxs.iter().enumerate() {
                flat[bi * max_active + ni] = idx as i32;
            }
        }
        (flat, max_active)
    }

    /// `indices_batch` を flatten して [`Self::indices_buf`] へ 1 回だけ upload
    /// する (forward/backward が同じ device buffer を参照する — 以前の実装は
    /// forward と backward でそれぞれ別々に flatten + upload しており、host 側
    /// ループと H2D 転送が丸ごと重複していた)。戻り値の `max_active` はその後の
    /// kernel launch (`router_forward_f64`/`router_backward_scatter_f64`) に必要。
    fn upload_indices(&mut self, indices_batch: &[Vec<u32>]) -> GpuResult<usize> {
        let batch = indices_batch.len();
        let (flat, max_active) = Self::flatten_indices(indices_batch);
        self.indices_buf.ensure_capacity(&self.stream, batch * max_active)?;
        self.indices_buf.upload_sync(&self.stream, &flat)?;
        Ok(max_active)
    }

    /// GPU forward: [`Self::indices_buf`] (直前に [`Self::upload_indices`] 済) →
    /// `logits` (`batch × num_buckets`、host へ read back 済)。`batch == 0`
    /// なら空 `Vec` を返す。
    fn forward(&mut self, batch: usize, max_active: usize) -> GpuResult<Vec<f64>> {
        if batch == 0 {
            return Ok(Vec::new());
        }
        self.logits_buf.ensure_capacity(&self.stream, batch * self.num_buckets)?;

        // SAFETY: kernel signature と args の個数・順序・型は一致する
        // (`router_forward_f64` 定義参照)。`self.indices_buf`/`self.logits_buf` は
        // 呼出前に必要な capacity を確保済で、読み書き範囲 (`batch * num_buckets`
        // / `batch * max_active`) は capacity 以内。
        unsafe {
            cuda_launch! {
                kernel: router_forward_f64,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(batch * self.num_buckets),
                args: [
                    slice(self.w),
                    slice(self.indices_buf.buf),
                    slice_mut(self.logits_buf.buf),
                    batch as u32,
                    self.num_buckets as u32,
                    self.num_features as u32,
                    max_active as u32
                ]
            }
        }?;

        let mut logits = vec![0.0_f64; batch * self.num_buckets];
        // SAFETY: `logits` はこの後の `stream.synchronize()` まで生存する。
        unsafe { gpu_runtime::memcpy_dtoh_async(&mut logits, &self.logits_buf.buf, &self.stream)? };
        self.stream.synchronize()?;
        Ok(logits)
    }

    /// GPU backward (atomic scatter, `grad` を 0 初期化してから加算) +
    /// GPU Adam step。[`Self::indices_buf`] (直前に [`Self::upload_indices`] 済)
    /// と `d_logits` (host が既に batch size で平均済のもの、
    /// `RouterKPAbsWeights::train_oracle_batch` の `grad_w *= inv_n` に相当) を使う。
    fn backward_and_adam(
        &mut self,
        batch: usize,
        max_active: usize,
        d_logits: &[f64],
        lr: f64,
        weight_decay: f64,
    ) -> GpuResult<()> {
        if batch == 0 {
            return Ok(());
        }
        self.d_logits_buf.ensure_capacity(&self.stream, batch * self.num_buckets)?;
        self.d_logits_buf.upload_sync(&self.stream, d_logits)?;

        memset_zero(&self.stream, &self.grad).map_err(|e| gpu_runtime::Error::KernelArtifact(e.to_string()))?;

        // SAFETY: kernel signature と args の個数・順序・型は一致する
        // (`router_backward_scatter_f64` 定義参照、`grad` は atomic scatter 先
        // として `&[f64]` で渡す)。`memset_zero` を同じ stream 上で先に発行済で、
        // 以降 stream 上の順序が保証される。
        unsafe {
            cuda_launch! {
                kernel: router_backward_scatter_f64,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(batch * self.num_buckets),
                args: [
                    slice(self.d_logits_buf.buf),
                    slice(self.indices_buf.buf),
                    slice(self.grad),
                    batch as u32,
                    self.num_buckets as u32,
                    self.num_features as u32,
                    max_active as u32
                ]
            }
        }?;

        self.t += 1;
        // f32 scalar 精度の注記は `bins/nnue_train/src/kernels/layerstack.rs`
        // の `router_adam_step_f64` doc を参照。
        let bias_correction1 = (1.0 - ADAM_BETA1.powi(self.t as i32)) as f32;
        let bias_correction2 = (1.0 - ADAM_BETA2.powi(self.t as i32)) as f32;

        // SAFETY: kernel signature と args の個数・順序・型は一致する
        // (`router_adam_step_f64` 定義参照)。`weights`/`m`/`v` は互いに別
        // allocation で、`grad` は本 fn の直前の backward scatter で埋めた
        // 最新の値。
        unsafe {
            cuda_launch! {
                kernel: router_adam_step_f64,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(self.w.len()),
                args: [
                    slice_mut(self.w),
                    slice_mut(self.m),
                    slice_mut(self.v),
                    slice(self.grad),
                    lr as f32,
                    weight_decay as f32,
                    ADAM_BETA1 as f32,
                    ADAM_BETA2 as f32,
                    ADAM_EPS as f32,
                    bias_correction1,
                    bias_correction2,
                    self.w.len() as u32
                ]
            }
        }?;

        Ok(())
    }

    /// `--router-mode hard-EM` の M step。
    /// `RouterKPAbsWeights::train_oracle_batch` と数式上同一 (forward のみ
    /// GPU で計算した `logits` を使う点だけが違う)。
    pub(crate) fn train_oracle_batch(
        &mut self,
        indices_batch: &[Vec<u32>],
        oracle_targets: &[Vec<f64>],
        lr: f64,
        weight_decay: f64,
        balance_weight: f64,
    ) -> GpuResult<RouterTrainStats> {
        debug_assert_eq!(indices_batch.len(), oracle_targets.len());
        let n = indices_batch.len();
        let num_buckets = self.num_buckets;
        if n == 0 {
            return Ok(RouterTrainStats {
                cross_entropy_loss: 0.0,
                balance_loss: 0.0,
                bucket_usage: vec![0.0; num_buckets],
            });
        }

        let max_active = self.upload_indices(indices_batch)?;
        let logits = self.forward(n, max_active)?;

        // 1st pass (`train_oracle_batch` と同じ): probs をキャッシュしつつ
        // router 自身の argmax の分布 f を求める。
        let mut all_probs: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut dispatch_count = vec![0u32; num_buckets];
        for bi in 0..n {
            let logits_i = &logits[bi * num_buckets..(bi + 1) * num_buckets];
            let (dispatch_bucket, _, probs) = RouterKPAbsWeights::bucket_and_probs(logits_i);
            dispatch_count[dispatch_bucket as usize] += 1;
            all_probs.push(probs);
        }
        let mut f = vec![0.0_f64; num_buckets];
        for k in 0..num_buckets {
            f[k] = dispatch_count[k] as f64 / n as f64;
        }

        // 2nd pass (`train_oracle_batch` と同じ式): cross entropy + 負荷分散勾配。
        let mut d_logits_flat = vec![0.0_f64; n * num_buckets];
        let mut total_ce_loss = 0.0_f64;
        let mut avg_probs = vec![0.0_f64; num_buckets];

        for bi in 0..n {
            let target = &oracle_targets[bi];
            debug_assert_eq!(target.len(), num_buckets);
            let probs = &all_probs[bi];
            for k in 0..num_buckets {
                if target[k] != 0.0 {
                    total_ce_loss -= target[k] * probs[k].max(1e-15).ln();
                }
                avg_probs[k] += probs[k];
            }

            let mut d_logits = probs.clone();
            for k in 0..num_buckets {
                d_logits[k] -= target[k];
            }

            if balance_weight != 0.0 {
                let dot: f64 = f.iter().zip(probs.iter()).map(|(&fi, &pi)| fi * pi).sum();
                let n_buckets = num_buckets as f64;
                for j in 0..num_buckets {
                    d_logits[j] += balance_weight * n_buckets * probs[j] * (f[j] - dot);
                }
            }

            for k in 0..num_buckets {
                d_logits_flat[bi * num_buckets + k] = d_logits[k];
            }
        }

        let inv_n = 1.0_f64 / n as f64;
        for g in d_logits_flat.iter_mut() {
            *g *= inv_n;
        }
        for p in avg_probs.iter_mut() {
            *p *= inv_n;
        }

        self.backward_and_adam(n, max_active, &d_logits_flat, lr, weight_decay)?;

        let balance_loss: f64 =
            num_buckets as f64 * f.iter().zip(avg_probs.iter()).map(|(&fi, &pi)| fi * pi).sum::<f64>();

        Ok(RouterTrainStats { cross_entropy_loss: total_ce_loss / n as f64, balance_loss, bucket_usage: f })
    }

    /// `--router-mode backprop` の M step。
    /// `RouterKPAbsWeights::train_backprop_batch` と数式上同一。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn train_backprop_batch(
        &mut self,
        indices_batch: &[Vec<u32>],
        errs_batch: &[Vec<f64>],
        lr: f64,
        weight_decay: f64,
        balance_weight: f64,
        top_k: usize,
    ) -> GpuResult<RouterTrainStats> {
        debug_assert_eq!(indices_batch.len(), errs_batch.len());
        let n = indices_batch.len();
        let num_buckets = self.num_buckets;
        debug_assert!(top_k >= 1 && top_k <= num_buckets);
        if n == 0 {
            return Ok(RouterTrainStats {
                cross_entropy_loss: 0.0,
                balance_loss: 0.0,
                bucket_usage: vec![0.0; num_buckets],
            });
        }

        let max_active = self.upload_indices(indices_batch)?;
        let logits = self.forward(n, max_active)?;

        let mut all_probs: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut dispatch_count = vec![0u32; num_buckets];
        for bi in 0..n {
            let logits_i = &logits[bi * num_buckets..(bi + 1) * num_buckets];
            let (dispatch_bucket, _, probs) = RouterKPAbsWeights::bucket_and_probs(logits_i);
            dispatch_count[dispatch_bucket as usize] += 1;
            all_probs.push(probs);
        }
        let mut f = vec![0.0_f64; num_buckets];
        for k in 0..num_buckets {
            f[k] = dispatch_count[k] as f64 / n as f64;
        }

        let mut d_logits_flat = vec![0.0_f64; n * num_buckets];
        let mut total_expected_loss = 0.0_f64;
        let mut avg_probs = vec![0.0_f64; num_buckets];

        for bi in 0..n {
            let errs = &errs_batch[bi];
            debug_assert_eq!(errs.len(), num_buckets);
            let probs = &all_probs[bi];
            for k in 0..num_buckets {
                avg_probs[k] += probs[k];
            }

            let selected = top_k_error_ranked_indices(errs, top_k);
            let sum_sel: f64 = selected.iter().map(|&k| probs[k]).sum();
            let expected: f64 = selected.iter().map(|&k| (probs[k] / sum_sel) * errs[k]).sum();
            total_expected_loss += expected;

            let mut d_logits = vec![0.0_f64; num_buckets];
            for &k in &selected {
                let q_k = probs[k] / sum_sel;
                d_logits[k] = q_k * (errs[k] - expected);
            }

            if balance_weight != 0.0 {
                let dot: f64 = f.iter().zip(probs.iter()).map(|(&fi, &pi)| fi * pi).sum();
                let n_buckets = num_buckets as f64;
                for j in 0..num_buckets {
                    d_logits[j] += balance_weight * n_buckets * probs[j] * (f[j] - dot);
                }
            }

            for k in 0..num_buckets {
                d_logits_flat[bi * num_buckets + k] = d_logits[k];
            }
        }

        let inv_n = 1.0_f64 / n as f64;
        for g in d_logits_flat.iter_mut() {
            *g *= inv_n;
        }
        for p in avg_probs.iter_mut() {
            *p *= inv_n;
        }

        self.backward_and_adam(n, max_active, &d_logits_flat, lr, weight_decay)?;

        let balance_loss: f64 =
            num_buckets as f64 * f.iter().zip(avg_probs.iter()).map(|(&fi, &pi)| fi * pi).sum::<f64>();

        Ok(RouterTrainStats {
            cross_entropy_loss: total_expected_loss / n as f64,
            balance_loss,
            bucket_usage: f,
        })
    }

    /// GPU 上の重み / Adam moment / step 数を host へ read back する。
    ///
    /// 呼び出し側 (`crates/nnue-train::trainer::run`) は、この戻り値を使って
    /// (a) `RouterKPAbs::overwrite_weights` で process-global state
    /// (dataloader の bucket 割当が読む) を更新し、(b) 呼び出し側が保持する
    /// `RouterAdamState` (checkpoint 保存に使う) の `m_w`/`v_w`/`t` を上書き
    /// する責務を持つ (`--router-refresh-interval` ごと、および checkpoint
    /// 保存直前に必ず呼ぶこと)。
    pub(crate) fn sync_to_host(&self) -> GpuResult<(Vec<f64>, Vec<f64>, Vec<f64>, u64)> {
        let w = self.w.to_host_vec(&self.stream)?;
        let m = self.m.to_host_vec(&self.stream)?;
        let v = self.v.to_host_vec(&self.stream)?;
        Ok((w, m, v, self.t))
    }
}
