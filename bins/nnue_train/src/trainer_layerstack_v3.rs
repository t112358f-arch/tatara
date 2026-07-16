//! `layerstack_v3` GPU trainer — [`crate::trainer_layerstack::GpuTrainer`] ("V2")
//! の bucket 部分を **bucketごとに異なる `l1_out`/`l2_out` サイズ** に一般化した版。
//!
//! # 設計 (詳細は `docs/decisions/2026-07-14-layerstack-v3-per-bucket-dims.md`
//! および `docs/decisions/2026-07-15-layerstack-v3-gpu-trainer.md` を参照)
//!
//! V2 の per-bucket dense kernel (`dense_mm_fwd_bucket_tiled_l1_sorted` /
//! `dense_mm_bwd_weight_bucket_tiled_{l2,l3}` 等) は **9 bucket が同一の
//! `l1_out`/`l2_out` を持つ前提** (固定 register fan-out / 単一 `out_dim` 引数) で
//! 設計されている。bucketごとに違うサイズを持たせるには、この「1 launch で 9
//! bucket 分をまとめて処理する」設計を崩す必要がある。
//!
//! 採用したアプローチ: **bucketごとに独立した dense stack として扱う**。
//!
//! 1. FT (feature transformer) は bucket 非依存なので V2 と完全に同一 (この file
//!    はコード自体もほぼそのまま流用、`ft_post_perspective_grad_fused` の
//!    2 つ目の入力に恒久的に 0 の buffer を渡すことで L1f (このアーキには無い)
//!    の分岐を無害化している)。
//! 2. `combined` (b × ft_out) を bucket_idx でソートした `combined_sorted` を
//!    作る (V2 と同じ `count_buckets` / `exclusive_scan_aligned` /
//!    `scatter_bucket_perm` / `permute_rows_f32` を **align=1 (padding無し)** で
//!    再利用)。
//! 3. bucketごとに **専用の (0-indexed, `batch_size` 分確保済の) buffer** に
//!    L1→活性化→L2→活性化→L3 を `cuBLAS Sgemm` (V2 の L1f shared-dense と同じ
//!    パターン) で計算する。`combined_sorted` の該当区間は生ポインタ + offset で
//!    直接読む (host 側で `offsets[bucket]` を管理、GPU 側の
//!    `bucket_offsets_dev` と同じ prefix sum を host 側でも計算して使う)。
//!    活性化・concat・slice 等は V2 と同じ generic kernel (`crelu_fwd` /
//!    `abs_pow2_scale_fwd` / `concat_l1sqr_main_fwd` / `slice_extract_2d` /
//!    `bias_add_per_row` / `elementwise_add`) をそのまま再利用する — これらは
//!    「batch 数 × dim」しか見ない bucket 非依存な kernel なので、bucket ごとに
//!    別 buffer で複数回呼んでも問題ない。
//! 4. bucket ごとの `net_output` (0-indexed, 1 列) を新設の
//!    [`scatter_by_perm_offset`](crate::gather_by_perm_offset) kernel で
//!    original order の共有 `net_output` buffer に書き戻す。backward は逆に
//!    [`gather_by_perm_offset`] で `dy_net_output` (original order) から
//!    bucket ごとの `dy` を集める。
//! 5. L1 の入力側 grad (`dcombined`) だけは ft_out 幅 (= 大きい) なので、
//!    bucket ごとに dedicated buffer を持つと `9 × batch × ft_out` のメモリを
//!    食う。これを避けるため、`dcombined_sorted` は **共有の 1 本の buffer**
//!    (`batch × ft_out`) にして、cuBLAS の出力ポインタを bucket ごとに
//!    `offsets[bucket] * ft_out` 分オフセットして直接書く (V2 の
//!    `combined_sorted` と同じ発想)。最後に 1 回の `inverse_permute_rows_f32`
//!    で original order の `dcombined` に scatter する。
//!
//! per-bucket dense stack 自体 (L1/L2/L3 + 活性化) の kernel 呼び出し列は
//! [`crate::trainer_simple::SimpleGpuTrainer`] の L1/L2/L3 dense pipeline
//! (cuBLAS ベース) とほぼ同じで、そこに layerstack 固有の skip connection /
//! SqrCReLU concat ([`crate::trainer_layerstack::GpuTrainer`] 参照) を足した形。
//!
//! # スコープ外 (V1 実装で意図的に外したもの)
//!
//! [`nnue_format::layerstack_v3_weights`] と同じ理由で、L1f (shared factorized
//! L1) / PSQT shortcut / threat feature / FT factorizer / FP16 fast path
//! (`--ft-fp16` 等) はこの trainer では扱わない。FP32 のみ。TF32 (cuBLAS Tensor
//! Core) は opt-in で扱う (V2 と同じ `CublasHandle::new(.., enable_tf32)`)。
//!
//! # 検証状況 — 重要
//!
//! **本 file は GPU 環境で一度もビルド/実行されていない** (開発サンドボックスに
//! CUDA が無い)。設計は V2 / Simple の実装パターンを注意深く踏襲しているが、
//! 実際に `cargo build --features gpu` → smoke test → 実 GPU での学習確認が
//! 必須。特に以下は要注意:
//! - `CublasHandle` の sgemm 呼び出しの shape 引数 (m/n/k の対応)。
//! - 新設 2 kernel (`gather_by_perm_offset` / `scatter_by_perm_offset`,
//!   `kernels/layerstack.rs`) の launch 引数。
//! - 各 per-bucket buffer の確保サイズ (`batch_size` 分、bucket 数 9 倍の
//!   メモリを consume する点に注意 — 大きい `--batch-size` では VRAM 使用量を
//!   見ながら調整すること)。

use std::path::Path;

use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig, cuda_launch};
use nnue_format::layerstack_v3_weights::{LayerStackV3Weights, NUM_BUCKETS_V3, l2_in_for};
use nnue_train::optimizer::radam_compute_step_size_denom;
use shogi_features::FeatureSetSpec;

use crate::*;
use crate::{arch::*, kernel_module::*, trainer_common::*};

/// 全 buckets の壁時間・メモリを妥当な範囲に保つための bucket 数。
/// [`NUM_BUCKETS_V3`] と常に同じ (progress8kpabs/9kpabs/kingrank9 いずれも 9
/// bucket)。
const N: usize = NUM_BUCKETS_V3;

/// bucket 1 個分の weight (+ Ranger optimizer state) + 専用 activation
/// workspace。`l1_out`/`l2_out` はこの bucket 固有のサイズ (コンストラクタ引数、
/// 以後不変)。
struct Bucket {
    l1_out: usize,
    l2_out: usize,
    l2_in: usize,
    l1_eff: usize,

    // ---- weight (+ Ranger state) ----
    l1_w: DeviceBuffer<f32>, // (ft_out, l1_out) row-major (in-major)
    l1_w_m: DeviceBuffer<f32>,
    l1_w_v: DeviceBuffer<f32>,
    l1_w_slow: DeviceBuffer<f32>,
    l1_w_grad: DeviceBuffer<f32>,
    l1_b: DeviceBuffer<f32>, // (l1_out)
    l1_b_m: DeviceBuffer<f32>,
    l1_b_v: DeviceBuffer<f32>,
    l1_b_slow: DeviceBuffer<f32>,
    l1_b_grad: DeviceBuffer<f32>,

    l2_w: DeviceBuffer<f32>, // (l2_in, l2_out)
    l2_w_m: DeviceBuffer<f32>,
    l2_w_v: DeviceBuffer<f32>,
    l2_w_slow: DeviceBuffer<f32>,
    l2_w_grad: DeviceBuffer<f32>,
    l2_b: DeviceBuffer<f32>, // (l2_out)
    l2_b_m: DeviceBuffer<f32>,
    l2_b_v: DeviceBuffer<f32>,
    l2_b_slow: DeviceBuffer<f32>,
    l2_b_grad: DeviceBuffer<f32>,

    l3_w: DeviceBuffer<f32>, // (l2_out) [out_dim=1]
    l3_w_m: DeviceBuffer<f32>,
    l3_w_v: DeviceBuffer<f32>,
    l3_w_slow: DeviceBuffer<f32>,
    l3_w_grad: DeviceBuffer<f32>,
    l3_b: DeviceBuffer<f32>, // (1)
    l3_b_m: DeviceBuffer<f32>,
    l3_b_v: DeviceBuffer<f32>,
    l3_b_slow: DeviceBuffer<f32>,
    l3_b_grad: DeviceBuffer<f32>,

    // ---- forward activation workspace (0-indexed, `batch_size` 分確保) ----
    l1_pre: DeviceBuffer<f32>,        // (batch, l1_out)
    l1_main: DeviceBuffer<f32>,       // (batch, l1_eff)
    l1_skip: DeviceBuffer<f32>,       // (batch, 1)
    l1_sqr: DeviceBuffer<f32>,        // (batch, l1_eff)
    l2_pre: DeviceBuffer<f32>,        // (batch, l2_in)
    l2_input: DeviceBuffer<f32>,      // (batch, l2_in)
    l2_dense_out: DeviceBuffer<f32>,  // (batch, l2_out)
    l2_acted: DeviceBuffer<f32>,      // (batch, l2_out)
    l3_out: DeviceBuffer<f32>,        // (batch, 1)
    net_output: DeviceBuffer<f32>,    // (batch, 1)

    // ---- backward activation-grad workspace ----
    dy: DeviceBuffer<f32>,                    // (batch, 1) = dl3_out = dl1_skip
    dl2_acted: DeviceBuffer<f32>,              // (batch, l2_out)
    dl2_out: DeviceBuffer<f32>,                // (batch, l2_out)
    dl2_input: DeviceBuffer<f32>,              // (batch, l2_in)
    dl2_pre: DeviceBuffer<f32>,                // (batch, l2_in)
    dl1_sqr: DeviceBuffer<f32>,                // (batch, l1_eff)
    dl1_main_from_concat: DeviceBuffer<f32>,   // (batch, l1_eff)
    dl1_main_from_sqr: DeviceBuffer<f32>,      // (batch, l1_eff)
    dl1_main: DeviceBuffer<f32>,               // (batch, l1_eff)
    dl1_total: DeviceBuffer<f32>,              // (batch, l1_out)
}

impl Bucket {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: &CudaStream,
        batch_size: usize,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert!(l1_out >= 2, "l1_out must be >= 2 (1 skip + >=1 main), got {l1_out}");
        assert!(l2_out >= 1, "l2_out must be >= 1, got {l2_out}");
        let l1_eff = l1_out - L1_SKIP;
        let l2_in = l2_in_for(l1_out);
        debug_assert_eq!(l2_in, l1_eff * 2);
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(stream, n.max(1)).map_err(Into::into)
        };
        let w = |n: usize, fan_in: usize, s: u64| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::from_host(stream, &xavier_init(n, fan_in, s)).map_err(Into::into)
        };
        let l1_w_n = ft_out * l1_out;
        let l2_w_n = l2_in * l2_out;
        let l3_w_n = l2_out;
        Ok(Self {
            l1_out,
            l2_out,
            l2_in,
            l1_eff,
            l1_w: w(l1_w_n, ft_out, seed ^ 0x11)?,
            l1_w_m: z(l1_w_n)?,
            l1_w_v: z(l1_w_n)?,
            l1_w_slow: z(l1_w_n)?,
            l1_w_grad: z(l1_w_n)?,
            l1_b: z(l1_out)?,
            l1_b_m: z(l1_out)?,
            l1_b_v: z(l1_out)?,
            l1_b_slow: z(l1_out)?,
            l1_b_grad: z(l1_out)?,
            l2_w: w(l2_w_n, l2_in, seed ^ 0x22)?,
            l2_w_m: z(l2_w_n)?,
            l2_w_v: z(l2_w_n)?,
            l2_w_slow: z(l2_w_n)?,
            l2_w_grad: z(l2_w_n)?,
            l2_b: z(l2_out)?,
            l2_b_m: z(l2_out)?,
            l2_b_v: z(l2_out)?,
            l2_b_slow: z(l2_out)?,
            l2_b_grad: z(l2_out)?,
            l3_w: w(l3_w_n, l2_out, seed ^ 0x33)?,
            l3_w_m: z(l3_w_n)?,
            l3_w_v: z(l3_w_n)?,
            l3_w_slow: z(l3_w_n)?,
            l3_w_grad: z(l3_w_n)?,
            l3_b: z(1)?,
            l3_b_m: z(1)?,
            l3_b_v: z(1)?,
            l3_b_slow: z(1)?,
            l3_b_grad: z(1)?,
            l1_pre: z(batch_size * l1_out)?,
            l1_main: z(batch_size * l1_eff)?,
            l1_skip: z(batch_size * L1_SKIP)?,
            l1_sqr: z(batch_size * l1_eff)?,
            l2_pre: z(batch_size * l2_in)?,
            l2_input: z(batch_size * l2_in)?,
            l2_dense_out: z(batch_size * l2_out)?,
            l2_acted: z(batch_size * l2_out)?,
            l3_out: z(batch_size)?,
            net_output: z(batch_size)?,
            dy: z(batch_size)?,
            dl2_acted: z(batch_size * l2_out)?,
            dl2_out: z(batch_size * l2_out)?,
            dl2_input: z(batch_size * l2_in)?,
            dl2_pre: z(batch_size * l2_in)?,
            dl1_sqr: z(batch_size * l1_eff)?,
            dl1_main_from_concat: z(batch_size * l1_eff)?,
            dl1_main_from_sqr: z(batch_size * l1_eff)?,
            dl1_main: z(batch_size * l1_eff)?,
            dl1_total: z(batch_size * l1_out)?,
        })
    }
}

/// 決定論的な xavier-uniform 初期化 (`±1/sqrt(fan_in)` の一様分布)。
/// `nnue_train::init` には依存せず、xorshift64 PRNG で自己完結させる
/// (bucket ごとに違う `seed` を渡すことで層間の相関を避ける)。
fn xavier_init(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    let limit = 1.0_f64 / (fan_in.max(1) as f64).sqrt();
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    if state == 0 {
        state = 0xD1B54A32D192ED03;
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545F4914F6CDD1D);
        let u = (r >> 11) as f64 / ((1u64 << 53) as f64); // [0, 1)
        out.push(((u * 2.0 - 1.0) * limit) as f32);
    }
    out
}

/// bucket ごとの host 側 row 数と、`combined_sorted`/`dcombined_sorted` 上の
/// (align なし) 開始 offset。GPU 側の `count_buckets` + `exclusive_scan_aligned
/// (align=1)` + `scatter_bucket_perm` と同じ prefix sum を host 側でも計算する
/// (cuBLAS の `m` 引数と launch config は host 側スカラなので、都度 GPU から
/// read back せずに済むよう host 側で独立に再計算する)。
fn bucket_counts_and_offsets(bucket_idx: &[i32]) -> ([usize; N], [usize; N]) {
    let mut counts = [0usize; N];
    for &b in bucket_idx {
        if b >= 0 && (b as usize) < N {
            counts[b as usize] += 1;
        }
    }
    let mut offsets = [0usize; N];
    let mut acc = 0usize;
    for g in 0..N {
        offsets[g] = acc;
        acc += counts[g];
    }
    (counts, offsets)
}

pub(crate) struct LayerStackV3GpuTrainer {
    stream: std::sync::Arc<CudaStream>,
    module: std::sync::Arc<CudaModule>,
    cublas: CublasHandle,

    pub(crate) feature_set: FeatureSetSpec,
    ft_out: usize,

    // FT (shared)
    ft_w: DeviceBuffer<f32>,
    ft_w_m: DeviceBuffer<f32>,
    ft_w_v: DeviceBuffer<f32>,
    ft_w_slow: DeviceBuffer<f32>,
    ft_w_grad: DeviceBuffer<f32>,
    ft_b: DeviceBuffer<f32>,
    ft_b_m: DeviceBuffer<f32>,
    ft_b_v: DeviceBuffer<f32>,
    ft_b_slow: DeviceBuffer<f32>,
    ft_b_grad: DeviceBuffer<f32>,

    buckets: Vec<Bucket>, // len == N, buckets[g].{l1_out,l2_out} may differ per g

    // ---- FT-related shared workspace (batch_size 分、bucket 非依存) ----
    batch_size: usize,
    ft_in: usize,
    max_active: usize,
    ft_stm_out: DeviceBuffer<f32>,
    ft_nstm_out: DeviceBuffer<f32>,
    combined: DeviceBuffer<f32>,        // (batch, ft_out), original order
    combined_sorted: DeviceBuffer<f32>, // (batch, ft_out), bucket-sorted (align=1)
    dcombined: DeviceBuffer<f32>,          // (batch, ft_out), original order
    dcombined_sorted: DeviceBuffer<f32>,   // (batch, ft_out), bucket-sorted
    dcombined_zero: DeviceBuffer<f32>,     // (batch, ft_out), 常に 0 (L1f 不在の代替)
    dft_stm_out: DeviceBuffer<f32>,
    dft_nstm_out: DeviceBuffer<f32>,
    net_output: DeviceBuffer<f32>,    // (batch,), original order
    dy_net_output: DeviceBuffer<f32>, // (batch,), original order

    // FT inverse-index backward scratch
    feat_counts: DeviceBuffer<u32>,
    feat_offsets: DeviceBuffer<u32>,
    feat_block_sums: DeviceBuffer<u32>,
    feat_block_offsets: DeviceBuffer<u32>,
    feat_write_ctr: DeviceBuffer<u32>,
    feat_positions: DeviceBuffer<u32>,

    // input buffers (H2D)
    stm_idx_dev: DeviceBuffer<i32>,
    nstm_idx_dev: DeviceBuffer<i32>,
    bucket_idx_dev: DeviceBuffer<i32>,
    score_dev: DeviceBuffer<f32>,
    wdl_dev: DeviceBuffer<f32>,

    // bucket sort scratch (align=1、padding無し)
    bucket_counts_dev: DeviceBuffer<u32>,
    bucket_offsets_dev: DeviceBuffer<u32>,
    bucket_write_ctr_dev: DeviceBuffer<u32>,
    bucket_perm_dev: DeviceBuffer<i32>,       // len == batch_size
    bucket_idx_sorted_dev: DeviceBuffer<i32>, // len == batch_size (未使用だが scatter_bucket_perm の出力として確保)

    loss_acc: DeviceBuffer<f64>,
    weight_sum_acc: DeviceBuffer<f64>,
    step_count: u64,
    weight_decay: f32,
    /// `step()` が forward/backward より前に保存する scheduled lr。全 optimizer
    /// group で共通の lr を使う (per-group lr_mult 分離は V1 実装ではやらない)。
    pending_lr: f32,
    /// step 先頭の入力 H2D 用 ring (`InputUploadRing`)。本 trainer は
    /// double-buffer (active/back) の swap は行わず、`step()` が forward
    /// → backward → optimizer を完全に同期実行する (次 step の H2D が前 step の
    /// compute と物理 buffer を取り合うことは無い) ため、swap 無しでも安全。
    input_ring: InputUploadRing,
    ft_w_n: usize,
    ft_b_n: usize,

/// `Bucket::new` に渡す per-bucket `(l1_out, l2_out)` 一覧 + 共通 `ft_out`。
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerStackV3Dims {
    pub(crate) ft_out: usize,
    pub(crate) l1_out: [usize; N],
    pub(crate) l2_out: [usize; N],
}

impl LayerStackV3GpuTrainer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch_size: usize,
        dims: LayerStackV3Dims,
        feature_set: FeatureSetSpec,
        weight_decay: f32,
        tf32: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert!(
            dims.ft_out > 0 && dims.ft_out.is_multiple_of(128),
            "ft_out must be a positive multiple of 128, got {}",
            dims.ft_out
        );
        assert!(batch_size > 0, "batch_size must be > 0");
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(ctx, "nnue_train")?;
        let ft_in = feature_set.ft_in();
        let max_active = feature_set.max_active();
        let ft_out = dims.ft_out;

        let ft_w_n = feature_set.train_ft_in() * ft_out;
        let ft_b_n = ft_out;
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(&stream, n.max(1)).map_err(Into::into)
        };

        let mut buckets = Vec::with_capacity(N);
        for g in 0..N {
            buckets.push(Bucket::new(
                &stream,
                batch_size,
                ft_out,
                dims.l1_out[g],
                dims.l2_out[g],
                0xB000_0000_0000_0000 ^ (g as u64),
            )?);
        }

        let trainer = Self {
            stream: stream.clone(),
            module,
            cublas: CublasHandle::new(&stream, tf32)?,
            feature_set,
            ft_out,
            ft_w: DeviceBuffer::from_host(&stream, &xavier_init(ft_w_n, ft_in, 0xF7))?,
            ft_w_m: z(ft_w_n)?,
            ft_w_v: z(ft_w_n)?,
            ft_w_slow: z(ft_w_n)?,
            ft_w_grad: z(ft_w_n)?,
            ft_b: z(ft_b_n)?,
            ft_b_m: z(ft_b_n)?,
            ft_b_v: z(ft_b_n)?,
            ft_b_slow: z(ft_b_n)?,
            ft_b_grad: z(ft_b_n)?,
            buckets,
            batch_size,
            ft_in,
            max_active,
            ft_stm_out: z(batch_size * ft_out)?,
            ft_nstm_out: z(batch_size * ft_out)?,
            combined: z(batch_size * ft_out)?,
            combined_sorted: z(batch_size * ft_out)?,
            dcombined: z(batch_size * ft_out)?,
            dcombined_sorted: z(batch_size * ft_out)?,
            dcombined_zero: z(batch_size * ft_out)?,
            dft_stm_out: z(batch_size * ft_out)?,
            dft_nstm_out: z(batch_size * ft_out)?,
            net_output: z(batch_size)?,
            dy_net_output: z(batch_size)?,
            feat_counts: DeviceBuffer::<u32>::zeroed(&stream, ft_in)?,
            feat_offsets: DeviceBuffer::<u32>::zeroed(&stream, ft_in + 1)?,
            feat_block_sums: DeviceBuffer::<u32>::zeroed(&stream, ft_in.div_ceil(1024))?,
            feat_block_offsets: DeviceBuffer::<u32>::zeroed(&stream, ft_in.div_ceil(1024) + 1)?,
            feat_write_ctr: DeviceBuffer::<u32>::zeroed(&stream, ft_in)?,
            feat_positions: DeviceBuffer::<u32>::zeroed(&stream, batch_size * max_active)?,
            stm_idx_dev: DeviceBuffer::<i32>::zeroed(&stream, batch_size * max_active)?,
            nstm_idx_dev: DeviceBuffer::<i32>::zeroed(&stream, batch_size * max_active)?,
            bucket_idx_dev: DeviceBuffer::<i32>::zeroed(&stream, batch_size)?,
            score_dev: DeviceBuffer::<f32>::zeroed(&stream, batch_size)?,
            wdl_dev: DeviceBuffer::<f32>::zeroed(&stream, batch_size)?,
            bucket_counts_dev: DeviceBuffer::<u32>::zeroed(&stream, N + 1)?,
            bucket_offsets_dev: DeviceBuffer::<u32>::zeroed(&stream, N + 1)?,
            bucket_write_ctr_dev: DeviceBuffer::<u32>::zeroed(&stream, N + 1)?,
            bucket_perm_dev: DeviceBuffer::<i32>::zeroed(&stream, batch_size)?,
            bucket_idx_sorted_dev: DeviceBuffer::<i32>::zeroed(&stream, batch_size)?,
            loss_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            weight_sum_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            step_count: 0,
            weight_decay,
            pending_lr: 0.0,
            input_ring: InputUploadRing::new(ctx, batch_size, max_active)?,
            ft_w_n,
            ft_b_n,
        };
        Ok(trainer)
    }

    fn l1_out_of(&self, g: usize) -> usize {
        self.buckets[g].l1_out
    }
    fn l2_out_of(&self, g: usize) -> usize {
        self.buckets[g].l2_out
    }

    /// 1 batch 分の forward のみ (loss kernel まで、backward/optimizer 無し)。
    /// `net_output` (device, original order) と `loss_acc` を書く。
    fn forward(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        let b_u32 = b as u32;
        let ft_out = self.ft_out;
        let ft_out_u32 = ft_out as u32;

        self.input_ring.upload(
            &self.stream,
            &self.stm_idx_dev,
            batch.stm_indices,
            &self.nstm_idx_dev,
            batch.nstm_indices,
            &self.bucket_idx_dev,
            batch.bucket_idx,
            &self.score_dev,
            batch.score,
            &self.wdl_dev,
            batch.wdl,
        )?;

        memset_zero(&self.stream, &self.loss_acc)?;
        memset_zero(&self.stream, &self.weight_sum_acc)?;

        // -- FT forward (shared, V2 と同一) --
        cuda_launch! {
            kernel: sparse_ft_forward, stream: self.stream, module: self.module,
            config: cfg_1d(b * ft_out / 4),
            args: [slice(self.ft_w), slice(self.stm_idx_dev), slice_mut(self.ft_stm_out),
                   b_u32, ft_out_u32, self.ft_in as u32, self.max_active as u32]
        }?;
        cuda_launch! {
            kernel: sparse_ft_forward, stream: self.stream, module: self.module,
            config: cfg_1d(b * ft_out / 4),
            args: [slice(self.ft_w), slice(self.nstm_idx_dev), slice_mut(self.ft_nstm_out),
                   b_u32, ft_out_u32, self.ft_in as u32, self.max_active as u32]
        }?;
        cuda_launch! {
            kernel: ft_post_perspective_fwd, stream: self.stream, module: self.module,
            config: cfg_1d(b * ft_out),
            args: [slice(self.ft_stm_out), slice(self.ft_nstm_out), slice(self.ft_b),
                   slice_mut(self.combined), b_u32, ft_out_u32, FT_POST_SCALE]
        }?;

        // -- bucket sort (align=1、padding無し): combined を bucket_idx でソートし
        //    combined_sorted を作る。GPU 側 histogram/scan/scatter は V2 と同一
        //    kernel を再利用する (align 引数だけ 1)。
        memset_zero(&self.stream, &self.bucket_counts_dev)?;
        memset_zero(&self.stream, &self.bucket_write_ctr_dev)?;
        memset_minus_one_i32(&self.stream, &self.bucket_perm_dev)?;
        memset_minus_one_i32(&self.stream, &self.bucket_idx_sorted_dev)?;
        cuda_launch! {
            kernel: count_buckets, stream: self.stream, module: self.module,
            config: cfg_1d(b),
            args: [slice(self.bucket_idx_dev), slice(self.bucket_counts_dev), b_u32, N as u32]
        }?;
        cuda_launch! {
            kernel: exclusive_scan_aligned, stream: self.stream, module: self.module,
            config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
            args: [slice(self.bucket_counts_dev), slice(self.bucket_offsets_dev), (N + 1) as u32, 1_u32]
        }?;
        cuda_launch! {
            kernel: scatter_bucket_perm, stream: self.stream, module: self.module,
            config: cfg_1d(b),
            args: [slice(self.bucket_idx_dev), slice(self.bucket_offsets_dev),
                   slice(self.bucket_write_ctr_dev), slice(self.bucket_perm_dev),
                   slice(self.bucket_idx_sorted_dev), b_u32, N as u32]
        }?;
        cuda_launch! {
            kernel: permute_rows_f32, stream: self.stream, module: self.module,
            config: cfg_1d(b * ft_out),
            args: [slice(self.combined), slice(self.bucket_perm_dev),
                   slice_mut(self.combined_sorted), b_u32, ft_out_u32]
        }?;

        // host 側で同じ prefix sum を計算 (cuBLAS の m 引数 / kernel launch size に使う)。
        let (counts, offsets) = bucket_counts_and_offsets(batch.bucket_idx);

        for g in 0..N {
            let cnt = counts[g];
            if cnt == 0 {
                continue;
            }
            let cnt_u32 = cnt as u32;
            let l1_out = self.l1_out_of(g);
            let l2_out = self.l2_out_of(g);
            let l1_eff = self.buckets[g].l1_eff;
            let l2_in = self.buckets[g].l2_in;
            let byte_off = (offsets[g] * ft_out * std::mem::size_of::<f32>()) as u64;

            // SAFETY: `combined_sorted` は cudaMalloc 由来、`batch_size * ft_out`
            // 要素確保済 (`offsets[g] + cnt <= b <= batch_size` は
            // `bucket_counts_and_offsets` の prefix sum が保証)。offset した
            // 読み取り range `[offsets[g]*ft_out, (offsets[g]+cnt)*ft_out)` は
            // 確保範囲内に収まる。`cu_deviceptr()` は driver 管理の device
            // pointer (整数値)、+byte_off のポインタ演算は同一 allocation 内な
            // ので有効。`self.cublas` は `self.stream` に bind 済で同 stream 上
            // in-order 実行 (直前の `permute_rows_f32` 完了後に読む)。
            let combined_g_ptr =
                unsafe { (self.combined_sorted.cu_deviceptr() + byte_off) as *const f32 };

            unsafe {
                self.cublas.sgemm_fwd_rowmajor(
                    cnt_u32 as i32,
                    l1_out as i32,
                    ft_out as i32,
                    combined_g_ptr,
                    self.buckets[g].l1_w.cu_deviceptr() as *const f32,
                    self.buckets[g].l1_pre.cu_deviceptr() as *mut f32,
                )?;
            }
            cuda_launch! {
                kernel: bias_add_per_row, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_out),
                args: [slice(self.buckets[g].l1_b), slice_mut(self.buckets[g].l1_pre), cnt_u32, l1_out as u32]
            }?;
            cuda_launch! {
                kernel: slice_extract_2d, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_eff),
                args: [slice(self.buckets[g].l1_pre), slice_mut(self.buckets[g].l1_main),
                       cnt_u32, l1_out as u32, 0_u32, l1_eff as u32]
            }?;
            cuda_launch! {
                kernel: slice_extract_2d, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * L1_SKIP),
                args: [slice(self.buckets[g].l1_pre), slice_mut(self.buckets[g].l1_skip),
                       cnt_u32, l1_out as u32, l1_eff as u32, L1_SKIP as u32]
            }?;
            cuda_launch! {
                kernel: abs_pow2_scale_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_eff),
                args: [slice(self.buckets[g].l1_main), slice_mut(self.buckets[g].l1_sqr),
                       L1_SQR_SCALE, (cnt * l1_eff) as u32]
            }?;
            cuda_launch! {
                kernel: concat_l1sqr_main_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l2_in),
                args: [slice(self.buckets[g].l1_sqr), slice(self.buckets[g].l1_main),
                       slice_mut(self.buckets[g].l2_pre), cnt_u32, l1_eff as u32, l1_eff as u32]
            }?;
            cuda_launch! {
                kernel: crelu_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l2_in),
                args: [slice(self.buckets[g].l2_pre), slice_mut(self.buckets[g].l2_input), (cnt * l2_in) as u32]
            }?;

            unsafe {
                self.cublas.sgemm_fwd_rowmajor(
                    cnt_u32 as i32,
                    l2_out as i32,
                    l2_in as i32,
                    self.buckets[g].l2_input.cu_deviceptr() as *const f32,
                    self.buckets[g].l2_w.cu_deviceptr() as *const f32,
                    self.buckets[g].l2_dense_out.cu_deviceptr() as *mut f32,
                )?;
            }
            cuda_launch! {
                kernel: bias_add_per_row, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l2_out),
                args: [slice(self.buckets[g].l2_b), slice_mut(self.buckets[g].l2_dense_out), cnt_u32, l2_out as u32]
            }?;
            cuda_launch! {
                kernel: crelu_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l2_out),
                args: [slice(self.buckets[g].l2_dense_out), slice_mut(self.buckets[g].l2_acted), (cnt * l2_out) as u32]
            }?;

            unsafe {
                self.cublas.sgemm_fwd_rowmajor(
                    cnt_u32 as i32,
                    1_i32,
                    l2_out as i32,
                    self.buckets[g].l2_acted.cu_deviceptr() as *const f32,
                    self.buckets[g].l3_w.cu_deviceptr() as *const f32,
                    self.buckets[g].l3_out.cu_deviceptr() as *mut f32,
                )?;
            }
            cuda_launch! {
                kernel: bias_add_per_row, stream: self.stream, module: self.module,
                config: cfg_1d(cnt),
                args: [slice(self.buckets[g].l3_b), slice_mut(self.buckets[g].l3_out), cnt_u32, 1_u32]
            }?;
            cuda_launch! {
                kernel: elementwise_add, stream: self.stream, module: self.module,
                config: cfg_1d(cnt),
                args: [slice(self.buckets[g].l3_out), slice(self.buckets[g].l1_skip),
                       slice_mut(self.buckets[g].net_output), cnt_u32]
            }?;
            cuda_launch! {
                kernel: scatter_by_perm_offset, stream: self.stream, module: self.module,
                config: cfg_1d(cnt),
                args: [slice(self.buckets[g].net_output), slice(self.bucket_perm_dev),
                       offsets[g] as u32, slice(self.net_output), cnt_u32, 1_u32]
            }?;
        }

        // -- loss (共有、original order、V2 と同一) --
        match loss {
            LossKind::Sigmoid { scale } => {
                cuda_launch! {
                    kernel: loss_wdl, stream: self.stream, module: self.module,
                    config: cfg_1d(b),
                    args: [slice(self.net_output), slice(self.score_dev), slice(self.wdl_dev),
                           batch.per_pos_norm, slice_mut(self.dy_net_output), slice(self.loss_acc),
                           wdl_lambda, scale, b_u32]
                }?;
            }
            LossKind::Wrm { .. } => {
                // WRM loss はこの v1 実装ではまだ配線していない (`loss_wrm` /
                // `wrm_weight_sum` kernel の正確な引数列を this codebase の
                // 実装で確認しないまま配線するリスクを避けるため、意図的に
                // 未対応としてここで明示 error にする)。Sigmoid loss
                // (`--win-rate-model` を指定しない既定経路) のみ対応。
                return Err(
                    "layerstack_v3 trainer does not yet support --win-rate-model \
                     (WRM) loss; only the default sigmoid-MSE loss is wired. \
                     This is a known v1 gap, not a silent fallback."
                        .into(),
                );
            }
        }
        Ok(())
    }

    fn backward(&mut self, batch: &BatchData) -> Result<(), Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        let ft_out = self.ft_out;
        let ft_out_u32 = ft_out as u32;

        memset_zero(&self.stream, &self.ft_b_grad)?;
        memset_zero(&self.stream, &self.dcombined_sorted)?;
        for g in 0..N {
            memset_zero(&self.stream, &self.buckets[g].l1_w_grad)?;
            memset_zero(&self.stream, &self.buckets[g].l1_b_grad)?;
            memset_zero(&self.stream, &self.buckets[g].l2_w_grad)?;
            memset_zero(&self.stream, &self.buckets[g].l2_b_grad)?;
            memset_zero(&self.stream, &self.buckets[g].l3_w_grad)?;
            memset_zero(&self.stream, &self.buckets[g].l3_b_grad)?;
        }

        let (counts, offsets) = bucket_counts_and_offsets(batch.bucket_idx);

        for g in 0..N {
            let cnt = counts[g];
            if cnt == 0 {
                continue;
            }
            let cnt_u32 = cnt as u32;
            let l1_out = self.l1_out_of(g);
            let l2_out = self.l2_out_of(g);
            let l1_eff = self.buckets[g].l1_eff;
            let l2_in = self.buckets[g].l2_in;
            let byte_off = (offsets[g] * ft_out * std::mem::size_of::<f32>()) as u64;

            cuda_launch! {
                kernel: gather_by_perm_offset, stream: self.stream, module: self.module,
                config: cfg_1d(cnt),
                args: [slice(self.dy_net_output), slice(self.bucket_perm_dev), offsets[g] as u32,
                       slice_mut(self.buckets[g].dy), cnt_u32, 1_u32]
            }?;

            // L3 backward: dl2_acted (bwd input), l3_w_grad (bwd weight), l3_b_grad
            unsafe {
                self.cublas.sgemm_x_yt_rowmajor(
                    cnt_u32 as i32, l2_out as i32, 1_i32,
                    self.buckets[g].dy.cu_deviceptr() as *const f32,
                    self.buckets[g].l3_w.cu_deviceptr() as *const f32,
                    self.buckets[g].dl2_acted.cu_deviceptr() as *mut f32,
                )?;
            }
            unsafe {
                self.cublas.sgemm_xt_y_rowmajor(
                    l2_out as i32, 1_i32, cnt_u32 as i32,
                    self.buckets[g].l2_acted.cu_deviceptr() as *const f32,
                    self.buckets[g].dy.cu_deviceptr() as *const f32,
                    self.buckets[g].l3_w_grad.cu_deviceptr() as *mut f32,
                )?;
            }
            cuda_launch! {
                kernel: bias_grad, stream: self.stream, module: self.module,
                config: cfg_1d(cnt),
                args: [slice(self.buckets[g].dy), slice(self.buckets[g].l3_b_grad), cnt_u32, 1_u32]
            }?;

            cuda_launch! {
                kernel: crelu_grad, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l2_out),
                args: [slice(self.buckets[g].l2_dense_out), slice(self.buckets[g].dl2_acted),
                       slice_mut(self.buckets[g].dl2_out), (cnt * l2_out) as u32]
            }?;

            // L2 backward
            unsafe {
                self.cublas.sgemm_x_yt_rowmajor(
                    cnt_u32 as i32, l2_in as i32, l2_out as i32,
                    self.buckets[g].dl2_out.cu_deviceptr() as *const f32,
                    self.buckets[g].l2_w.cu_deviceptr() as *const f32,
                    self.buckets[g].dl2_input.cu_deviceptr() as *mut f32,
                )?;
            }
            unsafe {
                self.cublas.sgemm_xt_y_rowmajor(
                    l2_in as i32, l2_out as i32, cnt_u32 as i32,
                    self.buckets[g].l2_input.cu_deviceptr() as *const f32,
                    self.buckets[g].dl2_out.cu_deviceptr() as *const f32,
                    self.buckets[g].l2_w_grad.cu_deviceptr() as *mut f32,
                )?;
            }
            cuda_launch! {
                kernel: bias_grad, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l2_out),
                args: [slice(self.buckets[g].dl2_out), slice(self.buckets[g].l2_b_grad), cnt_u32, l2_out as u32]
            }?;

            cuda_launch! {
                kernel: crelu_grad, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l2_in),
                args: [slice(self.buckets[g].l2_pre), slice(self.buckets[g].dl2_input),
                       slice_mut(self.buckets[g].dl2_pre), (cnt * l2_in) as u32]
            }?;
            cuda_launch! {
                kernel: concat_l1sqr_main_grad, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_eff),
                args: [slice(self.buckets[g].dl2_pre), slice_mut(self.buckets[g].dl1_sqr),
                       slice_mut(self.buckets[g].dl1_main_from_concat), cnt_u32, l1_eff as u32]
            }?;
            cuda_launch! {
                kernel: abs_pow2_scale_grad, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_eff),
                args: [slice(self.buckets[g].l1_main), slice(self.buckets[g].dl1_sqr),
                       slice_mut(self.buckets[g].dl1_main_from_sqr), L1_SQR_SCALE, (cnt * l1_eff) as u32]
            }?;
            cuda_launch! {
                kernel: elementwise_add, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_eff),
                args: [slice(self.buckets[g].dl1_main_from_concat), slice(self.buckets[g].dl1_main_from_sqr),
                       slice_mut(self.buckets[g].dl1_main), (cnt * l1_eff) as u32]
            }?;
            memset_zero(&self.stream, &self.buckets[g].dl1_total)?;
            cuda_launch! {
                kernel: slice_scatter_2d, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_eff),
                args: [slice(self.buckets[g].dl1_main), slice_mut(self.buckets[g].dl1_total),
                       cnt_u32, l1_eff as u32, l1_out as u32, 0_u32]
            }?;
            cuda_launch! {
                kernel: slice_scatter_2d, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * L1_SKIP),
                args: [slice(self.buckets[g].dy), slice_mut(self.buckets[g].dl1_total),
                       cnt_u32, L1_SKIP as u32, l1_out as u32, l1_eff as u32]
            }?;

            // L1 backward: dcombined_sorted (bucket segment、bwd input), l1_w_grad (bwd weight), l1_b_grad
            let combined_g_ptr =
                unsafe { (self.combined_sorted.cu_deviceptr() + byte_off) as *const f32 };
            let dcombined_g_ptr =
                unsafe { (self.dcombined_sorted.cu_deviceptr() + byte_off) as *mut f32 };
            // SAFETY: `combined_g_ptr`/`dcombined_g_ptr` は forward の同じ offset
            // 計算と同じ理由で範囲内 (`offsets[g] + cnt <= b <= batch_size`)。
            // `dcombined_sorted` は本関数冒頭で 0 memset 済で、他 bucket の
            // segment は書かないため、bucket 間で書き込みが競合しない
            // (disjoint な byte range)。
            unsafe {
                self.cublas.sgemm_x_yt_rowmajor(
                    cnt_u32 as i32, ft_out as i32, l1_out as i32,
                    self.buckets[g].dl1_total.cu_deviceptr() as *const f32,
                    self.buckets[g].l1_w.cu_deviceptr() as *const f32,
                    dcombined_g_ptr,
                )?;
            }
            unsafe {
                self.cublas.sgemm_xt_y_rowmajor(
                    ft_out as i32, l1_out as i32, cnt_u32 as i32,
                    combined_g_ptr,
                    self.buckets[g].dl1_total.cu_deviceptr() as *const f32,
                    self.buckets[g].l1_w_grad.cu_deviceptr() as *mut f32,
                )?;
            }
            cuda_launch! {
                kernel: bias_grad, stream: self.stream, module: self.module,
                config: cfg_1d(cnt * l1_out),
                args: [slice(self.buckets[g].dl1_total), slice(self.buckets[g].l1_b_grad), cnt_u32, l1_out as u32]
            }?;
        }

        // -- dcombined_sorted (bucket-sorted) -> dcombined (original order) --
        cuda_launch! {
            kernel: inverse_permute_rows_f32, stream: self.stream, module: self.module,
            config: cfg_1d(b * ft_out),
            args: [slice(self.dcombined_sorted), slice(self.bucket_perm_dev),
                   slice(self.dcombined), b as u32, ft_out_u32]
        }?;

        // -- FT backward (shared、V2 の fused kernel を L1f 不在の代わりに
        //    dcombined_zero (常に0) を第2入力として渡して再利用する) --
        let b_u32 = b as u32;
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused, stream: self.stream, module: self.module,
            config: cfg_1d(b * ft_out / 2),
            args: [slice(self.dcombined), slice(self.dcombined_zero), slice(self.ft_stm_out),
                   slice(self.ft_b), slice_mut(self.dft_stm_out), slice(self.ft_b_grad),
                   b_u32, ft_out_u32, 0_u32, ft_out_u32, FT_POST_SCALE]
        }?;
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused, stream: self.stream, module: self.module,
            config: cfg_1d(b * ft_out / 2),
            args: [slice(self.dcombined), slice(self.dcombined_zero), slice(self.ft_nstm_out),
                   slice(self.ft_b), slice_mut(self.dft_nstm_out), slice(self.ft_b_grad),
                   b_u32, ft_out_u32, (ft_out / 2) as u32, ft_out_u32, FT_POST_SCALE]
        }?;

        let ft_in = self.ft_in;
        let max_active = self.max_active;
        for (iter_idx, idx_dev) in [&self.stm_idx_dev, &self.nstm_idx_dev].into_iter().enumerate() {
            memset_zero(&self.stream, &self.feat_counts)?;
            memset_zero(&self.stream, &self.feat_write_ctr)?;
            cuda_launch! {
                kernel: build_feature_counts, stream: self.stream, module: self.module,
                config: cfg_1d(b * max_active),
                args: [slice(idx_dev), slice(self.feat_counts), b_u32, max_active as u32, ft_in as u32]
            }?;
            let prefix_blocks = ft_in.div_ceil(1024) as u32;
            cuda_launch! {
                kernel: prefix_sum_block_local, stream: self.stream, module: self.module,
                config: LaunchConfig { grid_dim: (prefix_blocks, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(self.feat_counts), slice(self.feat_offsets), slice(self.feat_block_sums), ft_in as u32]
            }?;
            cuda_launch! {
                kernel: exclusive_prefix_sum_small, stream: self.stream, module: self.module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(self.feat_block_sums), slice(self.feat_block_offsets), prefix_blocks]
            }?;
            cuda_launch! {
                kernel: prefix_sum_add_block_offset, stream: self.stream, module: self.module,
                config: LaunchConfig { grid_dim: (prefix_blocks, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(self.feat_offsets), slice(self.feat_block_offsets), ft_in as u32, prefix_blocks]
            }?;
            cuda_launch! {
                kernel: scatter_positions, stream: self.stream, module: self.module,
                config: cfg_1d(b * max_active),
                args: [slice(idx_dev), slice(self.feat_offsets), slice(self.feat_write_ctr),
                       slice(self.feat_positions), b_u32, max_active as u32, ft_in as u32]
            }?;
            let d_config = LaunchConfig {
                grid_dim: (ft_in as u32, (ft_out / 128) as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            if iter_idx == 0 {
                cuda_launch! {
                    kernel: gather_and_sum_per_feature_overwrite, stream: self.stream, module: self.module,
                    config: d_config,
                    args: [slice(self.dft_stm_out), slice(self.feat_positions), slice(self.feat_offsets),
                           slice(self.ft_w_grad), ft_in as u32, ft_out_u32]
                }?;
            } else {
                cuda_launch! {
                    kernel: gather_and_sum_per_feature_add, stream: self.stream, module: self.module,
                    config: d_config,
                    args: [slice(self.dft_nstm_out), slice(self.feat_positions), slice(self.feat_offsets),
                           slice(self.ft_w_grad), ft_in as u32, ft_out_u32]
                }?;
            }
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.step_count += 1;
        let (step_size, denom) =
            radam_compute_step_size_denom(self.step_count, BETA1, BETA2, N_SMA_THRESHOLD);
        let wd = self.weight_decay;
        // 1 (ft_w) + 1 (ft_b) + N * 6 (l1_w/l1_b/l2_w/l2_b/l3_w/l3_b) group。
        // FT weight は他と同じ扱い (ft/dense/bias の param-group 分離は行わず、
        // 全 group 共通の `--weight-decay` を使う V1 実装)。
        let ft_w_n = self.ft_w_n;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
            args: [slice_mut(self.ft_w), slice_mut(self.ft_w_m), slice_mut(self.ft_w_v),
                   slice_mut(self.ft_w_grad), self.pending_lr, step_size, denom, wd,
                   BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX, ft_w_n as u32]
        }?;
        let ft_b_n = self.ft_b_n;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(ft_b_n),
            args: [slice_mut(self.ft_b), slice_mut(self.ft_b_m), slice_mut(self.ft_b_v),
                   slice_mut(self.ft_b_grad), self.pending_lr, step_size, denom, wd,
                   BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX, ft_b_n as u32]
        }?;
        let ft_out = self.ft_out;
        for g in 0..N {
            let l1_out = self.buckets[g].l1_out;
            let l2_out = self.buckets[g].l2_out;
            let l2_in = self.buckets[g].l2_in;
            macro_rules! step_group {
                ($w:ident, $m:ident, $v:ident, $grad:ident, $n:expr, $clamp_min:expr, $clamp_max:expr) => {{
                    let n: usize = $n;
                    cuda_launch! {
                        kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(n),
                        args: [slice_mut(self.buckets[g].$w), slice_mut(self.buckets[g].$m),
                               slice_mut(self.buckets[g].$v), slice_mut(self.buckets[g].$grad),
                               self.pending_lr, step_size, denom, wd, BETA1, BETA2, EPS,
                               $clamp_min, $clamp_max, n as u32]
                    }?;
                }};
            }
            step_group!(l1_w, l1_w_m, l1_w_v, l1_w_grad, ft_out * l1_out, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX);
            step_group!(l1_b, l1_b_m, l1_b_v, l1_b_grad, l1_out, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX);
            step_group!(l2_w, l2_w_m, l2_w_v, l2_w_grad, l2_in * l2_out, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX);
            step_group!(l2_b, l2_b_m, l2_b_v, l2_b_grad, l2_out, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX);
            step_group!(l3_w, l3_w_m, l3_w_v, l3_w_grad, l2_out, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX);
            step_group!(l3_b, l3_b_m, l3_b_v, l3_b_grad, 1_usize, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX);
        }
        Ok(())
    }

    /// 1 batch 分の学習 step (forward + loss + backward + Ranger step)。
    /// `Σ err²` (position 数で割る前) を返す。
    pub(crate) fn step(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(0.0);
        }
        if b > self.batch_size {
            return Err(format!(
                "batch {b} exceeds workspace capacity {} (fixed at construction time)",
                self.batch_size
            )
            .into());
        }
        self.pending_lr = lr;
        self.forward(batch, wdl_lambda, loss)?;
        self.backward(batch)?;
        // `InputUploadRing`::upload の 2-slot pinned buffer 使い回し契約を満たすため、
        // この step の入力 buffer を読む compute (forward/backward) が enqueue され
        // 終わった時点で記録する (次+2 step の H2D がこの event を待つ)。
        self.input_ring.mark_step_done(&self.stream)?;
        self.optimizer_step()?;
        let loss_val = self.loss_acc.to_host_vec(&self.stream)?[0];
        Ok(loss_val)
    }

    /// forward + loss のみ (held-out validation)。
    pub(crate) fn validate(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(StepOutput { loss: 0.0, net_output: Vec::new() });
        }
        self.forward(batch, wdl_lambda, loss)?;
        let loss_val = self.loss_acc.to_host_vec(&self.stream)?[0];
        let mut net_output = self.net_output.to_host_vec(&self.stream)?;
        net_output.truncate(b);
        Ok(StepOutput { loss: loss_val, net_output })
    }

    /// 現在の weight を [`LayerStackV3Weights`] (host, f32) へ download する。
    pub(crate) fn to_layerstack_v3_weights(
        &self,
    ) -> Result<LayerStackV3Weights, Box<dyn std::error::Error>> {
        let mut l1_out = [0usize; N];
        let mut l2_out = [0usize; N];
        for g in 0..N {
            l1_out[g] = self.buckets[g].l1_out;
            l2_out[g] = self.buckets[g].l2_out;
        }
        let mut w = LayerStackV3Weights::zeroed(self.feature_set, self.ft_out, l1_out, l2_out);
        w.ft_w = self.ft_w.to_host_vec(&self.stream)?;
        w.ft_w.truncate(self.feature_set.ft_in() * self.ft_out);
        w.ft_b = self.ft_b.to_host_vec(&self.stream)?;
        for g in 0..N {
            w.l1_w[g] = self.buckets[g].l1_w.to_host_vec(&self.stream)?;
            w.l1_b[g] = self.buckets[g].l1_b.to_host_vec(&self.stream)?;
            w.l2_w[g] = self.buckets[g].l2_w.to_host_vec(&self.stream)?;
            w.l2_b[g] = self.buckets[g].l2_b.to_host_vec(&self.stream)?;
            w.l3_w[g] = self.buckets[g].l3_w.to_host_vec(&self.stream)?;
            w.l3_b[g] = self.buckets[g].l3_b.to_host_vec(&self.stream)?[0];
        }
        Ok(w)
    }

    /// resume 用の raw f32 checkpoint (簡易独自形式、量子化 `.bin` とは別物)。
    /// weight + Ranger state (`m`/`v`/`slow`) + `step_count` を書く。
    pub(crate) fn save_raw_checkpoint(
        &mut self,
        path: &Path,
        superbatch: usize,
        _run_id: &str,
        _lr_horizon: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("ckpt.tmp");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(b"TLV3CKPT")?;
        f.write_all(&(superbatch as u64).to_le_bytes())?;
        f.write_all(&self.step_count.to_le_bytes())?;
        let mut write_vec = |buf: &DeviceBuffer<f32>, out: &mut std::io::BufWriter<std::fs::File>| -> Result<(), Box<dyn std::error::Error>> {
            let v = buf.to_host_vec(&self.stream)?;
            out.write_all(&(v.len() as u64).to_le_bytes())?;
            for x in &v {
                out.write_all(&x.to_le_bytes())?;
            }
            Ok(())
        };
        write_vec(&self.ft_w, &mut f)?;
        write_vec(&self.ft_w_m, &mut f)?;
        write_vec(&self.ft_w_v, &mut f)?;
        write_vec(&self.ft_w_slow, &mut f)?;
        write_vec(&self.ft_b, &mut f)?;
        write_vec(&self.ft_b_m, &mut f)?;
        write_vec(&self.ft_b_v, &mut f)?;
        write_vec(&self.ft_b_slow, &mut f)?;
        for g in 0..N {
            write_vec(&self.buckets[g].l1_w, &mut f)?;
            write_vec(&self.buckets[g].l1_w_m, &mut f)?;
            write_vec(&self.buckets[g].l1_w_v, &mut f)?;
            write_vec(&self.buckets[g].l1_w_slow, &mut f)?;
            write_vec(&self.buckets[g].l1_b, &mut f)?;
            write_vec(&self.buckets[g].l1_b_m, &mut f)?;
            write_vec(&self.buckets[g].l1_b_v, &mut f)?;
            write_vec(&self.buckets[g].l1_b_slow, &mut f)?;
            write_vec(&self.buckets[g].l2_w, &mut f)?;
            write_vec(&self.buckets[g].l2_w_m, &mut f)?;
            write_vec(&self.buckets[g].l2_w_v, &mut f)?;
            write_vec(&self.buckets[g].l2_w_slow, &mut f)?;
            write_vec(&self.buckets[g].l2_b, &mut f)?;
            write_vec(&self.buckets[g].l2_b_m, &mut f)?;
            write_vec(&self.buckets[g].l2_b_v, &mut f)?;
            write_vec(&self.buckets[g].l2_b_slow, &mut f)?;
            write_vec(&self.buckets[g].l3_w, &mut f)?;
            write_vec(&self.buckets[g].l3_w_m, &mut f)?;
            write_vec(&self.buckets[g].l3_w_v, &mut f)?;
            write_vec(&self.buckets[g].l3_w_slow, &mut f)?;
            write_vec(&self.buckets[g].l3_b, &mut f)?;
            write_vec(&self.buckets[g].l3_b_m, &mut f)?;
            write_vec(&self.buckets[g].l3_b_v, &mut f)?;
            write_vec(&self.buckets[g].l3_b_slow, &mut f)?;
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

// ===========================================================================
// TrainerBackend impl — `nnue-train::trainer::run` から 1 batch ずつ呼ばれる
// ===========================================================================
//
// `trainer_common::trainer_backend_impl!` macro (GpuTrainer / SimpleGpuTrainer が
// 使うもの) は `self.loss_ring` (`AsyncLossRing`、非同期 loss readback) を前提に
// `flush_pending_loss` を実装する。本 trainer の `step()` は `loss_acc` を
// **毎 step 同期 (`to_host_vec`) で読み戻す** (async pipeline を持たない、V1 の
// 意図的な単純化) ため、`loss_ring` field 自体が無い。そのためこの macro は使わず、
// `TrainerBackend::flush_pending_loss` の default 実装 (`Ok(0.0)`, つまり「溜まって
// いる未報告 loss は無い」) をそのまま使う手動 impl にする。
impl nnue_train::trainer::TrainerBackend for LayerStackV3GpuTrainer {
    fn train_step(
        &mut self,
        batch: &nnue_train::dataloader::Batch,
        bucket_idx: &[i32],
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<f64> {
        if batch.feature_set != self.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.feature_set.canonical_name(),
            )));
        }
        let data = BatchData::from_batch_ref(batch, bucket_idx);
        self.step(&data, lr, wdl_lambda, loss)
            .map_err(|e| std::io::Error::other(format!("LayerStackV3GpuTrainer::step failed: {e}")))
    }

    fn validate_step(
        &mut self,
        batch: &nnue_train::dataloader::Batch,
        bucket_idx: &[i32],
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<nnue_train::trainer::ValidationStepOutput> {
        if batch.feature_set != self.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.feature_set.canonical_name(),
            )));
        }
        let data = BatchData::from_batch_ref(batch, bucket_idx);
        let out = self.validate(&data, wdl_lambda, loss).map_err(|e| {
            std::io::Error::other(format!("LayerStackV3GpuTrainer::validate failed: {e}"))
        })?;
        Ok(nnue_train::trainer::ValidationStepOutput {
            sum_sq_err: out.loss,
            net_output: out.net_output,
        })
    }

    fn save_checkpoint(&mut self, path: &Path) -> std::io::Result<()> {
        let weights = self.to_layerstack_v3_weights().map_err(|e| {
            std::io::Error::other(format!(
                "LayerStackV3GpuTrainer::to_layerstack_v3_weights failed: {e}"
            ))
        })?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        weights.save_quantised(&mut writer)?;
        std::io::Write::flush(&mut writer)?;
        Ok(())
    }

    fn save_resume_checkpoint(
        &mut self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
        lr_horizon: Option<usize>,
    ) -> std::io::Result<()> {
        self.save_raw_checkpoint(path, superbatch, run_id, lr_horizon)
            .map_err(|e| match e.downcast::<std::io::Error>() {
                Ok(io_err) => *io_err,
                Err(other) => std::io::Error::other(format!(
                    "LayerStackV3GpuTrainer::save_raw_checkpoint failed: {other}"
                )),
            })
    }
    // `flush_pending_loss` / `read_fp16_clamp_count` は trait の default (どちらも
    // 事実上 no-op / 0) を使う。
}

