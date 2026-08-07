//! `router` bucket mode — 学習可能なバケット選択ネットワーク。
//!
//! `progress_kpabs` (`progress8kpabs`) は固定 (frozen) の重みで
//! KP-absolute 特徴の重み和を取り、sigmoid を通して進行度を求める
//! (`81 * FE_OLD_END = 125,388` 個の重みのみ、bias は無い、`progress.bin` は
//! f64 LE で保存)。対して本 module の `RouterKPAbs` は、**`progress8kpabs` と
//! 全く同じ計算方法 (重み和のみ、bias 無し、f64 精度) を N 出力
//! (N = `--num-buckets` で指定するバケット数、任意) に拡張したもの**——
//! progress8kpabs の重みテーブルを N 個並べ、それぞれの重み和を N-way softmax
//! の logit として扱う多クラス分類器:
//!
//! ```text
//! logits[k] = Σ_i w[i][k]   (i は active な KP-absolute index、k は 0..N、bias 無し)
//! bucket = argmax_k logits[k]
//! ```
//!
//! 言語モデルの MoE のように、評価関数本体 (LayerStack の FT/L1/L2/L3) と
//! **ランダム初期化から一緒に学習**する。YaneuraOu 側 (`RouterIndex`) は、この
//! 重み和を progress8kpabs と全く同じ計算方法 (Q16.16/Q8.8/Q4.4 固定小数点の
//! 重み和) で N 回 (バケットごとに 1 回ずつ) 行い、argmax でバケットを選ぶ。
//!
//! ## 学習方式 (hard-EM / soft-EM / Top-K Hard Routing)
//!
//! GPU 側の LayerStack 学習ループ (`bins/nnue_train::trainer_layerstack`) は
//! per-position の `bucket_idx` を学習開始前に host 側で確定させる設計になって
//! おり、CUDA kernel 内で router の勾配を直接引き戻す経路を持たない。そのため
//! 本 router は次の EM 系の手法 (Jacobs & Jordan 1991 の Adaptive Mixtures of
//! Local Experts と同じ発想) で学習する:
//!
//! 1. (E step) 同じ batch を N 通りの固定 bucket 割当それぞれで forward し
//!    (`GpuTrainer::validate` を bucket = 0..N で N 回呼ぶだけで、新規 kernel
//!    は不要)、position ごとに **N 個すべての bucket の誤差**を求める。
//! 2. `oracle_targets_from_errors` で、その誤差から oracle ターゲット分布を
//!    作る。`--top-k` (`K`) によって 3 通りに振る舞いが変わる:
//!    - `K = 1` (既定、従来の hard-EM): 誤差最小の 1 bucket だけに one-hot。
//!    - `K = num_buckets`: 全 bucket を `softmax(-err)` で重み付けする
//!      **soft-EM** (Jacobs & Jordan の responsibility そのもの)。
//!    - `1 < K < num_buckets`: 誤差が小さい方から上位 K 個だけを残し、その中で
//!      `softmax(-err)` を取る **Top-K Hard Routing**。推論時 (`RouterIndex`)
//!      は argmax = 常に 1 bucket しか選ばないので、学習側だけが K 候補の
//!      相対的な良さに応じた確率で "hedge" する形になる。
//! 3. (M step) `RouterKPAbs::train_oracle_batch` で、そのターゲット分布との
//!    (soft-label) N-class cross entropy (+ 負荷分散補助損失) を本 module 内の
//!    CPU 実装で backprop し、Adam で router の重みを更新する。
//!
//! YaneuraOu 側の推論 (`RouterIndex`) は `--top-k` の値に関わらず常に argmax
//! (Top-1) で 1 bucket だけを選ぶ — Top-K/soft routing は探索を遅くするだけで
//! 恩恵が無いため、学習時だけの技法として `--top-k` を用意している。
//!
//! 呼び出し側 (`bins/nnue_train`) は毎 step (または `--router-refresh-interval`
//! 間隔) でこれを行い、評価関数本体の GPU 学習と router の CPU 学習を交互に
//! 進める。router の重みは学習開始時にランダム初期化され (`init_random`)、以後
//! 評価関数と共に更新される。
//!
//! ## プロセス全体で 1 個
//!
//! `progress_kpabs::ShogiProgressKPAbs` と同様、重みはプロセス global
//! (`OnceLock<RwLock<..>>`) に置く。dataloader の複数 worker スレッドは
//! `read()` でバケット割当のためだけに触り、学習ループのメインスレッドが
//! `train_oracle_batch` で `write()` して重みを更新する。

use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::{OnceLock, RwLock};

use shogi_format::{PackedSfenValue, ShogiBoard};

use crate::progress_kpabs::{SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs};

/// raw 保存形式のマジックナンバー (`.bin` 先頭 4 byte)。`progress.bin` と違い
/// ヘッダ付きにして、次元違いの取り違えを防ぐ。
const ROUTER_BIN_MAGIC: u32 = 0x526f_396b; // "Ro9k" 由来 (バケット数 9 固定だった頃の名残、値はそのまま維持)

/// router (bias 無しの重み和 × N 出力) の重み一式。`progress.bin` と同じ f64
/// 精度。
///
/// レイアウト: `w` は index-major、`w[idx * num_buckets + k]` (`idx` は
/// KP-absolute 特徴 index、`k` はバケット/クラス番号 `[0, num_buckets)`)。
/// `progress_kpabs` の重みテーブル (`[SQ_NB][fe_end] -> f64` 1 個、bias 無し)
/// を `num_buckets` 個並べたのと同じ形。
#[derive(Clone, Debug)]
pub struct RouterKPAbsWeights {
    pub num_buckets: usize,
    pub w: Vec<f64>,
}

/// `RouterKPAbsWeights::train_oracle_batch` の戻り値。ログ用の診断情報一式。
#[derive(Clone, Debug)]
pub struct RouterTrainStats {
    /// oracle ラベルとの N-class cross entropy (batch 平均)。
    pub cross_entropy_loss: f64,
    /// 負荷分散補助損失 (Switch Transformer 式、batch 平均)。`balance_weight`
    /// 込みではない生の値 (`N * Σ_i f_i * P_i`) なので、`0.0` に近いほど N
    /// バケットの使用率が均等 (最小値 `1.0`、N バケット均等使用時)。
    pub balance_loss: f64,
    /// このバッチで router 自身の argmax (= 実際に dispatch されるバケット) が
    /// 各バケットに落ちた比率 (`f_i`、長さ `num_buckets`、合計 1.0)。偏りの
    /// 直接的な監視用。
    pub bucket_usage: Vec<f64>,
}

/// `RouterKPAbsWeights` と同 shape・同精度 (f64) の Adam optimizer state。
#[derive(Clone, Debug)]
pub struct RouterAdamState {
    num_buckets: usize,
    m_w: Vec<f64>,
    v_w: Vec<f64>,
    t: u64,
}

impl RouterAdamState {
    pub fn zeros(num_buckets: usize) -> Self {
        Self {
            num_buckets,
            m_w: vec![0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * num_buckets],
            v_w: vec![0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * num_buckets],
            t: 0,
        }
    }

    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

/// 決定論的な xorshift64* (`rand` crate 非依存、smoke_dummy 等の既存流儀と同じ)。
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// `[-bound, bound]` の一様乱数。
    fn next_f64_symmetric(&mut self, bound: f64) -> f64 {
        let bits = self.next_u64() >> 11; // 53-bit (f64 mantissa 精度)
        let unit = bits as f64 / (1u64 << 53) as f64; // [0, 1)
        (unit * 2.0 - 1.0) * bound
    }
}

impl RouterKPAbsWeights {
    /// 全 0 初期化 (`logits` は常に 0、argmax は bucket 0 に固定される。
    /// テスト用の decidable baseline)。
    pub fn zeroed(num_buckets: usize) -> Self {
        assert!(num_buckets > 0, "router num_buckets must be >= 1");
        Self {
            num_buckets,
            w: vec![0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * num_buckets],
        }
    }

    /// ランダム初期化 (`言語モデルの MoE のように、ランダム化されたバケット選択
    /// ネットワークから始める`)。小さい一様乱数。
    pub fn random(num_buckets: usize, seed: u64) -> Self {
        assert!(num_buckets > 0, "router num_buckets must be >= 1");
        let mut rng = XorShift64::new(seed);
        // fan_in が非常に大きい (125,388) sparse 層なので、実際に効くのは
        // 1 position あたりの active index 数 (~76) 程度。それを目安に小さめの
        // bound を取る (progress_kpabs の frozen model 相当のスケール感)。
        let bound = 1.0_f64 / 76.0_f64.sqrt();
        let mut w = vec![0.0_f64; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * num_buckets];
        for v in w.iter_mut() {
            *v = rng.next_f64_symmetric(bound);
        }
        Self { num_buckets, w }
    }

    /// 指定 active index 集合から N 個の logit (= 重み和、bias 無し) を forward
    /// する。`indices` は
    /// `progress_kpabs::ShogiProgressKPAbs::for_each_active_index_board` が emit
    /// する `[0, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS)` の index 列。戻り値の長さは
    /// `self.num_buckets`。
    pub fn forward_logits(&self, indices: &[u32]) -> Vec<f64> {
        let mut logits = vec![0.0_f64; self.num_buckets];
        for &idx in indices {
            let base = idx as usize * self.num_buckets;
            debug_assert!(base + self.num_buckets <= self.w.len(), "router index out of range");
            for (l, &w) in logits.iter_mut().zip(&self.w[base..base + self.num_buckets]) {
                *l += w;
            }
        }
        logits
    }

    /// 局面から直接 forward する (`ShogiBoard` 版)。
    pub fn forward_board(&self, board: &ShogiBoard) -> Vec<f64> {
        let mut indices = Vec::new();
        ShogiProgressKPAbs::for_each_active_index_board(board, |idx| indices.push(idx as u32));
        self.forward_logits(&indices)
    }

    /// N-way softmax + argmax。`(bucket, gate = softmax(logits)[bucket], probs)`
    /// を返す。`gate` は現状 engine 側では使わない (engine は重み和の argmax の
    /// みで選ぶ。softmax は単調変換なので argmax は重み和そのものの argmax と
    /// 一致し、N 出力すべてに同じスケールが掛かっているため sigmoid/softmax を
    /// 経由せず直接比較してよい)。
    pub fn bucket_and_probs(logits: &[f64]) -> (u32, f64, Vec<f64>) {
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut exps = vec![0.0_f64; logits.len()];
        let mut sum = 0.0_f64;
        for (e, &l) in exps.iter_mut().zip(logits.iter()) {
            *e = (l - max_logit).exp();
            sum += *e;
        }
        let mut probs = vec![0.0_f64; logits.len()];
        let mut best_k = 0usize;
        let mut best_p = f64::NEG_INFINITY;
        for (k, e) in exps.iter().enumerate() {
            let p = e / sum;
            probs[k] = p;
            if p > best_p {
                best_p = p;
                best_k = k;
            }
        }
        (best_k as u32, best_p, probs)
    }

    /// 局面 1 つ分の bucket (`0..num_buckets`) を返す。
    pub fn bucket_from_indices(&self, indices: &[u32]) -> u32 {
        let logits = self.forward_logits(indices);
        Self::bucket_and_probs(&logits).0
    }

    /// `bucket_from_indices` の `ShogiBoard` 版。
    pub fn bucket_board(&self, board: &ShogiBoard) -> u32 {
        let logits = self.forward_board(board);
        Self::bucket_and_probs(&logits).0
    }

    /// oracle ターゲット分布 (E step で求める、詳細は
    /// [`crate::router_kpabs`] モジュール doc の "Soft-EM / Top-K Hard
    /// Routing" を参照) との N-class cross entropy に加え、Switch Transformer
    /// 式の負荷分散補助損失を足して 1 batch 分 backprop し、Adam で重みを
    /// 更新する。
    ///
    /// `indices_batch[i]` は position `i` の active index 列、
    /// `oracle_targets[i]` はその position の教師分布 (長さ `num_buckets`、
    /// 各要素 `>= 0` かつ合計 `1.0` — one-hot (hard-EM/`--top-k 1`) でも
    /// 密な分布 (soft-EM/`--top-k num-buckets`) でもよい)。両者は同じ長さで
    /// なければならない。
    ///
    /// ## 負荷分散補助損失
    ///
    /// hard-EM の cross entropy だけだと、あるバケットの評価関数がまだ弱い間は
    /// argmin-loss (oracle) に選ばれにくくなり → そのバケットへの学習データが
    /// 減り → さらに弱くなる、という偏り (expert collapse) が起こりうる。これを
    /// 抑えるため、Switch Transformer (Fedus et al. 2021) と同じ形の補助損失を
    /// 加える:
    ///
    /// ```text
    /// L_balance = N · Σ_i f_i · P_i
    /// ```
    ///
    /// - `N` = バケット数 (`self.num_buckets`)
    /// - `f_i` = この batch で router 自身の argmax (= 実際に推論で dispatch
    ///   されるバケット) が `i` になった position の割合 (勾配を流さない定数
    ///   扱い、Switch Transformer 論文と同じ stop-gradient)
    /// - `P_i` = この batch での softmax 確率 `P(bucket=i)` の平均 (微分可能)
    ///
    /// `f_i = P_i = 1/N` (完全に均等) のとき最小値 `1.0` を取り、偏るほど増える。
    /// `balance_weight` (`λ`) でこの項の強さを調整する (`0.0` で無効化、
    /// cross entropy のみの従来動作に戻る)。
    pub fn train_oracle_batch(
        &mut self,
        indices_batch: &[Vec<u32>],
        oracle_targets: &[Vec<f64>],
        adam: &mut RouterAdamState,
        lr: f64,
        weight_decay: f64,
        balance_weight: f64,
    ) -> RouterTrainStats {
        assert_eq!(indices_batch.len(), oracle_targets.len());
        assert_eq!(
            adam.num_buckets, self.num_buckets,
            "RouterAdamState num_buckets mismatch with RouterKPAbsWeights"
        );
        let n = indices_batch.len();
        let num_buckets = self.num_buckets;
        if n == 0 {
            return RouterTrainStats {
                cross_entropy_loss: 0.0,
                balance_loss: 0.0,
                bucket_usage: vec![0.0; num_buckets],
            };
        }

        // 1st pass: forward しつつ probs をキャッシュし、router 自身の argmax
        // (= 実際の dispatch 先) の分布 f_i を求める。f_i は 2nd pass の勾配計算
        // (`d_logits` に足す balance 項) で全 position 共通に使うため、先に
        // batch 全体を見ておく必要がある。
        let mut all_probs: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut dispatch_count = vec![0u32; num_buckets];
        for indices in indices_batch {
            let logits = self.forward_logits(indices);
            let (dispatch_bucket, _, probs) = Self::bucket_and_probs(&logits);
            dispatch_count[dispatch_bucket as usize] += 1;
            all_probs.push(probs);
        }
        let mut f = vec![0.0_f64; num_buckets];
        for k in 0..num_buckets {
            f[k] = dispatch_count[k] as f64 / n as f64;
        }

        // 2nd pass: cross entropy (soft target 対応) + (balance_weight != 0 なら)
        // 負荷分散項の勾配を合成して backprop する。
        let mut grad_w = vec![0.0_f64; self.w.len()];
        let mut total_ce_loss = 0.0_f64;
        let mut avg_probs = vec![0.0_f64; num_buckets];

        for (indices, (target, probs)) in indices_batch.iter().zip(oracle_targets.iter().zip(all_probs.iter())) {
            debug_assert_eq!(target.len(), num_buckets);
            // H(target, probs) = -Σ_k target_k * ln(probs_k) (soft-label cross
            // entropy、one-hot ならいつもの `-ln(probs[target_bucket])` に一致)。
            for k in 0..num_buckets {
                if target[k] != 0.0 {
                    total_ce_loss -= target[k] * probs[k].max(1e-15).ln();
                }
                avg_probs[k] += probs[k];
            }

            // d(cross_entropy)/d(logits) = softmax(logits) - target
            // (target が one-hot なら従来通り、soft/sparse-top-k な分布でも
            // softmax cross entropy の勾配としてそのまま成り立つ標準的な結果)。
            let mut d_logits = probs.clone();
            for k in 0..num_buckets {
                d_logits[k] -= target[k];
            }

            if balance_weight != 0.0 {
                // L_balance の position ごとの寄与 (f は stop-gradient):
                //   loss_b = N * Σ_i f_i * probs_b[i]
                // softmax jacobian より
                //   d(loss_b)/d(logits_b[j]) = N * probs_b[j] * (f_j - Σ_i f_i * probs_b[i])
                let dot: f64 = f.iter().zip(probs.iter()).map(|(&fi, &pi)| fi * pi).sum();
                let n_buckets = num_buckets as f64;
                for j in 0..num_buckets {
                    d_logits[j] += balance_weight * n_buckets * probs[j] * (f[j] - dot);
                }
            }

            for &idx in indices {
                let base = idx as usize * num_buckets;
                for k in 0..num_buckets {
                    grad_w[base + k] += d_logits[k];
                }
            }
        }

        let inv_n = 1.0_f64 / n as f64;
        for g in grad_w.iter_mut() {
            *g *= inv_n;
        }
        for p in avg_probs.iter_mut() {
            *p *= inv_n;
        }

        adam.t += 1;
        adam_step(&mut self.w, &grad_w, &mut adam.m_w, &mut adam.v_w, lr, weight_decay, adam.t);

        let balance_loss: f64 =
            num_buckets as f64 * f.iter().zip(avg_probs.iter()).map(|(&fi, &pi)| fi * pi).sum::<f64>();

        RouterTrainStats {
            cross_entropy_loss: total_ce_loss / n as f64,
            balance_loss,
            bucket_usage: f,
        }
    }

    /// raw 形式で書き出す: magic(u32) / num_weights(u32) / num_buckets(u32) /
    /// w(f64 LE)。`progress.bin` と同じ f64 精度。progress8kpabs 同様 bias は
    /// 無いので、これだけで完結する。
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&ROUTER_BIN_MAGIC.to_le_bytes())?;
        writer.write_all(&(SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS as u32).to_le_bytes())?;
        writer.write_all(&(self.num_buckets as u32).to_le_bytes())?;
        write_f64_slice(writer, &self.w)?;
        Ok(())
    }

    /// `write_to` の逆。`num_weights` 不一致は `Err` (KP-absolute 特徴の次元は
    /// 常に固定なので)。`num_buckets` は書かれていた値をそのまま使う (呼び出し
    /// 側で期待値と照合すること、`RouterKPAbs::load_from_bin` 等は行わない —
    /// これは `bins/nnue_train` 側で `--num-buckets` と突き合わせる)。
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let magic = read_u32(reader)?;
        if magic != ROUTER_BIN_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("router weights magic mismatch: got {magic:#x}"),
            ));
        }
        let num_weights = read_u32(reader)? as usize;
        if num_weights != SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "router weights input dim mismatch: expected {SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS}, got {num_weights}"
                ),
            ));
        }
        let num_buckets = read_u32(reader)? as usize;
        if num_buckets == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "router weights num_buckets must be >= 1",
            ));
        }
        let w = read_f64_vec(reader, num_weights * num_buckets)?;
        Ok(Self { num_buckets, w })
    }
}

fn adam_step(
    w: &mut [f64],
    grad: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    lr: f64,
    weight_decay: f64,
    t: u64,
) {
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPS: f64 = 1e-8;
    let bias_correction1 = 1.0 - BETA1.powi(t as i32);
    let bias_correction2 = 1.0 - BETA2.powi(t as i32);
    for i in 0..w.len() {
        let g = grad[i] + weight_decay * w[i];
        m[i] = BETA1 * m[i] + (1.0 - BETA1) * g;
        v[i] = BETA2 * v[i] + (1.0 - BETA2) * g * g;
        let m_hat = m[i] / bias_correction1;
        let v_hat = v[i] / bias_correction2;
        w[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
    }
}

fn write_f64_slice<W: Write>(writer: &mut W, values: &[f64]) -> io::Result<()> {
    for &v in values {
        writer.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f64_vec<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 8];
    for _ in 0..n {
        reader.read_exact(&mut buf)?;
        out.push(f64::from_le_bytes(buf));
    }
    Ok(out)
}

static ROUTER_STATE: OnceLock<RwLock<RouterKPAbsWeights>> = OnceLock::new();

/// `router` bucket mode の process-global エントリポイント。
///
/// `progress_kpabs::ShogiProgressKPAbs` (frozen 単一 model) と違い、本 struct が
/// 触る重みは学習中に変化し続ける。dataloader の複数 worker は `bucket_board` /
/// `bucket_from_indices` 経由で読み取り専用アクセスし、学習ループのメインスレッド
/// だけが `train_oracle_batch` で書き込む (`RwLock`)。
#[derive(Clone, Copy, Default)]
pub struct RouterKPAbs;

impl RouterKPAbs {
    /// ランダム初期化して global へ設置する。学習開始時に (resume でない限り)
    /// 一度だけ呼ぶ。二回目以降の呼び出しは `Err`。
    pub fn init_random(num_buckets: usize, seed: u64) -> Result<(), String> {
        ROUTER_STATE
            .set(RwLock::new(RouterKPAbsWeights::random(num_buckets, seed)))
            .map_err(|_| "router weights are already initialized in this process".into())
    }

    /// 既存の重み (resume 用、raw checkpoint 由来) を global へ設置する。
    pub fn init_with_weights(weights: RouterKPAbsWeights) -> Result<(), String> {
        ROUTER_STATE
            .set(RwLock::new(weights))
            .map_err(|_| "router weights are already initialized in this process".into())
    }

    /// `.bin` (raw `RouterKPAbsWeights::write_to` 形式) から読み込んで global へ
    /// 設置する。
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open '{}': {e}", path.display()))?;
        let weights = RouterKPAbsWeights::read_from(&mut file)
            .map_err(|e| format!("failed to parse '{}': {e}", path.display()))?;
        Self::init_with_weights(weights)?;
        Ok(Self)
    }

    fn state() -> &'static RwLock<RouterKPAbsWeights> {
        ROUTER_STATE
            .get()
            .expect("RouterKPAbs used before init_random / init_with_weights / load_from_bin")
    }

    /// 現在の重みを `.bin` に保存する。
    pub fn save_to_bin(path: &Path) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        Self::state().read().unwrap().write_to(&mut file)
    }

    /// 現在の重みの clone (checkpoint への埋め込み用)。
    pub fn snapshot() -> RouterKPAbsWeights {
        Self::state().read().unwrap().clone()
    }

    /// [`Self::snapshot`] の non-panicking 版。`init_random` /
    /// `init_with_weights` / `load_from_bin` / `load_full` が一度も呼ばれて
    /// いなければ `None` を返す。`bucket_mode` を引き回さずに「router で
    /// 学習中かどうか」を export 経路 (`nnue-format` 書き出し) から判定するのに
    /// 使う (router 以外のモードではこの global は触られないため常に
    /// `None`)。
    pub fn try_snapshot() -> Option<RouterKPAbsWeights> {
        ROUTER_STATE.get().map(|s| s.read().unwrap().clone())
    }

    /// 現在の bucket 数。global が未初期化なら `None`。
    pub fn num_buckets() -> Option<usize> {
        ROUTER_STATE.get().map(|s| s.read().unwrap().num_buckets)
    }

    /// `--router-resume` sidecar 形式: `RouterKPAbsWeights::write_to` に続けて
    /// Adam optimizer state (同 shape) を書く。resume で optimizer 状態ごと
    /// router 学習を再開したいときに使う (`--resume` 本体の raw checkpoint とは
    /// 別ファイル — router は process-global の独立 state なので main net の
    /// raw checkpoint format には同居させていない)。
    pub fn save_full(path: &Path, adam: &RouterAdamState) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let weights = Self::snapshot();
        weights.write_to(&mut file)?;
        write_f64_slice(&mut file, &adam.m_w)?;
        write_f64_slice(&mut file, &adam.v_w)?;
        file.write_all(&adam.t.to_le_bytes())?;
        Ok(())
    }

    /// [`Self::save_full`] の逆。読み込んだ重みは global へ設置し、Adam state は
    /// 呼び出し側に返す。
    pub fn load_full(path: &Path) -> io::Result<RouterAdamState> {
        let mut file = std::fs::File::open(path)?;
        let weights = RouterKPAbsWeights::read_from(&mut file)?;
        let num_buckets = weights.num_buckets;
        Self::init_with_weights(weights).map_err(io::Error::other)?;
        let mut adam = RouterAdamState::zeros(num_buckets);
        adam.m_w = read_f64_vec(&mut file, adam.m_w.len())?;
        adam.v_w = read_f64_vec(&mut file, adam.v_w.len())?;
        let mut t_buf = [0u8; 8];
        file.read_exact(&mut t_buf)?;
        adam.t = u64::from_le_bytes(t_buf);
        Ok(adam)
    }

    /// bucket 決定 (`BucketMode` の `bucket_board` 契約と同じシグネチャ)。
    /// `num_buckets` はロード済み router の bucket 数と一致しないと panic する
    /// (呼び出し側 = LayerStack の `--num-buckets` と router の重みの次元が
    /// 食い違っている、設定ミス)。
    pub fn bucket_board(&self, board: &ShogiBoard, num_buckets: usize) -> u8 {
        let state = Self::state().read().unwrap();
        assert_eq!(
            num_buckets, state.num_buckets,
            "router bucket count mismatch: LayerStack --num-buckets={num_buckets} but loaded router \
             weights have {} buckets",
            state.num_buckets
        );
        state.bucket_board(board) as u8
    }

    /// legacy delegating path (`progress_kpabs` と同様、`decode()` を挟む版)。
    pub fn bucket(&self, pos: &PackedSfenValue, num_buckets: usize) -> u8 {
        self.bucket_board(&pos.decode(), num_buckets)
    }

    /// KP-absolute active index 列を求める (E step / M step の両方で使う)。
    pub fn active_indices_board(board: &ShogiBoard) -> Vec<u32> {
        let mut indices = Vec::new();
        ShogiProgressKPAbs::for_each_active_index_board(board, |idx| indices.push(idx as u32));
        indices
    }

    /// hard-EM/soft-EM/Top-K Hard Routing の M step。oracle ターゲット分布で
    /// 1 batch 分 backprop + Adam 更新する (負荷分散補助損失込み、
    /// `balance_weight` で強さを調整)。戻り値は診断用の `RouterTrainStats`
    /// (ログ用)。
    pub fn train_oracle_batch(
        indices_batch: &[Vec<u32>],
        oracle_targets: &[Vec<f64>],
        adam: &mut RouterAdamState,
        lr: f64,
        weight_decay: f64,
        balance_weight: f64,
    ) -> RouterTrainStats {
        let mut w = Self::state().write().unwrap();
        w.train_oracle_batch(indices_batch, oracle_targets, adam, lr, weight_decay, balance_weight)
    }
}

/// E step で求めた per-position・per-bucket の誤差 (`errs[position][bucket]`、
/// 小さいほど良い) から、hard-EM / soft-EM / Top-K Hard Routing いずれの
/// oracle ターゲット分布も作れる汎用関数。
///
/// - `top_k == 1`: 誤差最小の 1 bucket だけに重み `1.0` を置く one-hot
///   (= 従来の hard-EM そのもの)。
/// - `top_k == num_buckets`: 全 bucket を `softmax(-err)` で重み付けする
///   古典的な soft-EM (Jacobs & Jordan 1991 の responsibility)。
/// - それ以外: 誤差が小さい方から `top_k` 個だけを残し、その中で
///   `softmax(-err)` を取って残りを 0 にする (Top-K Hard Routing —
///   実際の推論 (`RouterIndex`) が argmax = 常に 1 bucket しか使わないのと
///   一貫させつつ、学習側では "router が候補として残す上位 top_k 個には
///   その相対的な良さに応じた確率で学習させる" という中間的な信号を与える)。
///
/// `err` はそのまま `-err` を logit とみなして softmax するので、絶対スケールは
/// 気にしなくてよい (`oracle_pred_scale`/`sigmoid` 由来の `[0,1]` 程度の二乗誤差
/// を想定)。
pub fn oracle_targets_from_errors(errs: &[f64], top_k: usize) -> Vec<f64> {
    let num_buckets = errs.len();
    assert!(top_k >= 1 && top_k <= num_buckets, "top_k must be in [1, num_buckets]");

    // 誤差昇順 (良い順) に bucket index を並べ、上位 top_k だけを残す。
    let mut order: Vec<usize> = (0..num_buckets).collect();
    order.sort_by(|&a, &b| errs[a].partial_cmp(&errs[b]).unwrap_or(std::cmp::Ordering::Equal));
    let selected = &order[..top_k];

    // softmax(-err) を選ばれた top_k の中だけで取る (top_k==1 なら自動的に
    // one-hot、softmax の分母がその 1 要素だけになるため)。
    let min_err = selected.iter().map(|&k| errs[k]).fold(f64::INFINITY, f64::min);
    let mut weights = vec![0.0_f64; num_buckets];
    let mut sum = 0.0_f64;
    for &k in selected {
        let w = (-(errs[k] - min_err)).exp(); // min_err を引いてから exp: オーバーフロー対策
        weights[k] = w;
        sum += w;
    }
    for &k in selected {
        weights[k] /= sum;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::Color;

    fn sample_board() -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            ..Default::default()
        };
        board.black_king_sq = shogi_format::types::Square::new(4, 8);
        board.white_king_sq = shogi_format::types::Square::new(4, 0);
        board
    }

    #[test]
    fn zeroed_weights_pick_bucket_zero() {
        let w = RouterKPAbsWeights::zeroed(9);
        let board = sample_board();
        // 全 logits = 0 → 最初の argmax (bucket 0) が選ばれる。
        assert_eq!(w.bucket_board(&board), 0);
    }

    #[test]
    fn random_weights_are_deterministic_for_seed() {
        let a = RouterKPAbsWeights::random(9, 42);
        let b = RouterKPAbsWeights::random(9, 42);
        assert_eq!(a.w, b.w);
    }

    #[test]
    fn raw_round_trip_preserves_weights() {
        let w = RouterKPAbsWeights::random(9, 7);
        let mut buf = Vec::new();
        w.write_to(&mut buf).unwrap();
        let reloaded = RouterKPAbsWeights::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(w.num_buckets, reloaded.num_buckets);
        assert_eq!(w.w, reloaded.w);
    }

    #[test]
    fn raw_round_trip_preserves_weights_for_arbitrary_bucket_count() {
        for num_buckets in [1usize, 4, 16, 32] {
            let w = RouterKPAbsWeights::random(num_buckets, 7);
            let mut buf = Vec::new();
            w.write_to(&mut buf).unwrap();
            let reloaded = RouterKPAbsWeights::read_from(&mut buf.as_slice()).unwrap();
            assert_eq!(reloaded.num_buckets, num_buckets);
            assert_eq!(w.w, reloaded.w);
        }
    }

    fn one_hot(num_buckets: usize, k: usize) -> Vec<f64> {
        let mut v = vec![0.0_f64; num_buckets];
        v[k] = 1.0;
        v
    }

    #[test]
    fn train_oracle_batch_reduces_loss_on_repeated_label() {
        let mut w = RouterKPAbsWeights::random(9, 1);
        let mut adam = RouterAdamState::zeros(9);
        let board = sample_board();
        let indices = RouterKPAbs::active_indices_board(&board);
        let batch = vec![indices.clone(), indices.clone(), indices];
        let target = vec![one_hot(9, 3), one_hot(9, 3), one_hot(9, 3)];

        let stats0 = w.train_oracle_batch(&batch, &target, &mut adam, 0.05, 0.0, 0.0);
        let mut last = stats0.cross_entropy_loss;
        for _ in 0..20 {
            last = w.train_oracle_batch(&batch, &target, &mut adam, 0.05, 0.0, 0.0).cross_entropy_loss;
        }
        assert!(
            last < stats0.cross_entropy_loss,
            "cross-entropy should decrease when repeatedly trained on a fixed label \
             (loss0={}, last={last})",
            stats0.cross_entropy_loss,
        );
    }

    #[test]
    fn balance_loss_is_minimized_when_usage_is_uniform() {
        // N バケットへの割り当て頻度 f が均等 (dispatch_count が全部同じ) なら
        // balance_loss は最小値 1.0 (= N * (1/N) * (1/N) の合計) に近づくはず。
        // 逆に 1 バケットへ完全に偏っていれば N に近づく。
        let w = RouterKPAbsWeights::random(9, 99);
        let mut adam = RouterAdamState::zeros(9);
        let board = sample_board();
        let indices = RouterKPAbs::active_indices_board(&board);
        // 同一局面を積んだだけの batch では router は常に同じ argmax を出すため
        // 完全に 1 バケットへ偏る (f はほぼ one-hot) のが期待値。
        let batch = vec![indices.clone(), indices.clone(), indices];
        let target = vec![one_hot(9, 0), one_hot(9, 1), one_hot(9, 2)];
        let mut w2 = w.clone();
        let stats = w2.train_oracle_batch(&batch, &target, &mut adam, 0.0, 0.0, 0.0);
        assert!(
            stats.balance_loss > 1.0,
            "fully skewed usage should have balance_loss > 1.0 (min at perfectly uniform usage), got {}",
            stats.balance_loss
        );
        assert!((stats.bucket_usage.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn oracle_targets_top_k_1_is_one_hot_argmin() {
        // top_k == 1 は従来の hard-EM (誤差最小 bucket に one-hot) と一致する。
        let errs = vec![0.5, 0.1, 0.9, 0.3];
        let target = oracle_targets_from_errors(&errs, 1);
        assert_eq!(target, one_hot(4, 1)); // index 1 が最小誤差 (0.1)
    }

    #[test]
    fn oracle_targets_top_k_num_buckets_is_full_soft_em() {
        // top_k == num_buckets は全 bucket が softmax(-err) で重み付けされる
        // (どれも 0 にならない、Jacobs & Jordan の soft-EM responsibility)。
        let errs = vec![0.5, 0.1, 0.9, 0.3];
        let target = oracle_targets_from_errors(&errs, errs.len());
        assert!(target.iter().all(|&w| w > 0.0), "soft-EM should give every bucket nonzero weight");
        assert!((target.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // 誤差が小さいほど重みが大きいはず (単調性)。
        assert!(target[1] > target[3] && target[3] > target[0] && target[0] > target[2]);
    }

    #[test]
    fn oracle_targets_intermediate_top_k_zeroes_out_the_rest() {
        let errs = vec![0.5, 0.1, 0.9, 0.3, 0.7];
        let target = oracle_targets_from_errors(&errs, 2);
        // 誤差最小の 2 個 (index 1: 0.1, index 3: 0.3) だけが非 0。
        assert!(target[1] > 0.0 && target[3] > 0.0);
        assert_eq!(target[0], 0.0);
        assert_eq!(target[2], 0.0);
        assert_eq!(target[4], 0.0);
        assert!((target.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "router bucket count mismatch")]
    fn bucket_board_rejects_wrong_num_buckets() {
        RouterKPAbs::init_with_weights(RouterKPAbsWeights::zeroed(9)).unwrap_or(()); // 既に他 test で init 済のことがあるので握り潰す
        let board = sample_board();
        let _ = RouterKPAbs.bucket_board(&board, 8);
    }
}
