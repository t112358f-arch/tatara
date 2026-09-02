//! `--router-arch ft-by-ft` — 評価net自身の Feature Transformer (FT) に
//! Router を統合する Router Architecture。
//!
//! `router_kpabs` (`--router-arch kpabs`, default) との違い:
//! `RouterKPAbsWeights` は KP-absolute 疎入力上の**完全に独立な**線形モデル
//! だが、本 module の router は **評価net本体の FT と全く同じ sparse
//! feature set/入力** (`Batch::stm_indices` / `nstm_indices`、
//! `feature_set.ft_in()`) 上の別の重み行を使う。
//!
//! ## 学習時は分離、保存時に結合 (spec.md 16節)
//!
//! > 1つのftにまとめる仕様はyaneuraouでの推論高速化のための仕様なので、
//! > tataraでは必ずしも1つのftにする必要はなく、評価関数を保存する段階で
//! > 結合する仕様にしてもよいです
//!
//! したがって本 module は、学習中は評価net本体の FT ([`crate::feature_set`]
//! 経由で GPU 学習される、既存コード完全不変) とは**別**の
//! `(ft_in, R)` 重み行列として router を CPU 上で学習し ([`RouterFtByFt`]
//! process-global、`RouterKPAbs` と同じ E step (GPU) / M step (CPU) 分割)、
//! [`combine_ft_by_ft_columns`] で **保存する瞬間にだけ** 評価net本体の FT
//! 重みと列方向に結合する。
//!
//! ## レイアウト
//!
//! `ft_out` = 評価net本体の FT 出力 (片視点、pairwise 後 = 従来通り L1 に
//! 渡る次元)、`R = sqrt(num_buckets)` (偶数) として、
//!
//! ```text
//! H = ft_out / 2   (pairwise 前、片視点の「前半/後半」それぞれの幅)
//! r = R / 2        (router 出力を前半・後半に均等分割した片方の幅)
//! ```
//!
//! 片視点の **activation 前** FT 出力 (`accum_out = ft_out + R`) を
//!
//! ```text
//! [ normal_a(H) | router_a(r) | normal_b(H) | router_b(r) ]
//! ```
//!
//! の順で並べる。既存の CReLU→pairwise (`ELEMENT_WISE_MULTIPLY`) は「前半
//! `[0, half)` × 後半 `[half, 2*half)`」の対応 index 同士の積 (`half =
//! accum_out/2 = H + r`) なので、
//!
//! - `j ∈ [0, H)`: `normal_a[j] * normal_b[j]` → 採用 (L1 入力、次元は
//!   `ft_out` のまま不変)
//! - `j ∈ [H, H+r)`: `router_a[j-H] * router_b[j-H]` → 破棄
//!
//! router の argmax は pairwise/CReLU を経由せず、**activation 前の生の
//! accumulator** から `router_a ++ router_b` (`R` 個) を直接読んで行う。この
//! 配置により、既存の FT accumulator 差分更新 (incremental update) は一切
//! 変更不要 — 単に幅が `ft_out` から `ft_out + R` に増えるだけで、従来コード
//! は列の意味を知らないまま正しく動作する。
//!
//! YaneuraOu 側の対応する実装は `nnue_feature_transformer.h` /
//! `nnue_architecture.h` / `evaluate_nnue.cpp` (`TANUKI_ROUTER_ARCH_FTBYFT`)
//! を参照。

use std::fmt;
use std::io::{self, Read, Write};
use std::sync::{OnceLock, RwLock};

/// `--router-arch` の選択肢。`bucket_mode == Router` のときのみ意味を持つ。
///
/// `Kpabs` (default) は [`crate::router_kpabs`] (FT とは独立な
/// KP-absolute 線形モデル)、`FtByFt` は本 module (評価net本体の FT に統合)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouterArch {
    #[default]
    Kpabs,
    FtByFt,
}

// ===========================================================================
// 1. レイアウト計算・検証
// ===========================================================================

/// `--router-arch ft-by-ft` の妥当性検証エラー。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FtByFtError {
    /// `num_buckets` が完全平方数でない。
    NotPerfectSquare { num_buckets: usize },
    /// `num_buckets` は完全平方数だが `sqrt(num_buckets)` が奇数。
    OddRoot { num_buckets: usize, root: usize },
    /// `--ft-out` が奇数 (pairwise の前提である偶数を満たさない)。
    OddFtOut { ft_out: usize },
}

impl fmt::Display for FtByFtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FtByFtError::NotPerfectSquare { num_buckets } => write!(
                f,
                "--router-arch ft-by-ft requires --num-buckets to be a perfect square \
                 (got --num-buckets={num_buckets}); ft-by-ft routes STM/NSTM through R = \
                 sqrt(num_buckets) FT outputs each and combines them as an R x R grid \
                 (bucket_index = stm_index * R + nstm_index), so num_buckets must equal \
                 R * R for some positive integer R (e.g. 4, 9, 16, 36, 64 -- not {num_buckets})"
            ),
            FtByFtError::OddRoot { num_buckets, root } => write!(
                f,
                "--router-arch ft-by-ft requires sqrt(--num-buckets) to be even \
                 (--num-buckets={num_buckets} gives R=sqrt({num_buckets})={root}, which is odd); \
                 this implementation places the R router outputs by splitting them evenly \
                 across the pre-pairwise first/second half of the per-perspective FT output \
                 (R/2 on each side), which requires R to be even -- pick a --num-buckets whose \
                 square root is even (e.g. 16, 64, 144 for R=4, 8, 12)"
            ),
            FtByFtError::OddFtOut { ft_out } => write!(
                f,
                "--router-arch ft-by-ft requires --ft-out to be even (got --ft-out={ft_out}); \
                 the per-perspective FT output is split into two equal halves before the \
                 pairwise-multiply step"
            ),
        }
    }
}

impl std::error::Error for FtByFtError {}

/// `ft-by-ft` router の確定した次元・レイアウト。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FtByFtLayout {
    /// `--num-buckets` (= `R * R`)。
    pub num_buckets: usize,
    /// `R = sqrt(num_buckets)`。STM/NSTM それぞれの router 出力数。
    pub r: usize,
    /// `--ft-out` (片視点、pairwise 後 = 従来通り L1 に渡る次元)。
    pub ft_out: usize,
    /// `H = ft_out / 2` (pairwise 前、前半/後半それぞれの通常 FT 幅)。
    pub half_normal: usize,
    /// `r_half = R / 2` (router 出力を前半/後半に割った片方の幅)。
    pub half_router: usize,
    /// 片視点の activation 前 FT 出力次元 (`ft_out + R`)。
    pub accum_out: usize,
}

impl FtByFtLayout {
    /// `num_buckets` / `ft_out` を検証し、`ft-by-ft` の次元・レイアウトを
    /// 確定する。
    pub fn new(num_buckets: usize, ft_out: usize) -> Result<Self, FtByFtError> {
        if ft_out % 2 != 0 {
            return Err(FtByFtError::OddFtOut { ft_out });
        }
        let r = (num_buckets as f64).sqrt().round() as usize;
        if r == 0 || r * r != num_buckets {
            return Err(FtByFtError::NotPerfectSquare { num_buckets });
        }
        if r % 2 != 0 {
            return Err(FtByFtError::OddRoot { num_buckets, root: r });
        }
        let half_normal = ft_out / 2;
        let half_router = r / 2;
        Ok(Self {
            num_buckets,
            r,
            ft_out,
            half_normal,
            half_router,
            accum_out: ft_out + r,
        })
    }

    /// 片視点、activation 前の raw FT 出力における column 区間
    /// `[start, start+len)` を `(normal_a, router_a, normal_b, router_b)` の
    /// 順で返す。
    pub fn column_ranges(&self) -> FtByFtColumnRanges {
        let h = self.half_normal;
        let r = self.half_router;
        FtByFtColumnRanges {
            normal_a: (0, h),
            router_a: (h, h + r),
            normal_b: (h + r, 2 * h + r),
            router_b: (2 * h + r, 2 * h + 2 * r),
        }
    }

    pub fn kept_pairwise_out(&self) -> usize {
        self.half_normal
    }

    pub fn discarded_pairwise_out(&self) -> usize {
        self.half_router
    }
}

/// [`FtByFtLayout::column_ranges`] の戻り値。各 tuple は half-open な
/// `(start, end)`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FtByFtColumnRanges {
    pub normal_a: (usize, usize),
    pub router_a: (usize, usize),
    pub normal_b: (usize, usize),
    pub router_b: (usize, usize),
}

/// 評価net本体の FT 重み (`normal_w: (ft_in, ft_out)` row-major、
/// `normal_w[feat * ft_out + out]`; `normal_b: (ft_out)`) と、別途学習した
/// router 重み (`router_w: (ft_in, R)` row-major、`router_b: (R)`) を、
/// [`FtByFtLayout`] の列レイアウトに従って **1つの結合 FT** (`(ft_in,
/// ft_out+R)` row-major、`(ft_out+R)`) に結合する。
///
/// 保存 (`save_yaneuraou` 相当) の直前にだけ呼ぶ想定 — 学習中は本関数を呼ぶ
/// 必要はない (spec.md 16節、module doc 参照)。
///
/// 各 feature 行について、`normal_w` の行を `[0,H)`/`[H,ft_out)` の 2 分割
/// にし、`router_w` の行を `[0,r)`/`[r,R)` の 2 分割にして、
/// `normal[0..H] ++ router[0..r] ++ normal[H..ft_out] ++ router[r..R]` の
/// 順で結合する (bias も同じ規則)。
pub fn combine_ft_by_ft_columns(
    normal_w: &[f32],
    normal_b: &[f32],
    router_w: &[f32],
    router_b: &[f32],
    ft_in: usize,
    layout: &FtByFtLayout,
) -> (Vec<f32>, Vec<f32>) {
    let ft_out = layout.ft_out;
    let r = layout.r;
    let h = layout.half_normal;
    let r_half = layout.half_router;
    assert_eq!(normal_w.len(), ft_in * ft_out, "normal_w shape mismatch");
    assert_eq!(normal_b.len(), ft_out, "normal_b shape mismatch");
    assert_eq!(router_w.len(), ft_in * r, "router_w shape mismatch");
    assert_eq!(router_b.len(), r, "router_b shape mismatch");

    let accum_out = layout.accum_out;
    let mut combined_w = vec![0.0f32; ft_in * accum_out];
    for feat in 0..ft_in {
        let n_row = &normal_w[feat * ft_out..feat * ft_out + ft_out];
        let r_row = &router_w[feat * r..feat * r + r];
        let dst = &mut combined_w[feat * accum_out..feat * accum_out + accum_out];
        dst[0..h].copy_from_slice(&n_row[0..h]);
        dst[h..h + r_half].copy_from_slice(&r_row[0..r_half]);
        dst[h + r_half..2 * h + r_half].copy_from_slice(&n_row[h..ft_out]);
        dst[2 * h + r_half..2 * h + 2 * r_half].copy_from_slice(&r_row[r_half..r]);
    }

    let mut combined_b = vec![0.0f32; accum_out];
    combined_b[0..h].copy_from_slice(&normal_b[0..h]);
    combined_b[h..h + r_half].copy_from_slice(&router_b[0..r_half]);
    combined_b[h + r_half..2 * h + r_half].copy_from_slice(&normal_b[h..ft_out]);
    combined_b[2 * h + r_half..2 * h + 2 * r_half].copy_from_slice(&router_b[r_half..r]);

    (combined_w, combined_b)
}

// ===========================================================================
// 2. Loss / 勾配 (reference)
// ===========================================================================

/// `ft-by-ft` router の学習時 loss/勾配。`P(i,j) = softmax(s)[i] *
/// softmax(n)[j]` という設計そのものが、任意の目的分布 `T(i,j)` に対する
/// cross entropy を、STM 側・NSTM 側で完全に独立な 2 本の softmax cross
/// entropy に厳密分解する:
///
/// ```text
/// dL/ds[i] = softmax(s)[i] - marg_stm(i)     (marg_stm(i) = sum_j T(i,j))
/// dL/dn[j] = softmax(n)[j] - marg_nstm(j)    (marg_nstm(j) = sum_i T(i,j))
/// ```
pub mod loss {
    /// 数値的に安定な softmax。
    pub fn softmax(scores: &[f64]) -> Vec<f64> {
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    /// `scores` 中の最大値の index (argmax)。同値がある場合は最初の index。
    pub fn argmax(scores: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for (i, &v) in scores.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx
    }

    /// STM/NSTM 生スコアから bucket index (`stm_idx * R + nstm_idx`) を選ぶ
    /// (推論時と同じ argmax ベースの選択; YaneuraOu
    /// `router_ftbyft_index_for_nnue` と同じ規約)。
    pub fn predict_bucket(stm_scores: &[f64], nstm_scores: &[f64]) -> usize {
        let r = stm_scores.len();
        debug_assert_eq!(r, nstm_scores.len());
        argmax(stm_scores) * r + argmax(nstm_scores)
    }

    /// Hard target (bucket index が one-hot で与えられる場合) の勾配。
    /// 戻り値は `(grad_stm, grad_nstm, loss)`。
    pub fn hard_target_grad(
        stm_scores: &[f64],
        nstm_scores: &[f64],
        target_stm_idx: usize,
        target_nstm_idx: usize,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let p_stm = softmax(stm_scores);
        let p_nstm = softmax(nstm_scores);
        let loss = -(p_stm[target_stm_idx].max(f64::MIN_POSITIVE).ln())
            - (p_nstm[target_nstm_idx].max(f64::MIN_POSITIVE).ln());
        let mut grad_stm = p_stm;
        grad_stm[target_stm_idx] -= 1.0;
        let mut grad_nstm = p_nstm;
        grad_nstm[target_nstm_idx] -= 1.0;
        (grad_stm, grad_nstm, loss)
    }

    /// Soft target (`R x R` の joint 目的分布 `target[i * R + j] = T(i,j)`,
    /// 総和 1) の勾配。周辺分布を計算してから [`hard_target_grad`] と同じ形
    /// の勾配を返す。`shogi_features::router_kpabs::oracle_targets_from_errors`
    /// (top-k soft-EM ターゲット、N = R*R に対して既存のまま使える) の出力
    /// をそのまま `target` に渡す想定。
    ///
    /// 戻り値は `(grad_stm, grad_nstm, loss)`。
    pub fn soft_target_grad(
        stm_scores: &[f64],
        nstm_scores: &[f64],
        target: &[f64],
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let r = stm_scores.len();
        debug_assert_eq!(r, nstm_scores.len());
        debug_assert_eq!(target.len(), r * r);

        let mut marg_stm = vec![0.0f64; r];
        let mut marg_nstm = vec![0.0f64; r];
        for i in 0..r {
            for j in 0..r {
                let t = target[i * r + j];
                marg_stm[i] += t;
                marg_nstm[j] += t;
            }
        }

        let p_stm = softmax(stm_scores);
        let p_nstm = softmax(nstm_scores);

        let mut loss = 0.0f64;
        let mut grad_stm = vec![0.0f64; r];
        let mut grad_nstm = vec![0.0f64; r];
        for i in 0..r {
            loss -= marg_stm[i] * p_stm[i].max(f64::MIN_POSITIVE).ln();
            grad_stm[i] = p_stm[i] - marg_stm[i];
        }
        for j in 0..r {
            loss -= marg_nstm[j] * p_nstm[j].max(f64::MIN_POSITIVE).ln();
            grad_nstm[j] = p_nstm[j] - marg_nstm[j];
        }
        (grad_stm, grad_nstm, loss)
    }
}

// ===========================================================================
// 3. 重み・Adam state・CPU M-step (RouterKPAbsWeights と対をなす設計)
// ===========================================================================

/// `ft-by-ft` router の重み。評価net本体の FT と**同じ sparse feature 入力**
/// (`feature_set.ft_in()` 次元、`Batch::stm_indices`/`nstm_indices` と同じ
/// active index 表現) 上の `(ft_in, R)` 重み行列 + `(R)` bias。
///
/// STM/NSTM の forward で**同じ重みを共有**する (評価net本体の FT が
/// stm/nstm で重みを共有するのと同じ設計) — 各 perspective の active index
/// 集合 (king 相対など) が異なるだけ。
#[derive(Debug, Clone)]
pub struct RouterFtByFtWeights {
    ft_in: usize,
    r: usize,
    /// `(ft_in, r)` row-major、`w[feat * r + k]`。
    w: Vec<f64>,
    /// `(r)`。
    b: Vec<f64>,
}

impl RouterFtByFtWeights {
    pub fn zeroed(ft_in: usize, r: usize) -> Self {
        Self {
            ft_in,
            r,
            w: vec![0.0; ft_in * r],
            b: vec![0.0; r],
        }
    }

    /// `progress8kpabs`/`RouterKPAbsWeights::random` と同じ流儀の小さい
    /// ランダム初期化 (`N(0, 0.01^2)` 相当、`seed` から決定的に生成)。
    pub fn random(ft_in: usize, r: usize, seed: u64) -> Self {
        let mut state = seed ^ 0x9E3779B97F4A7C15;
        let mut next_f64 = move || {
            // splitmix64
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z as f64) / (u64::MAX as f64)
        };
        let scale = 0.01;
        let w = (0..ft_in * r)
            .map(|_| (next_f64() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            ft_in,
            r,
            w,
            b: vec![0.0; r],
        }
    }

    pub fn ft_in(&self) -> usize {
        self.ft_in
    }

    pub fn r(&self) -> usize {
        self.r
    }

    pub fn w(&self) -> &[f64] {
        &self.w
    }

    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// `w`/`b` を `f32` に変換したスナップショット (`combine_ft_by_ft_columns`
    /// へそのまま渡せる形)。
    pub fn to_f32(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.w.iter().map(|&v| v as f32).collect(),
            self.b.iter().map(|&v| v as f32).collect(),
        )
    }

    /// 1 position の active index 列 (`-1` 終端 padding は無視) から、raw
    /// (pre-activation) router 出力 `R` 個を計算する。評価net本体の FT と
    /// 同じ「active index の重み行の和 + bias」計算。
    pub fn forward_logits(&self, indices: &[i32]) -> Vec<f64> {
        let mut out = self.b.clone();
        for &idx in indices {
            if idx < 0 {
                continue;
            }
            let base = idx as usize * self.r;
            for k in 0..self.r {
                out[k] += self.w[base + k];
            }
        }
        out
    }
}

/// [`RouterFtByFtWeights`] 用 Adam state (`RouterAdamState` と対をなす)。
#[derive(Debug, Clone)]
pub struct RouterFtByFtAdamState {
    pub ft_in: usize,
    pub r: usize,
    pub m_w: Vec<f64>,
    pub v_w: Vec<f64>,
    pub m_b: Vec<f64>,
    pub v_b: Vec<f64>,
    pub t: u64,
}

impl RouterFtByFtAdamState {
    pub fn zeros(ft_in: usize, r: usize) -> Self {
        Self {
            ft_in,
            r,
            m_w: vec![0.0; ft_in * r],
            v_w: vec![0.0; ft_in * r],
            m_b: vec![0.0; r],
            v_b: vec![0.0; r],
            t: 0,
        }
    }
}

/// 1 batch 分の M step 統計 (`RouterTrainStats` と対をなす)。
#[derive(Debug, Clone)]
pub struct RouterFtByFtTrainStats {
    pub cross_entropy_loss: f64,
    /// 負荷分散補助損失 (Switch Transformer 式、STM側・NSTM側それぞれの
    /// R-way softmax に対する `RouterKPAbsWeights::train_oracle_batch` と
    /// 同じ式を独立に適用し、平均したもの)。`balance_weight == 0.0` なら
    /// `0.0`。詳細は [`RouterFtByFtWeights::train_oracle_batch`] のドキュメ
    /// ント参照。
    pub balance_loss: f64,
    /// この batch での STM 側 router 自身の argmax (= 実際に dispatch され
    /// る index) の分布 `f_stm` (長さ `R`)。
    pub bucket_usage_stm: Vec<f64>,
    /// NSTM 側の同上。
    pub bucket_usage_nstm: Vec<f64>,
    /// `kpabs` の `RouterTrainStats::bucket_usage` と同じ形 (長さ
    /// `R*R = num_buckets`) の joint bucket 使用率。各 position の実際の
    /// `(argmax(s), argmax(n))` の組から `bucket_index = stm_idx * R +
    /// nstm_idx` を数えた頻度 (STM/NSTM の独立性は仮定しない、実測の joint
    /// 頻度)。experiment.json への記録 (`RouterHistoryEntry::usage`) 用。
    pub joint_bucket_usage: Vec<f64>,
}

impl RouterFtByFtWeights {
    /// `--router-arch ft-by-ft` の hard-EM M step。
    ///
    /// `stm_indices`/`nstm_indices` は評価net本体の FT が使うのと**同じ**
    /// sparse active index 表現 (`Batch::stm_indices`/`nstm_indices`、行
    /// stride `max_active`、`-1` padding)。`oracle_targets[i]` は position
    /// `i` の `R*R` 個の bucket に対する soft-EM 目的分布
    /// (`shogi_features::router_kpabs::oracle_targets_from_errors` の出力を
    /// そのまま渡せる — 既存の E step (bucket 0..N の N 通り forward で誤差
    /// を求める) をそのまま再利用できる、`router_kpabs` と共通のロジック)。
    ///
    /// STM 側・NSTM 側で独立な softmax cross entropy に分解される
    /// ([`loss::soft_target_grad`]) ため、**同じ共有重み `w`/`b` に対して
    /// STM の active index からの勾配と NSTM の active index からの勾配を
    /// 両方加算**してから 1 回の Adam step を行う。
    ///
    /// ## 負荷分散補助損失 (`balance_weight`)
    ///
    /// `RouterKPAbsWeights::train_oracle_batch` と全く同じ Switch
    /// Transformer 式の補助損失を、STM 側・NSTM 側それぞれの `R`-way
    /// softmax に**独立に**適用する:
    ///
    /// ```text
    /// L_balance_stm  = R · Σ_i f_stm_i  · P_stm_i
    /// L_balance_nstm = R · Σ_j f_nstm_j · P_nstm_j
    /// ```
    ///
    /// - `f_stm_i` / `f_nstm_j` = この batch で router 自身の argmax
    ///   (実際に dispatch される index) が `i`/`j` になった position の割合
    ///   (stop-gradient)
    /// - `P_stm_i` / `P_nstm_j` = この batch での `softmax(s)[i]` /
    ///   `softmax(n)[j]` の平均 (微分可能)
    ///
    /// 両者を平均したものを [`RouterFtByFtTrainStats::balance_loss`] として
    /// 返す。`P(i,j) = softmax(s)[i] * softmax(n)[j]` の設計により、STM 側
    /// ・NSTM 側それぞれの softmax jacobian に対する寄与も独立に計算できる
    /// ため、kpabs の式をそのまま各側に適用するだけでよい (joint な `R*R`
    /// 版の負荷分散を別途定義する必要はない)。`balance_weight == 0.0` では
    /// cross entropy のみの従来動作 (kpabs と同じ既定)。
    #[allow(clippy::too_many_arguments)]
    pub fn train_oracle_batch(
        &mut self,
        stm_indices: &[i32],
        nstm_indices: &[i32],
        max_active: usize,
        n_positions: usize,
        oracle_targets: &[Vec<f64>],
        adam: &mut RouterFtByFtAdamState,
        lr: f64,
        weight_decay: f64,
        balance_weight: f64,
    ) -> RouterFtByFtTrainStats {
        assert_eq!(oracle_targets.len(), n_positions);
        assert_eq!(
            adam.r, self.r,
            "RouterFtByFtAdamState r mismatch with RouterFtByFtWeights"
        );
        let r = self.r;
        if n_positions == 0 {
            return RouterFtByFtTrainStats {
                cross_entropy_loss: 0.0,
                balance_loss: 0.0,
                bucket_usage_stm: vec![0.0; r],
                bucket_usage_nstm: vec![0.0; r],
                joint_bucket_usage: vec![0.0; r * r],
            };
        }

        // 1st pass: forward しつつ softmax probs をキャッシュし、STM/NSTM
        // それぞれの router 自身の argmax (= 実際の dispatch 先) の分布
        // `f_stm`/`f_nstm` を求める (`balance_weight == 0.0` でも計算コスト
        // は小さいので常に計算し、診断用の `bucket_usage_*` として返す)。
        // 併せて、STM/NSTM の独立性を仮定しない実測の joint 分布
        // (`bucket_index = stm_idx * R + nstm_idx` ごとの頻度、kpabs の
        // `RouterTrainStats::bucket_usage` と同じ形) も数える
        // (experiment.json の `RouterHistoryEntry::usage` 用)。
        let mut all_probs_stm: Vec<Vec<f64>> = Vec::with_capacity(n_positions);
        let mut all_probs_nstm: Vec<Vec<f64>> = Vec::with_capacity(n_positions);
        let mut dispatch_stm = vec![0u32; r];
        let mut dispatch_nstm = vec![0u32; r];
        let mut joint_dispatch = vec![0u32; r * r];
        for i in 0..n_positions {
            let stm_row = &stm_indices[i * max_active..(i + 1) * max_active];
            let nstm_row = &nstm_indices[i * max_active..(i + 1) * max_active];
            let s = self.forward_logits(stm_row);
            let n = self.forward_logits(nstm_row);
            let probs_s = loss::softmax(&s);
            let probs_n = loss::softmax(&n);
            let stm_idx = loss::argmax(&s);
            let nstm_idx = loss::argmax(&n);
            dispatch_stm[stm_idx] += 1;
            dispatch_nstm[nstm_idx] += 1;
            joint_dispatch[stm_idx * r + nstm_idx] += 1;
            all_probs_stm.push(probs_s);
            all_probs_nstm.push(probs_n);
        }
        let f_stm: Vec<f64> = dispatch_stm.iter().map(|&c| c as f64 / n_positions as f64).collect();
        let f_nstm: Vec<f64> = dispatch_nstm.iter().map(|&c| c as f64 / n_positions as f64).collect();
        let joint_bucket_usage: Vec<f64> = joint_dispatch
            .iter()
            .map(|&c| c as f64 / n_positions as f64)
            .collect();

        // 2nd pass: cross entropy (soft target 対応、marginal 分解済み) +
        // (balance_weight != 0 なら) 負荷分散項の勾配を合成して backprop する。
        let mut grad_w = vec![0.0f64; self.w.len()];
        let mut grad_b = vec![0.0f64; r];
        let mut total_loss = 0.0f64;
        let mut avg_probs_stm = vec![0.0f64; r];
        let mut avg_probs_nstm = vec![0.0f64; r];

        for i in 0..n_positions {
            let stm_row = &stm_indices[i * max_active..(i + 1) * max_active];
            let nstm_row = &nstm_indices[i * max_active..(i + 1) * max_active];
            let target = &oracle_targets[i];
            debug_assert_eq!(target.len(), r * r);
            let probs_s = &all_probs_stm[i];
            let probs_n = &all_probs_nstm[i];

            // marginal 分解 ([`loss::soft_target_grad`] と同じ式だが、ここでは
            // probs をキャッシュ済みなので直接展開する) + loss 計算。
            let mut marg_stm = vec![0.0f64; r];
            let mut marg_nstm = vec![0.0f64; r];
            for a in 0..r {
                for b in 0..r {
                    let t = target[a * r + b];
                    marg_stm[a] += t;
                    marg_nstm[b] += t;
                }
            }
            let mut grad_s = vec![0.0f64; r];
            let mut grad_n = vec![0.0f64; r];
            for k in 0..r {
                total_loss -= marg_stm[k] * probs_s[k].max(f64::MIN_POSITIVE).ln();
                grad_s[k] = probs_s[k] - marg_stm[k];
                avg_probs_stm[k] += probs_s[k];
            }
            for k in 0..r {
                total_loss -= marg_nstm[k] * probs_n[k].max(f64::MIN_POSITIVE).ln();
                grad_n[k] = probs_n[k] - marg_nstm[k];
                avg_probs_nstm[k] += probs_n[k];
            }

            if balance_weight != 0.0 {
                // `balance_loss` は STM/NSTM の平均 (`/2.0`) として報告する
                // ため、勾配もそれに合わせて `/2.0` する
                // (`d(L_stm/2 + L_nstm/2)/dw = (dL_stm/dw + dL_nstm/dw)/2`;
                // 有限差分でこのスケーリング忘れを検出済み ―
                // `router_ftbyft_balance_loss_matches_finite_difference`)。
                let n_buckets = r as f64;
                let dot_s: f64 = f_stm.iter().zip(probs_s.iter()).map(|(&fi, &pi)| fi * pi).sum();
                for j in 0..r {
                    grad_s[j] += 0.5 * balance_weight * n_buckets * probs_s[j] * (f_stm[j] - dot_s);
                }
                let dot_n: f64 = f_nstm.iter().zip(probs_n.iter()).map(|(&fi, &pi)| fi * pi).sum();
                for j in 0..r {
                    grad_n[j] += 0.5 * balance_weight * n_buckets * probs_n[j] * (f_nstm[j] - dot_n);
                }
            }

            for &idx in stm_row {
                if idx < 0 {
                    continue;
                }
                let base = idx as usize * r;
                for k in 0..r {
                    grad_w[base + k] += grad_s[k];
                }
            }
            for k in 0..r {
                grad_b[k] += grad_s[k];
            }

            for &idx in nstm_row {
                if idx < 0 {
                    continue;
                }
                let base = idx as usize * r;
                for k in 0..r {
                    grad_w[base + k] += grad_n[k];
                }
            }
            for k in 0..r {
                grad_b[k] += grad_n[k];
            }
        }

        let inv_n = 1.0 / n_positions as f64;
        for g in grad_w.iter_mut() {
            *g *= inv_n;
        }
        for g in grad_b.iter_mut() {
            *g *= inv_n;
        }
        for p in avg_probs_stm.iter_mut() {
            *p *= inv_n;
        }
        for p in avg_probs_nstm.iter_mut() {
            *p *= inv_n;
        }

        adam.t += 1;
        adam_step(&mut self.w, &grad_w, &mut adam.m_w, &mut adam.v_w, lr, weight_decay, adam.t);
        // bias は weight decay の対象外 (`RouterKPAbsWeights` に bias が無い
        // のと同様、評価net本体の FT bias も weight decay しない慣習に合わせる)。
        adam_step(&mut self.b, &grad_b, &mut adam.m_b, &mut adam.v_b, lr, 0.0, adam.t);

        let n_buckets = r as f64;
        let balance_loss_stm: f64 =
            n_buckets * f_stm.iter().zip(avg_probs_stm.iter()).map(|(&fi, &pi)| fi * pi).sum::<f64>();
        let balance_loss_nstm: f64 =
            n_buckets * f_nstm.iter().zip(avg_probs_nstm.iter()).map(|(&fi, &pi)| fi * pi).sum::<f64>();

        RouterFtByFtTrainStats {
            cross_entropy_loss: total_loss * inv_n,
            balance_loss: (balance_loss_stm + balance_loss_nstm) / 2.0,
            bucket_usage_stm: f_stm,
            bucket_usage_nstm: f_nstm,
            joint_bucket_usage,
        }
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&(self.ft_in as u32).to_le_bytes())?;
        writer.write_all(&(self.r as u32).to_le_bytes())?;
        write_f64_slice(writer, &self.w)?;
        write_f64_slice(writer, &self.b)?;
        Ok(())
    }

    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let ft_in = read_u32(reader)? as usize;
        let r = read_u32(reader)? as usize;
        let w = read_f64_vec(reader, ft_in * r)?;
        let b = read_f64_vec(reader, r)?;
        Ok(Self { ft_in, r, w, b })
    }
}

fn adam_step(w: &mut [f64], grad: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, weight_decay: f64, t: u64) {
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

// ===========================================================================
// 4. process-global entry point (`RouterKPAbs` と対をなす)
// ===========================================================================

static ROUTER_FTBYFT_STATE: OnceLock<RwLock<RouterFtByFtWeights>> = OnceLock::new();

/// `--router-arch ft-by-ft` の process-global エントリポイント。
/// `RouterKPAbs` と同じ役割 (dataloader/学習ループから読み書きされる)。
#[derive(Clone, Copy, Default)]
pub struct RouterFtByFt;

impl RouterFtByFt {
    /// ランダム初期化して global へ設置する。学習開始時に (resume でない
    /// 限り) 1 回だけ呼ぶこと。
    pub fn init_random(ft_in: usize, r: usize, seed: u64) -> Result<(), String> {
        let weights = RouterFtByFtWeights::random(ft_in, r, seed);
        Self::init_with_weights(weights)
    }

    pub fn init_with_weights(weights: RouterFtByFtWeights) -> Result<(), String> {
        ROUTER_FTBYFT_STATE
            .set(RwLock::new(weights))
            .map_err(|_| "RouterFtByFt already initialized".to_string())
    }

    pub fn overwrite_weights(w: Vec<f64>, b: Vec<f64>) {
        if let Some(lock) = ROUTER_FTBYFT_STATE.get() {
            let mut guard = lock.write().expect("RouterFtByFt RwLock poisoned");
            assert_eq!(guard.w.len(), w.len(), "overwrite_weights: w length mismatch");
            assert_eq!(guard.b.len(), b.len(), "overwrite_weights: b length mismatch");
            guard.w = w;
            guard.b = b;
        }
    }

    pub fn snapshot() -> RouterFtByFtWeights {
        ROUTER_FTBYFT_STATE
            .get()
            .expect("RouterFtByFt not initialized")
            .read()
            .expect("RouterFtByFt RwLock poisoned")
            .clone()
    }

    pub fn try_snapshot() -> Option<RouterFtByFtWeights> {
        ROUTER_FTBYFT_STATE.get().map(|lock| {
            lock.read()
                .expect("RouterFtByFt RwLock poisoned")
                .clone()
        })
    }

    pub fn active_indices_row<'a>(indices: &'a [i32], row: usize, max_active: usize) -> &'a [i32] {
        &indices[row * max_active..(row + 1) * max_active]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_example_64_buckets_2296_ft_out() {
        let layout = FtByFtLayout::new(64, 2296).expect("valid");
        assert_eq!(layout.r, 8);
        assert_eq!(layout.half_normal, 1148);
        assert_eq!(layout.half_router, 4);
        assert_eq!(layout.accum_out, 2304);
        assert_eq!(layout.kept_pairwise_out(), 1148);
        assert_eq!(layout.discarded_pairwise_out(), 4);

        let ranges = layout.column_ranges();
        assert_eq!(ranges.normal_a, (0, 1148));
        assert_eq!(ranges.router_a, (1148, 1152));
        assert_eq!(ranges.normal_b, (1152, 2300));
        assert_eq!(ranges.router_b, (2300, 2304));
    }

    #[test]
    fn rejects_non_perfect_square() {
        let err = FtByFtLayout::new(32, 2296).unwrap_err();
        assert!(matches!(err, FtByFtError::NotPerfectSquare { num_buckets: 32 }));
    }

    #[test]
    fn rejects_odd_root() {
        let err = FtByFtLayout::new(9, 2296).unwrap_err();
        assert!(matches!(
            err,
            FtByFtError::OddRoot { num_buckets: 9, root: 3 }
        ));
    }

    #[test]
    fn rejects_odd_ft_out() {
        let err = FtByFtLayout::new(64, 2295).unwrap_err();
        assert!(matches!(err, FtByFtError::OddFtOut { ft_out: 2295 }));
    }

    #[test]
    fn accepts_small_case() {
        let layout = FtByFtLayout::new(4, 8).expect("valid");
        assert_eq!(layout.r, 2);
        assert_eq!(layout.accum_out, 10);
        let ranges = layout.column_ranges();
        assert_eq!(ranges.normal_a, (0, 4));
        assert_eq!(ranges.router_a, (4, 5));
        assert_eq!(ranges.normal_b, (5, 9));
        assert_eq!(ranges.router_b, (9, 10));
    }

    #[test]
    fn loss_softmax_sums_to_one() {
        let p = loss::softmax(&[1.0, 2.0, -3.0, 0.5]);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
    }

    #[test]
    fn loss_predict_bucket_matches_argmax_grid() {
        let stm = [0.1, 5.0, -1.0];
        let nstm = [3.0, -2.0, 0.0];
        assert_eq!(loss::predict_bucket(&stm, &nstm), 1 * 3 + 0);
    }

    #[test]
    fn loss_hard_target_grad_matches_finite_difference() {
        let stm = vec![0.3f64, -1.2, 2.0, 0.7];
        let nstm = vec![-0.5f64, 1.1, 0.2, -2.0];
        let (grad_stm, grad_nstm, _loss0) = loss::hard_target_grad(&stm, &nstm, 2, 1);

        let eps = 1e-6f64;
        for i in 0..stm.len() {
            let mut s_plus = stm.clone();
            s_plus[i] += eps;
            let (_, _, loss_plus) = loss::hard_target_grad(&s_plus, &nstm, 2, 1);
            let mut s_minus = stm.clone();
            s_minus[i] -= eps;
            let (_, _, loss_minus) = loss::hard_target_grad(&s_minus, &nstm, 2, 1);
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numeric - grad_stm[i]).abs() < 1e-6,
                "stm[{i}]: analytic={}, numeric={}",
                grad_stm[i],
                numeric
            );
        }
    }

    #[test]
    fn loss_soft_target_matches_hard_target_for_onehot() {
        let stm = vec![0.3f64, -1.2, 2.0];
        let nstm = vec![-0.5f64, 1.1, 0.2];
        let r = 3;
        let (ti, tj) = (2usize, 0usize);
        let mut target = vec![0.0f64; r * r];
        target[ti * r + tj] = 1.0;

        let (gs_soft, gn_soft, loss_soft) = loss::soft_target_grad(&stm, &nstm, &target);
        let (gs_hard, gn_hard, loss_hard) = loss::hard_target_grad(&stm, &nstm, ti, tj);
        for i in 0..r {
            assert!((gs_soft[i] - gs_hard[i]).abs() < 1e-9);
            assert!((gn_soft[i] - gn_hard[i]).abs() < 1e-9);
        }
        assert!((loss_soft - loss_hard).abs() < 1e-9);
    }

    #[test]
    fn combine_columns_matches_layout_ranges() {
        let ft_in = 3;
        let layout = FtByFtLayout::new(4, 8).unwrap(); // R=2, H=4, r_half=1
        let normal_w: Vec<f32> = (0..ft_in * layout.ft_out).map(|i| i as f32).collect();
        let normal_b: Vec<f32> = (0..layout.ft_out).map(|i| 100.0 + i as f32).collect();
        let router_w: Vec<f32> = (0..ft_in * layout.r).map(|i| 1000.0 + i as f32).collect();
        let router_b: Vec<f32> = (0..layout.r).map(|i| 2000.0 + i as f32).collect();

        let (combined_w, combined_b) =
            combine_ft_by_ft_columns(&normal_w, &normal_b, &router_w, &router_b, ft_in, &layout);

        assert_eq!(combined_w.len(), ft_in * layout.accum_out);
        assert_eq!(combined_b.len(), layout.accum_out);

        // feature 0 の行を検証: normal_w row0 = [0,1,2,3] (H=4), router_w row0 = [1000,1001] (r=2)
        let row0 = &combined_w[0..layout.accum_out];
        assert_eq!(row0, &[0.0, 1.0, 2.0, 3.0, 1000.0, 4.0, 5.0, 6.0, 7.0, 1001.0]);

        assert_eq!(
            combined_b,
            vec![100.0, 101.0, 102.0, 103.0, 2000.0, 104.0, 105.0, 106.0, 107.0, 2001.0]
        );
    }

    #[test]
    fn router_ftbyft_weights_forward_and_train_reduce_loss() {
        let ft_in = 5;
        let r = 2;
        let mut weights = RouterFtByFtWeights::zeroed(ft_in, r);
        let mut adam = RouterFtByFtAdamState::zeros(ft_in, r);

        // 2 positions, max_active=2 (padding -1).
        let stm_indices = vec![0, 1, 2, -1];
        let nstm_indices = vec![1, 3, 0, -1];
        let max_active = 2;
        // targets: position0 -> bucket (0,1) = index 0*2+1=1; position1 -> bucket (1,0) = index 2.
        let mut t0 = vec![0.0; r * r];
        t0[0 * r + 1] = 1.0;
        let mut t1 = vec![0.0; r * r];
        t1[1 * r + 0] = 1.0;
        let targets = vec![t0, t1];

        let stats0 = weights.train_oracle_batch(
            &stm_indices, &nstm_indices, max_active, 2, &targets, &mut adam, 0.1, 0.0, 0.0,
        );
        let stats1 = weights.train_oracle_batch(
            &stm_indices, &nstm_indices, max_active, 2, &targets, &mut adam, 0.1, 0.0, 0.0,
        );
        // 同じターゲットで 2 step 学習すれば loss は下がるはず。
        assert!(
            stats1.cross_entropy_loss < stats0.cross_entropy_loss,
            "loss should decrease: {} -> {}",
            stats0.cross_entropy_loss,
            stats1.cross_entropy_loss
        );
    }

    #[test]
    fn router_ftbyft_joint_bucket_usage_sums_to_one_and_matches_dispatch() {
        // weights=0 の場合、全 index の raw score は 0 (softmax は一様) だが
        // argmax は実装の tie-break (最初の index) で決定的に 0 になる。
        // よって joint_bucket_usage はすべて bucket (0,0) = index 0 に集中
        // するはず。
        let ft_in = 5;
        let r = 2;
        let mut weights = RouterFtByFtWeights::zeroed(ft_in, r);
        let mut adam = RouterFtByFtAdamState::zeros(ft_in, r);
        let stm_indices = vec![0, 1, 2, -1];
        let nstm_indices = vec![1, 3, 0, -1];
        let max_active = 2;
        let t0 = vec![0.0; r * r];
        let t1 = vec![0.0; r * r];
        let targets = vec![t0, t1];

        let stats = weights.train_oracle_batch(
            &stm_indices, &nstm_indices, max_active, 2, &targets, &mut adam, 0.0, 0.0, 0.0,
        );
        assert_eq!(stats.joint_bucket_usage.len(), r * r);
        let sum: f64 = stats.joint_bucket_usage.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
        assert!((stats.joint_bucket_usage[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn router_ftbyft_balance_loss_matches_finite_difference() {
        // balance_weight != 0 のときの勾配 (softmax jacobian 経由の balance
        // 項) を、重み全体を動かした有限差分と突き合わせて検証する。
        // (weight decay=0、n_positions=3、R=2、balance_weight=0.7)
        let ft_in = 4;
        let r = 2;
        let stm_indices = vec![0, 1, 2, 3, 0, -1, 1, 2, -1, -1];
        let nstm_indices = vec![1, 2, -1, -1, 3, 0, -1, -1, 2, 3];
        let max_active = 2;
        let mut t0 = vec![0.0; r * r];
        t0[0] = 1.0; // (0,0)
        let mut t1 = vec![0.0; r * r];
        t1[3] = 1.0; // (1,1)
        let mut t2 = vec![0.0; r * r];
        t2[1] = 1.0; // (0,1)
        let targets = vec![t0, t1, t2];
        let balance_weight = 0.7;

        // 有限差分は argmax (dispatch) が微小摂動で反転しないことを要求する
        // ので、softmax がほぼ一様になる (`RouterFtByFtWeights::random` の
        // ような) 小さすぎる初期値は避け、はっきり分離した重みを使う。
        let mut base = RouterFtByFtWeights::zeroed(ft_in, r);
        base.w = vec![2.0, -1.5, 0.3, 1.8, -2.0, 0.5, 1.1, -0.7];

        fn total_objective(stats: &RouterFtByFtTrainStats, balance_weight: f64) -> f64 {
            stats.cross_entropy_loss + balance_weight * stats.balance_loss
        }

        let objective = |w_override: &RouterFtByFtWeights| -> f64 {
            let mut w = w_override.clone();
            let mut adam = RouterFtByFtAdamState::zeros(ft_in, r);
            let stats = w.train_oracle_batch(
                &stm_indices,
                &nstm_indices,
                max_active,
                3,
                &targets,
                &mut adam,
                0.0,
                0.0,
                balance_weight,
            );
            total_objective(&stats, balance_weight)
        };

        // 勾配 (lr=0 で読み取れる Adam 前の raw gradient がほしいが、この
        // API は Adam step 込みなので、代わりに極小 lr で 1 step 進めた後の
        // weight 差分の符号が有限差分の降下方向と一致するか、を簡便に検証
        // する: objective(w) と objective(w - eps*sign) を比較するのではな
        // く、直接 w の一要素を動かした有限差分と、`train_oracle_batch` が
        // 内部計算する解析的勾配 (loss::soft_target_grad + balance 項) を
        // 比較する。
        let mut w_plus = base.clone();
        let mut w_minus = base.clone();
        let eps = 1e-4;
        let probe = 3usize; // w[3] を動かす (ft_in*r=8 要素のうちの1つ)
        w_plus.w[probe] += eps;
        w_minus.w[probe] -= eps;

        let f_plus = objective(&w_plus);
        let f_minus = objective(&w_minus);
        let numeric_grad = (f_plus - f_minus) / (2.0 * eps);

        // 解析的勾配は forward_logits + softmax + (soft_target_grad 相当の
        // marginal 勾配) + balance 項を手計算して求める (train_oracle_batch
        // の内部ロジックと同じ式を、この probe 一点についてだけ再現する)。
        let feat = probe / r;
        let k = probe % r;
        let mut analytic_grad = 0.0f64;
        // dispatch (argmax) 分布は base の重みで固定 (balance 項の f_* は
        // stop-gradient なので、有限差分でも base の f_* を使うのが正しい
        // 比較になる -- ただし train_oracle_batch は w_plus/w_minus 自身の
        // dispatch を使ってしまうため、eps を十分小さくして argmax が変わ
        // らない前提で比較する)。
        for i in 0..3 {
            let stm_row = &stm_indices[i * max_active..(i + 1) * max_active];
            let nstm_row = &nstm_indices[i * max_active..(i + 1) * max_active];
            let uses_feat_stm = stm_row.contains(&(feat as i32));
            let uses_feat_nstm = nstm_row.contains(&(feat as i32));
            if !uses_feat_stm && !uses_feat_nstm {
                continue;
            }
            let s = base.forward_logits(stm_row);
            let n = base.forward_logits(nstm_row);
            let target = &targets[i];
            let (grad_s, grad_n, _) = loss::soft_target_grad(&s, &n, target);
            if uses_feat_stm {
                analytic_grad += grad_s[k] / 3.0;
            }
            if uses_feat_nstm {
                analytic_grad += grad_n[k] / 3.0;
            }
        }
        // balance 項 (stop-gradient の f_* は base の forward から再計算)。
        let mut dispatch_stm = vec![0u32; r];
        let mut dispatch_nstm = vec![0u32; r];
        let mut probs_s_all = Vec::new();
        let mut probs_n_all = Vec::new();
        for i in 0..3 {
            let stm_row = &stm_indices[i * max_active..(i + 1) * max_active];
            let nstm_row = &nstm_indices[i * max_active..(i + 1) * max_active];
            let s = base.forward_logits(stm_row);
            let n = base.forward_logits(nstm_row);
            dispatch_stm[loss::argmax(&s)] += 1;
            dispatch_nstm[loss::argmax(&n)] += 1;
            probs_s_all.push(loss::softmax(&s));
            probs_n_all.push(loss::softmax(&n));
        }
        let f_stm: Vec<f64> = dispatch_stm.iter().map(|&c| c as f64 / 3.0).collect();
        let f_nstm: Vec<f64> = dispatch_nstm.iter().map(|&c| c as f64 / 3.0).collect();
        for i in 0..3 {
            let stm_row = &stm_indices[i * max_active..(i + 1) * max_active];
            let nstm_row = &nstm_indices[i * max_active..(i + 1) * max_active];
            let probs_s = &probs_s_all[i];
            let probs_n = &probs_n_all[i];
            if stm_row.contains(&(feat as i32)) {
                let dot: f64 = f_stm.iter().zip(probs_s.iter()).map(|(&fi, &pi)| fi * pi).sum();
                analytic_grad +=
                    (0.5 * balance_weight * r as f64 * probs_s[k] * (f_stm[k] - dot)) / 3.0;
            }
            if nstm_row.contains(&(feat as i32)) {
                let dot: f64 = f_nstm.iter().zip(probs_n.iter()).map(|(&fi, &pi)| fi * pi).sum();
                analytic_grad +=
                    (0.5 * balance_weight * r as f64 * probs_n[k] * (f_nstm[k] - dot)) / 3.0;
            }
        }

        assert!(
            (numeric_grad - analytic_grad).abs() < 1e-2,
            "numeric={numeric_grad}, analytic={analytic_grad}"
        );
    }
}
