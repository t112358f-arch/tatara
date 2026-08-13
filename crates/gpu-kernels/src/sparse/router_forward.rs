//! `--bucket-mode router` の GPU forward kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn router_forward_f64`) は `bins/nnue_train/src/kernels/`
//! に定義されている (cuda-oxide rustc-codegen-cuda backend は bin entry 経由で
//! 到達可能な kernel しか PTX 化しないため)。本 module の `router_forward_cpu`
//! は GPU と同じロジックを host に書き写したもので、GPU↔CPU 数値同等性テスト
//! の reference に使う。
//!
//! ## アルゴリズム
//!
//! `shogi_features::router_kpabs::RouterKPAbsWeights::forward_logits` (bias 無し
//! の重み和、N bucket 分) の GPU 版。`weight` は **index-major**
//! (`weight[idx * num_buckets + k]`、`RouterKPAbsWeights::w` と同レイアウト —
//! `sparse_ft_forward` の column-major `weight[col * rows + row]` と数式上同一
//! の layout で、`rows = num_buckets` に対応する)。
//!
//! ```text
//! per (batch_index bi, bucket_index k):
//!     sum = 0
//!     for ni in 0..max_active:
//!         idx = indices[bi * max_active + ni]
//!         if idx >= 0 && (idx as usize) < num_features:
//!             sum += weight[idx * num_buckets + k]
//!     out[bi * num_buckets + k] = sum
//! ```
//!
//! - `weight` (size `num_features * num_buckets`): index-major
//! - `indices` (size `batch * max_active`): per position の active
//!   KP-absolute index (`shogi_features::router_kpabs::RouterKPAbs::
//!   active_indices_board` 由来)、`-1` は固定幅化のための padding (skip)。
//!   `>= num_features` も defensive に silent skip
//! - `out` (size `batch * num_buckets`): per position の router logits
//!
//! `f64` 固定 (`RouterKPAbsWeights` が `progress.bin` と同じ f64 精度を要求する
//! ため、`sparse_ft_forward` 系と異なり f32 化しない)。

/// Reference CPU 実装。
///
/// Out-of-place 出力: `out[bi * num_buckets + k]` を 1 entry ずつ埋める。
///
/// 入力前提:
/// - `weight.len() == num_features * num_buckets` (index-major、
///   `weight[idx * num_buckets + k]`)
/// - `indices.len() == batch * max_active` (`-1` padding 許容、
///   `>= num_features` も silent skip)
/// - `out.len() == batch * num_buckets`
///
/// 引数数 (6) は入出力 + sparse 形状を漏れなく渡すため
/// `clippy::too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn router_forward_cpu(
    weight: &[f64],
    indices: &[i32],
    out: &mut [f64],
    batch: usize,
    num_buckets: usize,
    num_features: usize,
    max_active: usize,
) {
    for bi in 0..batch {
        for k in 0..num_buckets {
            let mut sum = 0.0_f64;
            for ni in 0..max_active {
                let idx = indices[bi * max_active + ni];
                if idx >= 0 && (idx as usize) < num_features {
                    sum += weight[(idx as usize) * num_buckets + k];
                }
            }
            out[bi * num_buckets + k] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// shape (batch=2, num_buckets=2, num_features=3, max_active=4) で
    /// `sparse_ft_forward_cpu` の基準テストと対応する値になる (index-major
    /// layout が column-major `weight[col*rows+row]` と数式上同じであることを
    /// 確認する)。
    #[test]
    fn matches_index_major_layout() {
        // num_features=3, num_buckets=2 の index-major weight。
        let weights = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0_i32, 1, -1, -1, 2, 2, 1, 0];
        let mut out = vec![0.0_f64; 4];

        router_forward_cpu(&weights, &indices, &mut out, 2, 2, 3, 4);

        assert_eq!(out, vec![2.0_f64, 4.0, 10.0, 14.0]);
    }

    /// 全 padding (-1) の position は output が 0 になる。
    #[test]
    fn all_padding_yields_zero() {
        let weights = vec![1.0_f64, 2.0, 3.0, 4.0]; // num_features=2, num_buckets=2
        let indices = vec![-1_i32; 6];
        let mut out = vec![999.0_f64, 999.0];
        router_forward_cpu(&weights, &indices, &mut out, 1, 2, 2, 6);
        assert_eq!(out, vec![0.0_f64, 0.0]);
    }

    /// `idx >= num_features` の異常入力は silent skip。
    #[test]
    fn out_of_range_index_is_silently_skipped() {
        let weights = vec![1.0_f64, 2.0, 3.0, 4.0]; // num_features=2, num_buckets=2
        let indices = vec![0_i32, 5, -1];
        let mut out = vec![0.0_f64; 2];
        router_forward_cpu(&weights, &indices, &mut out, 1, 2, 2, 3);
        assert_eq!(out, vec![1.0_f64, 2.0]);
    }

    /// 同一 index が重複しても、`RouterKPAbsWeights::forward_logits` (`for &idx
    /// in indices { ... }`) と同じく重み合計になる。
    #[test]
    fn duplicate_indices_are_summed() {
        let weights = vec![1.0_f64, 10.0, 2.0, 20.0]; // num_features=2, num_buckets=2
        let indices = vec![0_i32, 0, 1];
        let mut out = vec![0.0_f64; 2];
        router_forward_cpu(&weights, &indices, &mut out, 1, 2, 2, 3);
        assert_eq!(out, vec![4.0_f64, 40.0]);
    }

    /// batch=0 / num_buckets=0 / num_features=0 / max_active=0 は panic せず no-op。
    #[test]
    fn zero_dimension_is_no_op() {
        let weights: Vec<f64> = vec![];
        let indices: Vec<i32> = vec![];
        let mut out: Vec<f64> = vec![];
        router_forward_cpu(&weights, &indices, &mut out, 0, 0, 0, 0);
        assert!(out.is_empty());
    }
}
