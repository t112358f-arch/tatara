//! `--bucket-mode router` の GPU backward (atomic scatter) kernel の reference
//! CPU 実装。
//!
//! GPU 側 (`#[kernel] fn router_backward_scatter_f64`) は
//! `bins/nnue_train/src/kernels/` に定義されている (cuda-oxide
//! rustc-codegen-cuda backend は bin entry 経由で到達可能な kernel しか PTX 化
//! しないため)。本 module の `router_backward_scatter_cpu` は GPU と同じ
//! ロジックを host に書き写したもので、GPU↔CPU 数値同等性テストの reference
//! に使う。
//!
//! ## アルゴリズム
//!
//! [`super::router_forward::router_forward_cpu`] の対 backward。caller
//! (`bins/nnue_train::router_gpu`) が host 側で softmax cross entropy + 負荷
//! 分散補助損失の勾配 (`d_logits`、batch size で平均済) を計算した後、それを
//! `RouterKPAbsWeights::train_oracle_batch` / `train_backprop_batch` の
//! `for &idx in indices { grad_w[base+k] += d_logits[k] }` ループと同じ意味で
//! `grad_weight` (index-major) へ scatter-add する。
//!
//! ```text
//! per (batch_index bi, bucket_index k):
//!     g = d_logits[bi * num_buckets + k]
//!     for ni in 0..max_active:
//!         idx = indices[bi * max_active + ni]
//!         if idx >= 0 && (idx as usize) < num_features:
//!             grad_weight[idx * num_buckets + k] += g   # GPU: atomicAdd, CPU: 累積
//! ```
//!
//! - **`grad_weight` の初期化**: 本 fn は **accumulate** semantics で既存値に
//!   add する。host が呼び出し前に 0 clear する責務 (`sparse_ft_backward_cpu`
//!   と同じ契約)
//! - layout: index-major (`router_forward_cpu` と同 `weight[idx*num_buckets+k]`)
//! - thread 配置: flat 1D `tid = bi * num_buckets + k` (forward と同型 idiom、
//!   atomic scatter で衝突を吸収)
//!
//! `f64` 固定 (`router_forward_cpu` と同じ理由)。

/// Reference CPU 実装。
///
/// In-place accumulate: `grad_weight` に既存値を add する (host が呼び出し前に
/// 0 で初期化する責務)。
///
/// 入力前提:
/// - `d_logits.len() == batch * num_buckets`
/// - `indices.len() == batch * max_active` (`-1` padding 許容、
///   `>= num_features` も silent skip)
/// - `grad_weight.len() == num_features * num_buckets` (index-major)
///
/// 引数数 (7) は入出力 + sparse 形状を漏れなく渡すため
/// `clippy::too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn router_backward_scatter_cpu(
    d_logits: &[f64],
    indices: &[i32],
    grad_weight: &mut [f64],
    batch: usize,
    num_buckets: usize,
    num_features: usize,
    max_active: usize,
) {
    for bi in 0..batch {
        for k in 0..num_buckets {
            let g = d_logits[bi * num_buckets + k];
            for ni in 0..max_active {
                let idx = indices[bi * max_active + ni];
                if idx >= 0 && (idx as usize) < num_features {
                    grad_weight[(idx as usize) * num_buckets + k] += g;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::router_forward::router_forward_cpu;

    /// forward/backward は互いに転置の関係にある: `<forward(w), d_out>` ==
    /// `<w, backward(d_out)>` (有限差分ではなく厳密な線形性チェック)。
    #[test]
    fn is_adjoint_of_forward() {
        let num_features = 3;
        let num_buckets = 2;
        let batch = 2;
        let max_active = 4;
        let weights = vec![0.1_f64, 0.2, -0.3, 0.4, 0.5, -0.6];
        let indices = vec![0_i32, 1, -1, -1, 2, 0, 1, -1];

        let mut logits = vec![0.0_f64; batch * num_buckets];
        router_forward_cpu(&weights, &indices, &mut logits, batch, num_buckets, num_features, max_active);

        // 適当な d_logits を与えて backward。
        let d_logits = vec![1.0_f64, -2.0, 0.5, 3.0];
        let mut grad_weight = vec![0.0_f64; num_features * num_buckets];
        router_backward_scatter_cpu(
            &d_logits,
            &indices,
            &mut grad_weight,
            batch,
            num_buckets,
            num_features,
            max_active,
        );

        // <forward(w), d_out> = Σ_bi Σ_k logits[bi,k] * d_logits[bi,k]
        let lhs: f64 = logits.iter().zip(d_logits.iter()).map(|(&a, &b)| a * b).sum();
        // <w, backward(d_out)> = Σ_idx Σ_k w[idx,k] * grad_weight[idx,k]
        let rhs: f64 = weights.iter().zip(grad_weight.iter()).map(|(&a, &b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-9, "lhs={lhs} rhs={rhs}");
    }

    /// 同一 (idx, k) cell に複数 position が書き込む場合、和が累積される
    /// (host 呼び出し前 0 初期化前提の accumulate semantics)。
    #[test]
    fn duplicate_targets_accumulate() {
        let num_features = 2;
        let num_buckets = 2;
        // 2 position とも idx=0 を参照。
        let indices = vec![0_i32, -1, 0_i32, -1];
        let d_logits = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut grad_weight = vec![0.0_f64; num_features * num_buckets];
        router_backward_scatter_cpu(&d_logits, &indices, &mut grad_weight, 2, num_buckets, num_features, 2);
        assert_eq!(grad_weight, vec![4.0_f64, 6.0, 0.0, 0.0]);
    }

    /// 既存値への accumulate (0 初期化しない呼び出し) も足し込みになる。
    #[test]
    fn accumulates_onto_existing_values() {
        let indices = vec![0_i32];
        let d_logits = vec![5.0_f64];
        let mut grad_weight = vec![10.0_f64];
        router_backward_scatter_cpu(&d_logits, &indices, &mut grad_weight, 1, 1, 1, 1);
        assert_eq!(grad_weight, vec![15.0_f64]);
    }

    /// `idx >= num_features` は silent skip。
    #[test]
    fn out_of_range_index_is_silently_skipped() {
        let indices = vec![5_i32];
        let d_logits = vec![1.0_f64];
        let mut grad_weight = vec![0.0_f64; 2];
        router_backward_scatter_cpu(&d_logits, &indices, &mut grad_weight, 1, 1, 2, 1);
        assert_eq!(grad_weight, vec![0.0_f64, 0.0]);
    }
}
