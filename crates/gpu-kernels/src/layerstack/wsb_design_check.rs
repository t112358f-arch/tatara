//! `wsb` (WithSharedBucket) の学習側設計 ── 「選択bucketと共有bucketの2つの
//! forwardの平均を出力とし、backwardはその0.5分割 + 共有入力への勾配の和」
//! ── を、本番 kernel の CPU reference (このファイルの `*_cpu` 関数、
//! `gpu_cpu_equivalence_tests` が本番 GPU kernel との数値一致を保証している)
//! と [`crate::layerstack::crelu`] を使い、有限差分で数値検証する。
//!
//! `docs/decisions/2026-09-16-wsb-shared-bucket.md` §2b の実装レシピの核心
//! ("選択bucket分・共有bucket分をそれぞれ既存の汎用 per-bucket kernel に
//! そのまま渡すだけで正しい forward/backward になる") が正しいことの根拠。
//! `trainer_layerstack.rs` 自体の GPU kernel launch orchestration はまだ
//! 実装されていない (cuda-oxide のビルド・実行検証が必要) — 本テストは
//! そこで使う予定の *kernel呼び出しパターン* が数学的に正しいことだけを
//! 検証する。
#[cfg(test)]
mod wsb_design_check {
    use crate::layerstack::crelu::{crelu_fwd_cpu, crelu_grad_cpu};
    use crate::layerstack::dense_mm_bucket::{
        bias_grad_bucket_cpu, dense_mm_bwd_input_bucket_cpu, dense_mm_bwd_weight_bucket_cpu,
        dense_mm_fwd_bucket_cpu,
    };

    struct Params {
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
    }

    struct Dims {
        batch: usize,
        in_dim: usize,
        h: usize,
        total_buckets: usize,
    }

    /// WSBの「batch doubling」設計そのもの: FT出力 x (shared) -> L1 (per-bucket
    /// affine) -> crelu -> L2 (per-bucket affine, out=1) の2層トイネットで、
    /// 選択bucketと共有bucket、それぞれについて**既存の汎用per-bucket kernel を
    /// そのまま**呼び、forwardの平均・backwardの0.5分割+共有入力勾配の和を計算する。
    #[allow(clippy::too_many_arguments)]
    fn forward_backward_wsb(
        dims: &Dims,
        p: &Params,
        x: &[f32],
        bucket_idx: &[i32],
        shared_bucket: i32,
        target: &[f32],
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let Dims { batch, in_dim, h, total_buckets } = *dims;
        let shared_idx = vec![shared_bucket; batch];

        // -- forward: 選択branch / 共有branch、それぞれ既存kernelをそのまま1回ずつ --
        let mut l1_sel = vec![0.0_f32; batch * h];
        dense_mm_fwd_bucket_cpu(x, &p.w1, &p.b1, bucket_idx, &mut l1_sel, batch, in_dim, h, total_buckets);
        let mut l1_shr = vec![0.0_f32; batch * h];
        dense_mm_fwd_bucket_cpu(x, &p.w1, &p.b1, &shared_idx, &mut l1_shr, batch, in_dim, h, total_buckets);

        let mut h_sel = vec![0.0_f32; batch * h];
        crelu_fwd_cpu(&l1_sel, &mut h_sel, batch * h);
        let mut h_shr = vec![0.0_f32; batch * h];
        crelu_fwd_cpu(&l1_shr, &mut h_shr, batch * h);

        let mut y_sel = vec![0.0_f32; batch];
        dense_mm_fwd_bucket_cpu(&h_sel, &p.w2, &p.b2, bucket_idx, &mut y_sel, batch, h, 1, total_buckets);
        let mut y_shr = vec![0.0_f32; batch];
        dense_mm_fwd_bucket_cpu(&h_shr, &p.w2, &p.b2, &shared_idx, &mut y_shr, batch, h, 1, total_buckets);

        let y_avg: Vec<f32> = (0..batch).map(|i| 0.5 * (y_sel[i] + y_shr[i])).collect();
        let loss: f32 = (0..batch).map(|i| { let d = y_avg[i] - target[i]; 0.5 * d * d }).sum();

        // -- backward: dL/dy_avg を0.5ずつ両branchへ配る --
        let dy: Vec<f32> = (0..batch).map(|i| 0.5 * (y_avg[i] - target[i])).collect();

        let mut dh_sel = vec![0.0_f32; batch * h];
        dense_mm_bwd_input_bucket_cpu(&dy, &p.w2, bucket_idx, &mut dh_sel, batch, h, 1, total_buckets);
        let mut dh_shr = vec![0.0_f32; batch * h];
        dense_mm_bwd_input_bucket_cpu(&dy, &p.w2, &shared_idx, &mut dh_shr, batch, h, 1, total_buckets);

        // grad_w2 / grad_b2: 選択branch分の呼出しは対象index (`bucket_idx`が指す
        // bucket群) だけ書き、共有branch分の呼出しは共有bucket 1個だけ書く
        // (`dense_mm_bwd_weight_bucket_cpu`は「対象外bucketには触れない」実装 --
        // 本番kernelがoverwrite-all-bucketsだった場合は別バッファ+結合が必要、
        // ADR §2b 参照)。ここでは同じ配列に2回呼んでも安全であることも検証する。
        let mut grad_w2 = vec![0.0_f32; total_buckets * h];
        dense_mm_bwd_weight_bucket_cpu(&h_sel, &dy, bucket_idx, &mut grad_w2, batch, h, 1, total_buckets);
        let mut grad_w2_shared_only = vec![0.0_f32; total_buckets * h];
        dense_mm_bwd_weight_bucket_cpu(&h_shr, &dy, &shared_idx, &mut grad_w2_shared_only, batch, h, 1, total_buckets);
        for i in 0..grad_w2.len() {
            grad_w2[i] += grad_w2_shared_only[i];
        }

        let mut grad_b2 = vec![0.0_f32; total_buckets];
        bias_grad_bucket_cpu(&dy, bucket_idx, &mut grad_b2, batch, 1, total_buckets);
        bias_grad_bucket_cpu(&dy, &shared_idx, &mut grad_b2, batch, 1, total_buckets);

        let mut dl1_sel = vec![0.0_f32; batch * h];
        crelu_grad_cpu(&l1_sel, &dh_sel, &mut dl1_sel, batch * h);
        let mut dl1_shr = vec![0.0_f32; batch * h];
        crelu_grad_cpu(&l1_shr, &dh_shr, &mut dl1_shr, batch * h);

        let mut dx_sel = vec![0.0_f32; batch * in_dim];
        dense_mm_bwd_input_bucket_cpu(&dl1_sel, &p.w1, bucket_idx, &mut dx_sel, batch, in_dim, h, total_buckets);
        let mut dx_shr = vec![0.0_f32; batch * in_dim];
        dense_mm_bwd_input_bucket_cpu(&dl1_shr, &p.w1, &shared_idx, &mut dx_shr, batch, in_dim, h, total_buckets);

        let mut grad_w1 = vec![0.0_f32; total_buckets * h * in_dim];
        dense_mm_bwd_weight_bucket_cpu(x, &dl1_sel, bucket_idx, &mut grad_w1, batch, in_dim, h, total_buckets);
        let mut grad_w1_shared_only = vec![0.0_f32; total_buckets * h * in_dim];
        dense_mm_bwd_weight_bucket_cpu(x, &dl1_shr, &shared_idx, &mut grad_w1_shared_only, batch, in_dim, h, total_buckets);
        for i in 0..grad_w1.len() {
            grad_w1[i] += grad_w1_shared_only[i];
        }

        let mut grad_b1 = vec![0.0_f32; total_buckets * h];
        bias_grad_bucket_cpu(&dl1_sel, bucket_idx, &mut grad_b1, batch, h, total_buckets);
        bias_grad_bucket_cpu(&dl1_shr, &shared_idx, &mut grad_b1, batch, h, total_buckets);

        // x はFT出力(shared)なので、両branchの寄与を足す (fc_0/L1入力へ流れる
        // 勾配の合算 -- ADR の "l1f" summing則と同じ原理)。
        let grad_x: Vec<f32> = (0..batch * in_dim).map(|i| dx_sel[i] + dx_shr[i]).collect();

        (loss, grad_w1, grad_b1, grad_w2, grad_b2, grad_x)
    }

    /// batch doubling / 個別呼出しを一切使わない、疑いようのないreference
    /// (finite-difference の基準)。
    fn loss_only(dims: &Dims, p: &Params, x: &[f32], bucket_idx: &[i32], shared_bucket: i32, target: &[f32]) -> f32 {
        let Dims { batch, in_dim, h, total_buckets: _ } = *dims;
        let mut loss = 0.0_f32;
        for i in 0..batch {
            let branch = |bucket: i32| -> f32 {
                let g = bucket as usize;
                let mut hbuf = vec![0.0_f32; h];
                for oi in 0..h {
                    let mut s = p.b1[g * h + oi];
                    for k in 0..in_dim {
                        s += x[i * in_dim + k] * p.w1[g * h * in_dim + oi * in_dim + k];
                    }
                    hbuf[oi] = s.clamp(0.0, 1.0);
                }
                let mut y = p.b2[g];
                for oi in 0..h {
                    y += hbuf[oi] * p.w2[g * h + oi];
                }
                y
            };
            let y_avg = 0.5 * (branch(bucket_idx[i]) + branch(shared_bucket));
            let d = y_avg - target[i];
            loss += 0.5 * d * d;
        }
        loss
    }

    fn xorshift(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (((*state >> 40) as i64 as f64) / (1u64 << 24) as f64) as f32
    }

    #[test]
    fn wsb_forward_matches_naive_branch_average() {
        let dims = Dims { batch: 5, in_dim: 4, h: 3, total_buckets: 4 };
        let shared_bucket = 3_i32;
        let mut rng = 0x243F6A8885A308D3_u64;
        let mut randf = || (xorshift(&mut rng) * 2.0 - 1.0) * 0.5;

        let p = Params {
            w1: (0..dims.total_buckets * dims.h * dims.in_dim).map(|_| randf()).collect(),
            b1: (0..dims.total_buckets * dims.h).map(|_| randf()).collect(),
            w2: (0..dims.total_buckets * dims.h).map(|_| randf()).collect(),
            b2: (0..dims.total_buckets).map(|_| randf()).collect(),
        };
        let x: Vec<f32> = (0..dims.batch * dims.in_dim).map(|_| randf()).collect();
        let bucket_idx: Vec<i32> = (0..dims.batch).map(|i| (i % 3) as i32).collect();
        let target: Vec<f32> = (0..dims.batch).map(|_| randf()).collect();

        let (loss, ..) = forward_backward_wsb(&dims, &p, &x, &bucket_idx, shared_bucket, &target);
        let loss_ref = loss_only(&dims, &p, &x, &bucket_idx, shared_bucket, &target);
        assert!((loss - loss_ref).abs() < 1e-5, "forward mismatch: {loss} vs {loss_ref}");
    }

    #[test]
    fn wsb_backward_matches_finite_differences() {
        let dims = Dims { batch: 5, in_dim: 4, h: 3, total_buckets: 4 };
        let shared_bucket = 3_i32;
        let mut rng = 0x243F6A8885A308D3_u64;
        let mut randf = || (xorshift(&mut rng) * 2.0 - 1.0) * 0.5;

        let p = Params {
            w1: (0..dims.total_buckets * dims.h * dims.in_dim).map(|_| randf()).collect(),
            b1: (0..dims.total_buckets * dims.h).map(|_| randf()).collect(),
            w2: (0..dims.total_buckets * dims.h).map(|_| randf()).collect(),
            b2: (0..dims.total_buckets).map(|_| randf()).collect(),
        };
        let x: Vec<f32> = (0..dims.batch * dims.in_dim).map(|_| randf()).collect();
        let bucket_idx: Vec<i32> = (0..dims.batch).map(|i| (i % 3) as i32).collect();
        let target: Vec<f32> = (0..dims.batch).map(|_| randf()).collect();

        let (_loss, grad_w1, grad_b1, grad_w2, grad_b2, grad_x) =
            forward_backward_wsb(&dims, &p, &x, &bucket_idx, shared_bucket, &target);

        let eps = 1e-3_f32;
        let floor = 5e-2_f32; // f32 finite-difference noise floor at this loss/eps scale
        let mut max_rel_err = 0.0_f32;

        macro_rules! check_param {
            ($field:ident, $analytic:expr) => {{
                let mut pm = Params { w1: p.w1.clone(), b1: p.b1.clone(), w2: p.w2.clone(), b2: p.b2.clone() };
                for idx in 0..pm.$field.len() {
                    let orig = pm.$field[idx];
                    pm.$field[idx] = orig + eps;
                    let l_plus = loss_only(&dims, &pm, &x, &bucket_idx, shared_bucket, &target);
                    pm.$field[idx] = orig - eps;
                    let l_minus = loss_only(&dims, &pm, &x, &bucket_idx, shared_bucket, &target);
                    pm.$field[idx] = orig;
                    let numeric = (l_plus - l_minus) / (2.0 * eps);
                    let analytic = $analytic[idx];
                    let rel = (numeric - analytic).abs() / (numeric.abs().max(analytic.abs()).max(floor));
                    max_rel_err = max_rel_err.max(rel);
                    assert!(rel < 1.0, "{}[{idx}] mismatch: numeric={numeric} analytic={analytic}", stringify!($field));
                }
            }};
        }
        check_param!(w1, grad_w1);
        check_param!(b1, grad_b1);
        check_param!(w2, grad_w2);
        check_param!(b2, grad_b2);

        // grad_x (共有入力 x への勾配 -- 両branchの和になっているはず)
        let mut xm = x.clone();
        for idx in 0..xm.len() {
            let orig = xm[idx];
            xm[idx] = orig + eps;
            let l_plus = loss_only(&dims, &p, &xm, &bucket_idx, shared_bucket, &target);
            xm[idx] = orig - eps;
            let l_minus = loss_only(&dims, &p, &xm, &bucket_idx, shared_bucket, &target);
            xm[idx] = orig;
            let numeric = (l_plus - l_minus) / (2.0 * eps);
            let analytic = grad_x[idx];
            let rel = (numeric - analytic).abs() / (numeric.abs().max(analytic.abs()).max(floor));
            max_rel_err = max_rel_err.max(rel);
            assert!(rel < 1.0, "grad_x[{idx}] mismatch: numeric={numeric} analytic={analytic}");
        }

        assert!(max_rel_err < 5e-2, "gradient check FAILED (max_rel_err={max_rel_err})");
    }
}
