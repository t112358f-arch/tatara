//! `--bucket-mode router` の GPU-resident 学習用 plain Adam optimizer step の
//! reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn router_adam_step_f64`) は `bins/nnue_train/src/kernels/`
//! に定義されている (cuda-oxide rustc-codegen-cuda backend の bin-entry 制約、
//! 他の kernel と同様)。本 module の `router_adam_step_cpu` は GPU と同じ
//! ロジックを host に書き写したもので、GPU↔CPU 数値同等性テストの reference
//! に使う。
//!
//! ## `radam_step` との違い
//!
//! `pointwise::radam_step` (Rectified Adam, `f32`, weight decay を掛けてから
//! update する AdamW、`min_w`/`max_w` clamp あり) と異なり、本 kernel は
//! `shogi_features::router_kpabs` の (crate-private) `adam_step` 関数と全く同じ
//! **plain Adam** (bias correction あり、weight decay は `grad += decay * w`
//! として grad に足すだけ、clamp 無し) を `f64` 精度で計算する:
//!
//! ```text
//! g          = grad[i] + weight_decay * w[i]
//! m[i]       = beta1 * m[i] + (1 - beta1) * g
//! v[i]       = beta2 * v[i] + (1 - beta2) * g * g
//! m_hat      = m[i] / bias_correction1        # = 1 - beta1^t
//! v_hat      = v[i] / bias_correction2        # = 1 - beta2^t
//! w[i]      -= lr * m_hat / (sqrt(v_hat) + eps)
//! ```
//!
//! `bias_correction1` / `bias_correction2` は host 側で step 番号 `t` から
//! 事前計算し (`radam_step` の `step_size`/`denom` と同じ流儀)、kernel には
//! 値渡しする。
//!
//! ## GPU kernel の scalar 精度についての注記
//!
//! `router_forward`/`router_backward` の buffer (`weight`/`m`/`v`/`grad`) は
//! `f64` だが、cuda-oxide kernel の scalar 引数に `f64` を直接渡す前例が本
//! codebase に無いため (`radam_step` 系はすべて `f32` scalar)、GPU kernel 側は
//! `lr`/`weight_decay`/`beta1`/`beta2`/`eps`/`bias_correction1`/
//! `bias_correction2` を `f32` で受け取り、kernel 内で `f64` へ upcast してから
//! 上記の式を計算する (buffer 自体は f64 のまま、精度劣化は scalar の丸め誤差
//! ~1e-7 程度に留まる)。本 CPU reference はその GPU 側の scalar 丸めを
//! 再現するため、`f32` 引数を受けて内部で `f64` にキャストする。host 側
//! (`bins/nnue_train::router_gpu`) の完全 `f64` CPU 実装
//! (`shogi_features::router_kpabs` 内部の `adam_step`) とは、この scalar 丸めの
//! 分だけ bit-exact ではなくなる (同数値傾向、近似同等)。

/// Reference CPU 実装 (GPU kernel の `f32` scalar 精度を再現)。
///
/// In-place 更新: `weights` / `m` / `v` を書き換える (`grad` は読み取りのみ、
/// `sparse_ft` 系の RAdam kernel と違い呼び出し側で zero-reset しない —
/// router は毎 step `grad_weight` を新規 0 初期化した buffer に scatter する
/// ため reset 不要)。
///
/// 入力前提: `weights.len() == m.len() == v.len() == grad.len() == n`。
#[allow(clippy::too_many_arguments)]
pub fn router_adam_step_cpu(
    weights: &mut [f64],
    m: &mut [f64],
    v: &mut [f64],
    grad: &[f64],
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) {
    let lr = lr as f64;
    let weight_decay = weight_decay as f64;
    let beta1 = beta1 as f64;
    let beta2 = beta2 as f64;
    let eps = eps as f64;
    let bias_correction1 = bias_correction1 as f64;
    let bias_correction2 = bias_correction2 as f64;

    for i in 0..weights.len() {
        let g = grad[i] + weight_decay * weights[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] / bias_correction1;
        let v_hat = v[i] / bias_correction2;
        weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// beta_correction を `t=1` 相当 (`1 - beta^1`) にして、既知の単純ケース
    /// (`m=v=0` 初期状態、weight_decay=0) で 1 step 分の更新式を検算する。
    #[test]
    fn matches_hand_computed_first_step() {
        let mut weights = vec![1.0_f64];
        let mut m = vec![0.0_f64];
        let mut v = vec![0.0_f64];
        let grad = vec![0.5_f64];
        let (lr, wd, beta1, beta2, eps) = (0.1_f32, 0.0_f32, 0.9_f32, 0.999_f32, 1e-8_f32);
        let bc1 = 1.0 - 0.9_f32;
        let bc2 = 1.0 - 0.999_f32;

        router_adam_step_cpu(&mut weights, &mut m, &mut v, &grad, lr, wd, beta1, beta2, eps, bc1, bc2);

        // g = 0.5, m = 0.1*0.5 = 0.05, v = 0.001*0.25 = 0.00025
        // m_hat = 0.05 / 0.1 = 0.5, v_hat = 0.00025 / 0.001 = 0.25
        // w -= 0.1 * 0.5 / (0.5 + 1e-8) ≈ 0.1 * 1.0 = 0.1 → w ≈ 0.9
        assert!((weights[0] - 0.9).abs() < 1e-5, "weights[0]={}", weights[0]);
    }

    /// 繰り返し同じ勾配方向で学習させると loss (ここでは |grad| 相当) が
    /// 単調に効き、weight が grad と逆方向に動き続ける (発散しない) ことを
    /// 確認する健全性チェック。
    #[test]
    fn repeated_steps_move_weight_toward_lower_loss() {
        let mut weights = vec![5.0_f64];
        let mut m = vec![0.0_f64];
        let mut v = vec![0.0_f64];
        let (lr, wd, beta1, beta2, eps) = (0.1_f32, 0.0_f32, 0.9_f32, 0.999_f32, 1e-8_f32);
        for t in 1..=50u32 {
            let grad = vec![weights[0]]; // d(0.5*w^2)/dw = w
            let bc1 = 1.0 - 0.9_f32.powi(t as i32);
            let bc2 = 1.0 - 0.999_f32.powi(t as i32);
            router_adam_step_cpu(&mut weights, &mut m, &mut v, &grad, lr, wd, beta1, beta2, eps, bc1, bc2);
        }
        assert!(weights[0] < 5.0, "weight should have decreased toward the minimum at 0, got {}", weights[0]);
    }
}
