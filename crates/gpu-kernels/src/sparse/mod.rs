//! Sparse FT kernel suite (HalfKA_hm 入力層用) の reference CPU 実装。
//!
//! 1 kernel = 1 file で配置する。GPU 側 `#[kernel]` の定義は
//! `bins/nnue_train/src/kernels/` 側に置く (cuda-oxide rustc-codegen-cuda
//! backend は bin crate 経由で到達可能な kernel しか PTX 化しないため)。
//!
//! ## 提供する module
//!
//! - `sparse_ft_forward` — HalfKA_hm sparse feature transform forward
//! - `sparse_ft_backward` — 同 backward、atomics scatter
//! - `ft_factorize` — FT factorizer の fold (forward 用畳み込み weight 生成) /
//!   reduce (同じ仮想行に対応する実行勾配の縮約)
//! - `router_forward` / `router_backward` — `--bucket-mode router` の GPU
//!   router (KP-absolute bucket 選択ネットワーク) forward / backward
//!   (atomics scatter)。`sparse_ft_forward`/`sparse_ft_backward` と同じ sparse
//!   embedding パターンだが f64 精度・index-major layout

pub mod ft_factorize;
pub mod router_backward;
pub mod router_forward;
pub mod sparse_ft_backward;
pub mod sparse_ft_forward;
