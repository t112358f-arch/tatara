//! `--router-arch ft-by-ft` のレイアウト計算・loss・重み・CPU学習は
//! [`shogi_features::router_ftbyft`] に実装されている
//! (`crates/nnue-train::trainer` の学習ループから直接呼べるようにするため、
//! `bins/nnue_train` 専用ではなく `shogi-features` crate に置いてある)。
//! ここは既存の `crate::router_ftbyft::...` 参照を壊さないための再export。
pub use shogi_features::router_ftbyft::*;
