//! `arch_kind` module — NNUE network アーキテクチャの種別。
//!
//! 入力 feature set (`shogi-features` の `FeatureSet`) とは独立した軸で、層構成
//! (bucket / PSQT / skip 接続の有無、weight group 数) と host training pipeline
//! の分岐を決める。学習 artifact / checkpoint が「どのアーキで学習されたか」を
//! 記録し、別アーキの weight を取り違えて読み込まないために、シリアライズ層で
//! ある本 crate に置く。

/// NNUE network のアーキテクチャ種別。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArchKind {
    /// progress8kpabs 9-bucket LayerStack (FT → bucket 化 L1/L2/L3 + L1f skip)。
    LayerStack,
    /// LayerStack V3: `ft_out` は bucket 間で共通、`l1_out`/`l2_out` は
    /// bucketごとに個別サイズを持てる ([`crate::layerstack_v3_weights`])。
    /// bucket 割り当て方式 (kingrank9 / progress8kpabs / progress9kpabs) は
    /// engine 側の実行時オプションで選ぶため、本 variant 自体はその方式を
    /// 区別しない (bucket 数は常に9固定)。
    LayerStackV3,
    /// bucket 無しの 4 層 dense アーキ (FT → L1 → L2 → L3)。
    Simple,
}

impl ArchKind {
    /// 全アーキ種別。
    pub const ALL: [ArchKind; 3] = [ArchKind::LayerStack, ArchKind::LayerStackV3, ArchKind::Simple];

    /// CLI サブコマンド名 / artifact identity が扱う flat な canonical 名。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            ArchKind::LayerStack => "layerstack",
            ArchKind::LayerStackV3 => "layerstack_v3",
            ArchKind::Simple => "simple",
        }
    }

    /// canonical 名から逆引きする。未知の名前は `None`。
    pub fn from_canonical_name(name: &str) -> Option<ArchKind> {
        ArchKind::ALL
            .into_iter()
            .find(|a| a.canonical_name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_name_round_trips() {
        for arch in ArchKind::ALL {
            assert_eq!(
                ArchKind::from_canonical_name(arch.canonical_name()),
                Some(arch)
            );
        }
    }

    #[test]
    fn from_canonical_name_rejects_unknown() {
        assert_eq!(ArchKind::from_canonical_name("bogus"), None);
        assert_eq!(ArchKind::from_canonical_name(""), None);
    }
}
