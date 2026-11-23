//! `--output-format yaneuraou` + Threat 特徴量の書き出しテスト (公開APIのみ使用)。
//!
//! YaneuraOu 側の読み込み (`FeatureTransformer::ReadParameters` / `threat.h`) との契約:
//!   * arch 文字列: `Features=...[73305->Nx2],Threat=<dims>[,ThreatProfile=<id>],Network=...`
//!   * FT 重み (LEB128, base のみ) の直後に `[ThreatProfile u32 (id != 0)] [Threat 重み raw i8]`
//!   * Threat 重みは feature-major (`[threat_index][ft_out]`)、量子化は base FT と同じ QA(=127)、±127 に clamp

use nnue_format::{LayerStackWeights, save_yaneuraou, save_yaneuraou_combined};
use shogi_features::bucket_mode::BucketMode;
use shogi_features::{FeatureSet, FeatureSetSpec, ThreatProfile, threat_dimensions_of};

const PROFILES: [(ThreatProfile, u32); 6] = [
    (ThreatProfile::Full, 0),
    (ThreatProfile::SameClass, 1),
    (ThreatProfile::SameClassMajorPawn, 2),
    (ThreatProfile::StepAttacker, 3),
    (ThreatProfile::FullSymDedup, 4),
    (ThreatProfile::CrossSide, 10),
];

fn bucket_mode() -> BucketMode {
    BucketMode::parse("k9k9").expect("k9k9 is a valid bucket mode")
}

fn spec(profile: Option<ThreatProfile>) -> FeatureSetSpec {
    let base = FeatureSet::HalfKaHmMerged.spec();
    match profile {
        Some(p) => base.with_threat_profile(p),
        None => base,
    }
}

fn zeroed(profile: Option<ThreatProfile>, ft_out: usize) -> LayerStackWeights {
    let bm = bucket_mode();
    LayerStackWeights::zeroed(spec(profile), ft_out, 16, 32, bm.total_buckets() as usize)
}

fn export(weights: &LayerStackWeights) -> Vec<u8> {
    let mut out = Vec::new();
    save_yaneuraou(&mut out, weights, bucket_mode(), None).expect("save_yaneuraou");
    out
}

fn arch_string(bytes: &[u8]) -> String {
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    String::from_utf8(bytes[12..12 + len].to_vec()).unwrap()
}

#[test]
fn arch_string_carries_threat_token() {
    let s = arch_string(&export(&zeroed(Some(ThreatProfile::Full), 32)));
    // profile 0 は ThreatProfile= を省略。[..] は Threat を含まない base の次元数。
    assert!(s.contains("[73305->32x2],Threat=216720,Network="), "{s}");
    assert!(!s.contains("ThreatProfile="), "{s}");

    let s = arch_string(&export(&zeroed(Some(ThreatProfile::FullSymDedup), 32)));
    assert!(s.contains("[73305->32x2],Threat=216720,ThreatProfile=4,Network="), "{s}");

    let s = arch_string(&export(&zeroed(Some(ThreatProfile::StepAttacker), 32)));
    assert!(s.contains("[73305->32x2],Threat=33408,ThreatProfile=3,Network="), "{s}");

    let s = arch_string(&export(&zeroed(Some(ThreatProfile::CrossSide), 32)));
    assert!(s.contains("[73305->32x2],Threat=96320,ThreatProfile=10,Network="), "{s}");
}

#[test]
fn threat_block_follows_ft_as_raw_i8() {
    let ft_out = 32;
    let base_len = FeatureSet::HalfKaHmMerged.spec().ft_in() * ft_out;
    for (profile, id) in PROFILES {
        let dims = threat_dimensions_of(profile);
        let block_len = dims * ft_out;
        let mut with_threat = zeroed(Some(profile), ft_out);
        let zero = zeroed(Some(profile), ft_out);
        assert_eq!(with_threat.ft_w.len(), base_len + block_len, "{profile}");
        assert_eq!(profile.profile_id(), id);

        with_threat.ft_w[base_len] = 0.5; // 0.5 * 127 = 63.5 -> 64
        with_threat.ft_w[base_len + 1] = 2.0; // +127 に飽和
        with_threat.ft_w[base_len + 2] = -2.0; // -127 に飽和 (-128 は使わない)
        with_threat.ft_w[base_len + block_len - 1] = -0.5; // -64

        let a = export(&with_threat);
        let b = export(&zero);
        assert_eq!(a.len(), b.len(), "{profile}: size must not depend on values");

        // 差分は Threat block の中だけ。最初の差分が block 先頭 (= Threat 重み 0 番)。
        let first = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
        let last = a.iter().zip(&b).rposition(|(x, y)| x != y).unwrap();
        assert_eq!(last, first + block_len - 1, "{profile}");
        assert_eq!(a[first], 64u8, "{profile}");
        assert_eq!(a[first + 1], 127u8, "{profile}");
        assert_eq!(a[first + 2], (-127i8) as u8, "{profile}");
        assert_eq!(a[last], (-64i8) as u8, "{profile}");
        // profile id が 0 以外のときは block の直前に u32 LE (rshogi と同じ配置)。
        if id != 0 {
            assert_eq!(a[first - 4..first], id.to_le_bytes(), "{profile}");
        }
    }
}

#[test]
fn feature_major_layout_places_row_then_column() {
    // Threat 重み [threat_index][ft_out]: index=3, 列=5 の1要素だけが立つ位置を検証する。
    let ft_out = 32;
    let base_len = FeatureSet::HalfKaHmMerged.spec().ft_in() * ft_out;
    let profile = ThreatProfile::StepAttacker;
    let mut w = zeroed(Some(profile), ft_out);
    w.ft_w[base_len + 3 * ft_out + 5] = 0.25; // -> 32 (31.75 -> 32)
    let a = export(&w);
    let b = export(&zeroed(Some(profile), ft_out));
    let first = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
    let diffs = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(diffs, 1);
    // block 先頭は profile id (4B) の直後。差分位置の block 先頭からのオフセットが 3*ft_out+5。
    let block_start = {
        let mut z = zeroed(Some(profile), ft_out);
        z.ft_w[base_len] = 1.0;
        let c = export(&z);
        c.iter().zip(&b).position(|(x, y)| x != y).unwrap()
    };
    assert_eq!(first - block_start, 3 * ft_out + 5);
    assert_eq!(a[first], 32u8);
}

#[test]
fn threat_free_export_has_no_threat_token() {
    let s = arch_string(&export(&zeroed(None, 32)));
    assert!(!s.contains("Threat"), "{s}");
}

#[test]
fn combined_ft_by_ft_export_rejects_threat() {
    use shogi_features::router_ftbyft::FtByFtLayout;

    let ft_out = 32;
    let bm = BucketMode::parse("routerft2ft2").expect("routerft2ft2");
    let layout = FtByFtLayout::new(bm.total_buckets() as usize, ft_out).unwrap();
    let weights = LayerStackWeights::zeroed(
        spec(Some(ThreatProfile::StepAttacker)),
        ft_out,
        16,
        32,
        bm.total_buckets() as usize,
    );
    let router_w = vec![0.0f32; weights.feature_set.ft_in() * layout.r];
    let router_b = vec![0.0f32; layout.r];
    let mut out = Vec::new();
    let err = save_yaneuraou_combined(&mut out, &weights, bm, &layout, &router_w, &router_b)
        .expect_err("threat + ft-by-ft must be rejected");
    assert!(err.to_string().contains("Threat"), "{err}");
}
