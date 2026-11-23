//! 検証用: 乱数重みの LayerStack を `save_yaneuraou` で書き出す (Threat 有り/無し)。
//! YaneuraOu 側のローダとの実ファイルでの突き合わせ (end-to-end) に使う。
//!
//! usage: export_yo_threat <out.bin> <ft_out> <bucket-mode> [threat-profile-id|none]
use nnue_format::{LayerStackWeights, save_yaneuraou};
use shogi_features::bucket_mode::BucketMode;
use shogi_features::{FeatureSet, ThreatProfile};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = &args[1];
    let ft_out: usize = args[2].parse().unwrap();
    let bm = BucketMode::parse(&args[3]).unwrap();
    let profile = match args.get(4).map(String::as_str) {
        None | Some("none") => None,
        Some(id) => Some(match id {
            "0" => ThreatProfile::Full,
            "1" => ThreatProfile::SameClass,
            "2" => ThreatProfile::SameClassMajorPawn,
            "3" => ThreatProfile::StepAttacker,
            "4" => ThreatProfile::FullSymDedup,
            "10" => ThreatProfile::CrossSide,
            other => panic!("unknown profile id {other}"),
        }),
    };
    let base = FeatureSet::HalfKaHmMerged.spec();
    let spec = match profile {
        Some(p) => base.with_threat_profile(p),
        None => base,
    };
    let mut w = LayerStackWeights::zeroed(spec, ft_out, 16, 32, bm.total_buckets() as usize);

    // 決定的な疑似乱数 (xorshift)。値域は各層で「評価値が0に潰れない」程度。
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move |scale: f32| -> f32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
    };
    for v in w.ft_w.iter_mut() {
        *v = rnd(0.15);
    }
    for v in w.l1_w.iter_mut().chain(w.l1f_w.iter_mut()) {
        *v = rnd(0.2);
    }
    for v in w.l2_w.iter_mut().chain(w.l3_w.iter_mut()) {
        *v = rnd(0.3);
    }
    let mut f = std::fs::File::create(out).unwrap();
    save_yaneuraou(&mut f, &w, bm, None).unwrap();
    println!("wrote {out}");
}
