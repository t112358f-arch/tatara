//! YaneuraOu SFNNWithoutPsqt evaluation-file serialization.

use std::io::{self, Write};

use shogi_features::FeatureSet;
use shogi_features::bucket_mode::BucketMode;
use shogi_features::router_kpabs::RouterKPAbsWeights;

use crate::LayerStackWeights;
use crate::layerstack_weights::{QA, QB, write_leb128_tensor_i16};

const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;
/// `router` の trailing weight block を示す magic。全 network block (bucket 数
/// = `weights.num_buckets`) を書き終えた**直後**にだけ現れうる (kingrank9
/// export では出現しない)。
/// エンジン側 (`evaluate_nnue.cpp`) は network 群を読み終えた後、EOF を要求する前に
/// この 4 byte を peek し、一致すれば router block を読み、不一致なら読み戻して
/// 既存の EOF 検査に進む — 後方互換 (router 無しの既存ファイルはそのまま読める)。
pub const YO_ROUTER9KPABS_HASH: u32 = 0x526f_3944; // "Ro9D" (Router9kpabs Data) 由来

/// 新形式 (yaneuraou本家の慣習): router (kpabs) の重み block を示す magic。
/// FeatureTransformerの直後・Network群の**前**にだけ現れる
/// (`evaluate_nnue.h`の`RouterKPAbs::Parameters::GetHashValue()`と同じ値)。
/// 旧形式の [`YO_ROUTER9KPABS_HASH`] (Network群の**後ろ**) とは非互換 —
/// `tools/convert_router_bucket_layout.py` で変換すること。
pub const YO_ROUTER_KPABS_HASH: u32 = 0x6f52_544b; // "oRTK"

/// progress<N> バケットの重み block を示す magic。FeatureTransformerの直後・
/// [`YO_ROUTER_KPABS_HASH`]/Network群の**前**にだけ現れる
/// (`evaluate_nnue.h`の`Progress::Parameters::GetHashValue()`と同じ値)。
pub const YO_PROGRESS_HASH: u32 = 0x6f50_524f; // "oPRO"

/// KP-absolute 特徴の次元数 (`Eval::fe_end`)。`Progress::Parameters`/
/// `RouterKPAbs::Parameters` の重み行列の列数と対応する
/// (`shogi_features::progress_kpabs::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS` =
/// `SQ_NB * FE_END`)。
const SQ_NB: usize = 81;
const FE_END: usize = shogi_features::progress_kpabs::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS / SQ_NB;

/// `f32` の重み1個を `Progress::Parameters`/`RouterKPAbs::Parameters` の
/// Q16.16 固定小数点 (`std::int32_t`) に量子化する
/// (`evaluate_nnue.cpp` の `bias_q16_`/`weights_q16_` と同じスケール)。
fn quantize_q16(w: f64) -> i32 {
    let scaled = (w * 65536.0).round();
    scaled.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// `--bucket-mode` に progress<N> 成分があるときだけ、FTの直後に
/// `Progress::Parameters` (bias_q16_=0 固定 + weights_q16_[SQ_NB][FE_END]) を
/// 書く。tatara の `ShogiProgressKPAbs` にはbias項が無い (常に0) ため、
/// `bias_q16_` は常に0を書く。
fn write_progress_block<W: Write>(writer: &mut W, bucket_mode: BucketMode) -> io::Result<()> {
    if bucket_mode.progress.is_none() {
        return Ok(());
    }
    let weights = shogi_features::progress_kpabs::ShogiProgressKPAbs::snapshot_weights();
    if weights.len() != SQ_NB * FE_END {
        return invalid_input(format!(
            "progress weights have {} entries but SQ_NB*FE_END is {} (feature table mismatch)",
            weights.len(),
            SQ_NB * FE_END
        ));
    }
    write_u32(writer, YO_PROGRESS_HASH)?;
    write_u32(writer, 0u32)?; // bias_q16_ = 0 (tatara's progress model has no bias term)
    for &w in weights {
        writer.write_all(&quantize_q16(f64::from(w)).to_le_bytes())?;
    }
    Ok(())
}

/// `RouterKPAbsWeights` (bias無し、`w[idx * num_buckets + bucket]`、
/// `idx = sq * FE_END + piece` の f64 flat 配列) を、C++側
/// `RouterKPAbs::Parameters::ReadParameters` が期待する生の
/// `weights_q16_[SQ_NB][FE_END][num_buckets]` (Q16.16固定小数点、bias無し)
/// 形式で書く。
///
/// **注意**: [`RouterKPAbsWeights::write_to`] はこれとは別物 — tatara 独自の
/// resume用チェックポイント形式 (専用magic + `num_weights`/`num_buckets`
/// ヘッダ + f64 flat配列そのまま) であり、yaneuraou側は読めない。以前
/// `save_yaneuraou`/`save_yaneuraou_combined` が誤って `write_to` を直接
/// 呼んでいたため、出力ファイルのrouterセクションが期待より大きくなり
/// (ヘッダ12バイト分 + f64(8byte/値) と i32(4byte/値) の差分)、末尾で
/// `stream.peek() != EOF` となって `FileCloseError` になるバグがあった。
fn write_router_kpabs_block<W: Write>(
    writer: &mut W,
    router: &RouterKPAbsWeights,
) -> io::Result<()> {
    let n = router.num_buckets;
    let expected_len = SQ_NB * FE_END * n;
    if router.w.len() != expected_len {
        return invalid_input(format!(
            "router weights have {} entries but SQ_NB*FE_END*num_buckets is {}",
            router.w.len(),
            expected_len
        ));
    }
    for &w in &router.w {
        writer.write_all(&quantize_q16(w).to_le_bytes())?;
    }
    Ok(())
}


/// YaneuraOu SFNN の KingRank9 (kingrank9 bucket-mode) LayerStack 数。
///
/// この値は KingRank9 の「両玉の段」分岐数という固定レイアウトの名前であり、
/// SFNN が書き出せる LayerStack 数の上限ではない。`router` bucket-mode は任意の
/// `--num-buckets N` (`weights.num_buckets`) を持ち、engine 側もそれに合わせて
/// `LayerStacks=N` でビルドした router edition (例:
/// `YANEURAOU_ENGINE_SFNN_halfkahm2_1536_16_32_routerN`) で読む。`save_yaneuraou`
/// は `weights.num_buckets` をそのまま書き出す (9 に限定しない)。
pub const YANEURAOU_LAYER_STACKS: usize = 9;

/// `weights.num_buckets` の健全性ガード (0 / 壊れた値による overflow を防ぐ)。
/// kernel やファイル形式が課す上限ではなく typo guard。
const MAX_LAYER_STACKS: usize = 4096;

const MAX_FT_OUT: usize = 8192;
const MAX_HIDDEN_DIM: usize = 4096;

struct YoFeature {
    feature_set: FeatureSet,
    yo_name: &'static str,
    gen_key: &'static str,
}

const YO_FEATURES: [YoFeature; 5] = [
    YoFeature {
        feature_set: FeatureSet::HalfKp,
        yo_name: "HalfKP",
        gen_key: "halfkp",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaSplit,
        yo_name: "HalfKA1",
        gen_key: "halfka1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaMerged,
        yo_name: "HalfKA2",
        gen_key: "halfka2",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmSplit,
        yo_name: "HalfKA_hm1",
        gen_key: "halfkahm1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmMerged,
        yo_name: "HalfKA_hm2",
        gen_key: "halfkahm2",
    },
];

/// LayerStack weights を YaneuraOu SFNNWithoutPsqt 形式で書き出す。
///
/// feature set と各層次元は weights の shape から決定する。bucket 数は
/// `weights.num_buckets` をそのまま `LayerStack=N` として書き出す (KingRank9 の
/// `9` に限定しない — `router` bucket-mode は任意の N を持つ)。YaneuraOu SFNN が
/// 表現できない拡張 feature、PSQT は reject する。bucket routing mode 自体は
/// weights に含まれないため、caller は学習 config 等から KingRank9 (または
/// router) であることを確認してから呼ぶ必要がある。
///
/// `router` が `Some` のとき、全 `weights.num_buckets` 個の network を書いた
/// 直後に `router` の重み ([`YO_ROUTER9KPABS_HASH`] + `RouterKPAbsWeights::write_to`)
/// を追記する。`None` (routerを使わない export 等) では router block を出力しない。
///
/// **保存形式 (yaneuraou本家の慣習に合わせた新形式)**: router (kpabs) の重み
/// block は FeatureTransformerの直後・Network群の**前**に置く
/// (`evaluate_nnue.h`/`.cpp` の `RouterKPAbs::Parameters` と対応する読み順)。
/// これは旧tatara/yaneuraou-private形式 (router blockがNetwork群の**後ろ**)
/// とは非互換 — 旧形式のファイルは `tools/convert_router_bucket_layout.py`
/// で変換すること。
pub fn save_yaneuraou<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
    bucket_mode: BucketMode,
    router: Option<&RouterKPAbsWeights>,
) -> io::Result<()> {
    if bucket_mode.total_buckets() as usize != weights.num_buckets {
        return invalid_input(format!(
            "bucket_mode {:?} composes to {} total buckets but weights.num_buckets is {}",
            bucket_mode.canonical_token(),
            bucket_mode.total_buckets(),
            weights.num_buckets
        ));
    }
    let router_n_from_mode = match bucket_mode.router {
        Some(shogi_features::bucket_mode::RouterSubMode::Kpabs { n }) => Some(n as usize),
        Some(shogi_features::bucket_mode::RouterSubMode::FtByFt { .. }) => {
            return invalid_input(
                "bucket_mode has a routerft<R>ft<R> component; use save_yaneuraou_combined \
                 instead of save_yaneuraou for ft-by-ft nets",
            );
        }
        None => None,
    };
    match (router_n_from_mode, router) {
        (Some(n), Some(r)) if r.num_buckets != n => {
            return invalid_input(format!(
                "bucket_mode ...routerkpabs{n} but the given router has {} buckets",
                r.num_buckets
            ));
        }
        (None, Some(_)) => {
            return invalid_input(
                "a router was given but bucket_mode has no routerkpabs<N> component",
            );
        }
        (Some(_), None) => {
            return invalid_input(
                "bucket_mode has a routerkpabs<N> component but no router weights were given",
            );
        }
        _ => {}
    }

    let arch = architecture(weights)?;
    validate_weights(&arch, weights, FtOutAlignment::MustBeMultipleOf32)?;

    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;

    write_u32(writer, YO_VERSION)?;
    write_u32(writer, YO_TOP_HASH)?;
    let arch_string = arch_string(&arch, bucket_mode);
    write_u32(
        writer,
        u32::try_from(arch_string.len()).expect("architecture string length fits in u32"),
    )?;
    writer.write_all(arch_string.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_b, QA as f64))?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_w, QA as f64))?;

    write_progress_block(writer, bucket_mode)?;

    if let Some(router) = router {
        write_u32(writer, YO_ROUTER_KPABS_HASH)?;
        write_router_kpabs_block(writer, router)?;
    }

    for bucket in 0..arch.num_buckets {
        write_u32(writer, YO_NETWORK_HASH)?;

        // factorizer 共有項は通常 export 前に L1 へ fold 済み。未 fold の weights を
        // caller が渡した場合にも同じ推論 weight になるよう加算する。
        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1f_b[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out + output * ft_out + input]
                    + weights.l1f_w[input * l1_out + output]
            })
        });
        write_affine(writer, l1_biases, l1_weights, ft_out, l1_out)?;

        let l2_biases = (0..l2_out).map(|output| weights.l2_b[bucket * l2_out + output]);
        let l2_weights = (0..l2_out).flat_map(|output| {
            (0..l2_in)
                .map(move |input| weights.l2_w[bucket * l2_out * l2_in + output * l2_in + input])
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, l2_out)?;

        write_affine(
            writer,
            std::iter::once(weights.l3_b[bucket]),
            (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]),
            l2_out,
            1,
        )?;
    }

    Ok(())
}

/// `--router-arch ft-by-ft` の保存: 評価net本体の `weights` (L1/L2/L3 は
/// **nominal** な `ft_out_nominal` で確定済み、`weights.ft_b`/`weights.ft_w`
/// も *まだ* `ft_out_nominal` 幅のまま) と、別途学習した router 重み
/// (`router_w`: `(ft_in, r)` row-major、`router_b`: `(r)`) を
/// [`shogi_features::router_ftbyft::combine_ft_by_ft_columns`] で列方向に
/// 結合してから、通常の `save_yaneuraou` と同じ byte layout で書き出す。
///
/// [`save_yaneuraou`] との違いは次の 2 点のみ:
/// - FT 重み/bias ブロックは `ft_out_nominal + layout.r` 幅の**物理**結合
///   FT (`ft_b`/`ft_w` はこの関数が結合するので、呼び出し側は nominal 幅の
///   ままでよい)。
/// - `arch_string`/`network_hash`/`ft_hash` は **nominal** な
///   `ft_out_nominal` を使う (L1 側が実際に受け取る次元、
///   `kTransformedFeatureDimensions` と一致させる必要があるため) —
///   YaneuraOu 側は `kAccumulatorDimensions` (= nominal + R) を FT の物理幅
///   として別途 architecture header から知るので、ファイル内の
///   `arch_string`/`ft_hash` に物理幅を出す必要はない (`nnue_architecture.h`
///   / `nnue_arch_gen.py` 参照)。
/// - edition suffix は `ls<N>`/`k3k3` ではなく `router_ft{R}ft{R}`
///   (`R = layout.r`) になる — `_router_kpabs_{N}` 側 (既存 `save_yaneuraou`
///   がそのまま使われる) と違い、ft-by-ft はネットワーク構造自体が変わる
///   ため専用の edition が必要 (spec.md 8節)。
///
/// `weights.psqt_w` が `Some` の場合や `weights.num_buckets != layout.r *
/// layout.r` の場合はエラーを返す。
pub fn save_yaneuraou_combined<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
    bucket_mode: BucketMode,
    layout: &shogi_features::router_ftbyft::FtByFtLayout,
    router_w: &[f32],
    router_b: &[f32],
) -> io::Result<()> {
    match bucket_mode.router {
        Some(shogi_features::bucket_mode::RouterSubMode::FtByFt { r }) if r as usize == layout.r => {}
        Some(shogi_features::bucket_mode::RouterSubMode::FtByFt { r }) => {
            return invalid_input(format!(
                "bucket_mode ...routerft{r}ft{r} does not match layout.r ({})",
                layout.r
            ));
        }
        _ => {
            return invalid_input(
                "save_yaneuraou_combined requires bucket_mode to have a routerft<R>ft<R> \
                 component",
            );
        }
    }
    if bucket_mode.total_buckets() as usize != weights.num_buckets {
        return invalid_input(format!(
            "bucket_mode {:?} composes to {} total buckets but weights.num_buckets is {}",
            bucket_mode.canonical_token(),
            bucket_mode.total_buckets(),
            weights.num_buckets
        ));
    }
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not representable in YaneuraOu SFNN");
    }
    if weights.num_buckets != layout.num_buckets {
        return invalid_input(format!(
            "weights.num_buckets ({}) does not match layout.num_buckets ({})",
            weights.num_buckets, layout.num_buckets
        ));
    }
    let ft_out_nominal = weights.ft_b.len();
    if ft_out_nominal != layout.ft_out {
        return invalid_input(format!(
            "weights.ft_b.len() ({ft_out_nominal}) does not match layout.ft_out ({})",
            layout.ft_out
        ));
    }
    // spec.md 11節: SIMD 幅の整列チェックは `ft_out` 単体ではなく物理 FT 幅
    // `ft_out + R` (= `layout.accum_out`、`kAccumulatorDimensions` と対応)
    // に対して行う — `ft_out` 単体がこれを満たす必要はない
    // (例: ft_out=2296 は 32 の倍数ではないが、R=8 の ft-by-ft では
    // 2296+8=2304 が 32 の倍数になっていればよい)。
    if ft_out_nominal == 0 || ft_out_nominal > MAX_FT_OUT {
        return invalid_input(format!(
            "unsupported FT output dimension {ft_out_nominal} (expected a positive value up to \
             {MAX_FT_OUT})"
        ));
    }
    if !layout.accum_out.is_multiple_of(32) {
        return invalid_input(format!(
            "unsupported ft-by-ft physical FT width {} = ft_out ({ft_out_nominal}) + R ({}) \
             (expected a positive multiple of 32, per spec.md 11節 -- pick a --ft-out/\
             --num-buckets combination whose sum is 32-aligned)",
            layout.accum_out, layout.r
        ));
    }

    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|feature_set| feature_set.spec() == weights.feature_set)
        .ok_or_else(|| invalid_input_err("feature set is not representable in YaneuraOu SFNN"))?;
    let ft_in = weights.feature_set.ft_in();
    if weights.ft_w.len() != ft_in * ft_out_nominal {
        return invalid_input(format!(
            "ft_w length mismatch: expected {}, got {}",
            ft_in * ft_out_nominal,
            weights.ft_w.len()
        ));
    }
    if router_w.len() != ft_in * layout.r {
        return invalid_input(format!(
            "router_w length mismatch: expected {}, got {}",
            ft_in * layout.r,
            router_w.len()
        ));
    }
    if router_b.len() != layout.r {
        return invalid_input(format!(
            "router_b length mismatch: expected {}, got {}",
            layout.r,
            router_b.len()
        ));
    }

    let l1_out = weights.l1f_b.len();
    let l2_out = weights.l2_b.len().checked_div(weights.num_buckets).unwrap_or(0);
    let l2_in = (l1_out - 1) * 2;
    validate_weights(
        &Architecture {
            feature_set,
            ft_out: ft_out_nominal,
            l1_out,
            l2_out,
            num_buckets: weights.num_buckets,
        },
        weights,
        FtOutAlignment::CheckedByCaller,
    )?;

    let (combined_ft_w, combined_ft_b) = shogi_features::router_ftbyft::combine_ft_by_ft_columns(
        &weights.ft_w,
        &weights.ft_b,
        router_w,
        router_b,
        ft_in,
        layout,
    );

    write_u32(writer, YO_VERSION)?;
    write_u32(writer, YO_TOP_HASH)?;
    let arch = Architecture {
        feature_set,
        ft_out: ft_out_nominal,
        l1_out,
        l2_out,
        num_buckets: weights.num_buckets,
    };
    let arch_string = ft_by_ft_arch_string(&arch, bucket_mode);
    write_u32(
        writer,
        u32::try_from(arch_string.len()).expect("architecture string length fits in u32"),
    )?;
    writer.write_all(arch_string.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    write_leb128_tensor_i16(writer, &quantize_i16(&combined_ft_b, QA as f64))?;
    write_leb128_tensor_i16(writer, &quantize_i16(&combined_ft_w, QA as f64))?;

    write_progress_block(writer, bucket_mode)?;

    for bucket in 0..arch.num_buckets {
        write_u32(writer, YO_NETWORK_HASH)?;

        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1f_b[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out_nominal).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out_nominal + output * ft_out_nominal + input]
                    + weights.l1f_w[input * l1_out + output]
            })
        });
        write_affine(writer, l1_biases, l1_weights, ft_out_nominal, l1_out)?;

        let l2_biases = (0..l2_out).map(|output| weights.l2_b[bucket * l2_out + output]);
        let l2_weights = (0..l2_out).flat_map(|output| {
            (0..l2_in)
                .map(move |input| weights.l2_w[bucket * l2_out * l2_in + output * l2_in + input])
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, l2_out)?;

        write_affine(
            writer,
            std::iter::once(weights.l3_b[bucket]),
            (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]),
            l2_out,
            1,
        )?;
    }

    Ok(())
}

/// [`save_yaneuraou_combined`] 用の `arch_string`。`bucket_mode` の
/// `routerft<R>ft<R>` トークン (`canonical_token()`が既に含む) を使う点以外は
/// [`arch_string`] と同じ (`nnue_arch_gen.py` の `routerft<R>ft<R>` 検出ロジック
/// と対応させる)。
fn ft_by_ft_arch_string(arch: &Architecture, bucket_mode: BucketMode) -> String {
    arch_string(arch, bucket_mode)
}

#[derive(Debug)]
struct Architecture {
    feature_set: FeatureSet,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
}

fn architecture(weights: &LayerStackWeights) -> io::Result<Architecture> {
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|feature_set| feature_set.spec() == weights.feature_set)
        .ok_or_else(|| invalid_input_err("feature set is not representable in YaneuraOu SFNN"))?;
    let ft_out = weights.ft_b.len();
    let l1_out = weights.l1f_b.len();
    let num_buckets = weights.num_buckets;
    if num_buckets == 0 || num_buckets > MAX_LAYER_STACKS {
        return invalid_input(format!(
            "unsupported LayerStack count {num_buckets} (expected 1..={MAX_LAYER_STACKS})"
        ));
    }
    let l2_out = weights.l2_b.len().checked_div(num_buckets).unwrap_or(0);
    Ok(Architecture {
        feature_set,
        ft_out,
        l1_out,
        l2_out,
        num_buckets,
    })
}

/// `bucket_mode` の layout を表す edition suffix
/// (`architectures/nnue_arch_gen.py` の `hand.../k.../progress.../router...`
/// トークン列と対応させる)。空 (`BucketMode::NONE`、単一バケット) のときは
/// `bucket_mode.canonical_token()` が `"NONE"` を返す — 呼び出し元
/// (`arch_string`) 側で `arch.num_buckets == 1` の特殊ケースとして扱う。
fn arch_string(arch: &Architecture, bucket_mode: BucketMode) -> String {
    let feature = YO_FEATURES
        .iter()
        .find(|feature| feature.feature_set == arch.feature_set)
        .expect("every FeatureSet has a YaneuraOu mapping");
    let input_size = arch.feature_set.spec().ft_in();
    let h1 = arch.l1_out - 1;
    // 旧来の配布net (kingrank9 / k3k3 単体、halfkahm2 1536-15-32) は
    // "SFNN-1536" という特別短縮名を使う互換性維持。
    let network = if arch.num_buckets == YANEURAOU_LAYER_STACKS
        && bucket_mode
            == (BucketMode {
                king: Some(shogi_features::bucket_mode::KingSubMode::K3K3),
                ..BucketMode::NONE
            })
        && arch.feature_set == FeatureSet::HalfKaHmMerged
        && arch.ft_out == 1536
        && arch.l1_out == 16
        && arch.l2_out == 32
    {
        "SFNN-1536".to_string()
    } else {
        format!(
            "SFNN_{}_{}_{}_{}_{}",
            feature.gen_key,
            arch.ft_out,
            h1,
            arch.l2_out,
            bucket_mode.canonical_token()
        )
        .to_ascii_uppercase()
    };
    format!(
        "ModelType=SFNNWithoutPsqt;Features={}(Friend)[{input_size}->{}x2],Network={network}{{LayerStack={}}}",

        feature.yo_name, arch.ft_out, arch.num_buckets
    )
}

/// [`validate_weights`] の FT 出力次元アラインメントチェックの挙動。
enum FtOutAlignment {
    /// 通常 (`--router-arch kpabs` 含む): `arch.ft_out` 単体が 32 の倍数で
    /// あることを要求する (物理 FT 幅 == `arch.ft_out` なので当然)。
    MustBeMultipleOf32,
    /// `--router-arch ft-by-ft` 用: `arch.ft_out` (nominal) 単体はチェック
    /// しない。物理 FT 幅 (`ft_out + R`) のアラインメントは呼び出し側
    /// (`save_yaneuraou_combined`) が既にチェック済み。
    CheckedByCaller,
}

fn validate_weights(
    arch: &Architecture,
    weights: &LayerStackWeights,
    ft_out_alignment: FtOutAlignment,
) -> io::Result<()> {
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not representable in YaneuraOu SFNN");
    }
    match ft_out_alignment {
        FtOutAlignment::MustBeMultipleOf32 => {
            if arch.ft_out == 0 || arch.ft_out > MAX_FT_OUT || !arch.ft_out.is_multiple_of(32) {
                return invalid_input(format!(
                    "unsupported FT output dimension {} (expected a positive multiple of 32 up to {MAX_FT_OUT})",
                    arch.ft_out
                ));
            }
        }
        FtOutAlignment::CheckedByCaller => {
            // `save_yaneuraou_combined` (`--router-arch ft-by-ft`) 用:
            // `arch.ft_out` (nominal) 単体は 32 の倍数である必要がない
            // (spec.md 11節 -- 整列が必要なのは物理 FT 幅 `ft_out + R` の方
            // で、呼び出し側が既にそちらをチェック済み)。ここでは正の値で
            // 上限内であることだけを確認する。
            if arch.ft_out == 0 || arch.ft_out > MAX_FT_OUT {
                return invalid_input(format!(
                    "unsupported FT output dimension {} (expected a positive value up to {MAX_FT_OUT})",
                    arch.ft_out
                ));
            }
        }
    }
    if arch.l1_out < 2 || arch.l1_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L1 output dimension {} (expected 2..={MAX_HIDDEN_DIM})",
            arch.l1_out
        ));
    }
    if arch.l2_out == 0 || arch.l2_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 output dimension {} (expected 1..={MAX_HIDDEN_DIM})",
            arch.l2_out
        ));
    }
    let l2_in = (arch.l1_out - 1) * 2;
    let spec = arch.feature_set.spec();
    let lengths = [
        ("ft_b", weights.ft_b.len(), arch.ft_out),
        ("ft_w", weights.ft_w.len(), spec.ft_in() * arch.ft_out),
        (
            "l1_b",
            weights.l1_b.len(),
            arch.num_buckets * arch.l1_out,
        ),
        (
            "l1_w",
            weights.l1_w.len(),
            arch.num_buckets * arch.l1_out * arch.ft_out,
        ),
        ("l1f_b", weights.l1f_b.len(), arch.l1_out),
        ("l1f_w", weights.l1f_w.len(), arch.ft_out * arch.l1_out),
        (
            "l2_b",
            weights.l2_b.len(),
            arch.num_buckets * arch.l2_out,
        ),
        ("l3_b", weights.l3_b.len(), arch.num_buckets),
        (
            "l2_w",
            weights.l2_w.len(),
            arch.num_buckets * arch.l2_out * l2_in,
        ),
        (
            "l3_w",
            weights.l3_w.len(),
            arch.num_buckets * arch.l2_out,
        ),
    ];
    for (name, actual, expected) in lengths {
        if actual != expected {
            return invalid_input(format!(
                "{name} length mismatch: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

fn write_affine<W, B, V>(
    writer: &mut W,
    biases: B,
    weights: V,
    input_dimensions: usize,
    output_dimensions: usize,
) -> io::Result<()>
where
    W: Write,
    B: IntoIterator<Item = f32>,
    V: IntoIterator<Item = f32>,
{
    for bias in biases {
        writer.write_all(&quantize_i32(bias, (QA * QB) as f64).to_le_bytes())?;
    }
    let padded_input = input_dimensions.div_ceil(32) * 32;
    let mut weights = weights.into_iter();
    for _ in 0..output_dimensions {
        for input in 0..padded_input {
            let value = if input < input_dimensions {
                weights
                    .next()
                    .ok_or_else(|| invalid_input_err("affine weight iterator is short"))?
            } else {
                0.0
            };
            writer.write_all(&[quantize_i8(value, QB as f64) as u8])?;
        }
    }
    if weights.next().is_some() {
        return invalid_input("affine weight iterator has extra values");
    }
    Ok(())
}

fn quantize_i16(values: &[f32], scale: f64) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            (value as f64 * scale)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

fn quantize_i32(value: f32, scale: f64) -> i32 {
    (value as f64 * scale)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn quantize_i8(value: f32, scale: f64) -> i8 {
    (value as f64 * scale)
        .round()
        .clamp(i8::MIN as f64, i8::MAX as f64) as i8
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_input_err(message))
}

fn invalid_input_err(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_architecture_string_matches_yaneuraou() {
        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            1536,
            16,
            32,
            YANEURAOU_LAYER_STACKS,
        );
        assert_eq!(
            arch_string(&architecture(&weights).unwrap()),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}"
        );
    }

    #[test]
    fn generated_architecture_names_match_yaneuraou_loader_contract() {
        let cases = [
            (
                FeatureSet::HalfKaHmMerged,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaHmMerged,
                512,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_HALFKAHM2_512_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKp,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->1536x2],Network=SFNN_HALFKP_1536_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaSplit,
                768,
                8,
                16,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA1(Friend)[138510->768x2],Network=SFNN_HALFKA1_768_7_16_K3K3{LayerStack=9}",
            ),
        ];

        for (feature_set, ft_out, l1_out, l2_out, expected) in cases {
            let weights = LayerStackWeights::zeroed(
                feature_set.spec(),
                ft_out,
                l1_out,
                l2_out,
                YANEURAOU_LAYER_STACKS,
            );
            assert_eq!(arch_string(&architecture(&weights).unwrap()), expected);
        }
    }

    #[test]
    fn rejects_zero_buckets() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 1);
        let mut weights = weights;
        weights.num_buckets = 0;
        let error = save_yaneuraou(&mut Vec::new(), &weights, None).unwrap_err();
        assert!(error.to_string().contains("LayerStack count"), "{error}");
    }

    /// `--bucket-mode router --num-buckets N` (N != 9) は KingRank9 以外の
    /// LayerStack 数を持つ。engine 側は `LayerStacks=N` でビルドした router
    /// edition (`YANEURAOU_ENGINE_SFNN_..._routerN` 等) でこれを読む。
    #[test]
    fn non_kingrank9_bucket_count_is_accepted_with_ls_suffix() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 8);
        assert_eq!(
            arch_string(&architecture(&weights).unwrap()),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->128x2],Network=SFNN_HALFKAHM2_128_15_32_LS8{LayerStack=8}"
        );

        let mut out = Vec::new();
        save_yaneuraou(&mut out, &weights, None).expect("save_yaneuraou accepts 8 buckets");
        // 8 network block を書いた分、9-bucket export よりファイルは短い。
        let mut nine_bucket_out = Vec::new();
        let nine_bucket =
            LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 9);
        save_yaneuraou(&mut nine_bucket_out, &nine_bucket, None).unwrap();
        assert!(out.len() < nine_bucket_out.len());
    }

    /// 1536/16/32 の標準構成でも `LayerStack=9` の `SFNN-1536` 特別扱い名は
    /// KingRank9 (N=9) 限定で、router 等の他 bucket 数では汎用の `_ls<N>` 命名に
    /// フォールバックする (`SFNN-1536` という名前自体は bucket 数を含まないため
    /// 既存 9-bucket 配布 net と紛らわしくなるのを避ける)。
    #[test]
    fn standard_1536_dims_with_non_9_buckets_do_not_use_bare_sfnn_1536_name() {
        let weights =
            LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 1536, 16, 32, 16);
        let arch_str = arch_string(&architecture(&weights).unwrap());
        assert!(!arch_str.contains("Network=SFNN-1536{"), "{arch_str}");
        assert_eq!(
            arch_str,
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN_HALFKAHM2_1536_15_32_LS16{LayerStack=16}"
        );
    }

    /// router 重み ([`RouterKPAbsWeights`]) を伴う export は bucket 数に依らず
    /// trailing block を書ける (9 固定だった頃の制約が bucket 数一般化後も
    /// 崩れていないことの回帰確認)。
    #[test]
    fn router_block_is_appended_for_non_9_bucket_export() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 16);
        let router = RouterKPAbsWeights::zeroed(16);
        let mut with_router = Vec::new();
        save_yaneuraou(&mut with_router, &weights, Some(&router)).unwrap();
        let mut without_router = Vec::new();
        save_yaneuraou(&mut without_router, &weights, None).unwrap();
        assert!(with_router.len() > without_router.len());
    }

    #[test]
    fn direct_export_matches_tatara_reload_export_byte_for_byte() {
        let mut weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        );
        weights.ft_b[0] = 0.25;
        weights.ft_w[17] = -0.5;
        weights.l1_b[2] = 0.75;
        weights.l1_w[31] = -0.25;
        weights.l1f_b[1] = 0.25;
        weights.l1f_w[7] = -0.5;
        weights.l2_b[4] = 0.5;
        weights.l2_w[9] = -0.75;
        weights.l3_b[3] = 0.125;
        weights.l3_w[5] = -0.125;

        let mut direct = Vec::new();
        save_yaneuraou(&mut direct, &weights, None).unwrap();

        let mut tatara = Vec::new();
        weights.save_quantised(&mut tatara, Some(28)).unwrap();
        let reloaded = LayerStackWeights::load_quantised(
            &mut tatara.as_slice(),
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        )
        .unwrap();
        let mut post_hoc = Vec::new();
        save_yaneuraou(&mut post_hoc, &reloaded, None).unwrap();

        assert_eq!(direct, post_hoc);
    }

    #[test]
    fn affine_weights_are_row_major_and_padded() {
        let mut output = Vec::new();
        write_affine(
            &mut output,
            [1.0, -1.0],
            [
                1.0 / 64.0,
                2.0 / 64.0,
                3.0 / 64.0,
                -1.0 / 64.0,
                -2.0 / 64.0,
                -3.0 / 64.0,
            ],
            3,
            2,
        )
        .unwrap();
        assert_eq!(
            i32::from_le_bytes(output[0..4].try_into().unwrap()),
            QA * QB
        );
        assert_eq!(
            i32::from_le_bytes(output[4..8].try_into().unwrap()),
            -(QA * QB)
        );
        assert_eq!(&output[8..11], &[1, 2, 3]);
        assert!(output[11..40].iter().all(|&byte| byte == 0));
        assert_eq!(&output[40..43], &[255, 254, 253]);
        assert!(output[43..72].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn save_yaneuraou_combined_produces_router_ft_arch_string_and_widened_ft_block() {
        use shogi_features::router_ftbyft::FtByFtLayout;

        // R=2, ft_out(nominal)=32 (multiple of 32, minimal valid case).
        let ft_out = 32;
        let num_buckets = 4; // = R*R, R=2
        let layout = FtByFtLayout::new(num_buckets, ft_out).unwrap();
        assert_eq!(layout.r, 2);
        assert_eq!(layout.accum_out, 34);

        let feature_set = FeatureSet::HalfKaHmMerged;
        let weights =
            LayerStackWeights::zeroed(feature_set.spec(), ft_out, 16, 32, num_buckets);
        let ft_in = feature_set.spec().ft_in();
        let router_w = vec![0.0f32; ft_in * layout.r];
        let router_b = vec![0.0f32; layout.r];

        let mut out = Vec::new();
        save_yaneuraou_combined(&mut out, &weights, &layout, &router_w, &router_b).unwrap();

        // header: version, top_hash, arch_string_len, arch_string, ft_hash, ...
        let arch_len = u32::from_le_bytes(out[8..12].try_into().unwrap()) as usize;
        let arch_string = std::str::from_utf8(&out[12..12 + arch_len]).unwrap();
        assert!(
            arch_string.contains("ROUTER_FT2FT2"),
            "arch_string should use the ft-by-ft edition suffix: {arch_string}"
        );
        // nominal ft_out (32), not the physical combined width (34), must
        // appear in the arch string (L1-facing dimension).
        assert!(arch_string.contains("[73305->32x2]"), "{arch_string}");

        // ft_hash marker follows arch_string, then LEB128 ft_b (34 entries)
        // and ft_w (ft_in * 34 entries) -- just check it doesn't panic and
        // produced a non-trivial byte stream longer than the un-combined
        // baseline would need (sanity, not an exact byte match).
        assert!(out.len() > 12 + arch_len + 4);
    }

    #[test]
    fn save_yaneuraou_combined_rejects_bucket_count_mismatch() {
        use shogi_features::router_ftbyft::FtByFtLayout;

        let ft_out = 32;
        let layout = FtByFtLayout::new(4, ft_out).unwrap(); // R=2, num_buckets=4
        let feature_set = FeatureSet::HalfKaHmMerged;
        // weights built with a DIFFERENT num_buckets (9) than layout (4).
        let weights = LayerStackWeights::zeroed(feature_set.spec(), ft_out, 16, 32, 9);
        let ft_in = feature_set.spec().ft_in();
        let router_w = vec![0.0f32; ft_in * layout.r];
        let router_b = vec![0.0f32; layout.r];

        let mut out = Vec::new();
        let err = save_yaneuraou_combined(&mut out, &weights, &layout, &router_w, &router_b)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
