//! `layerstack_v3` — bucketごとに **L1/L2 の出力次元を個別指定できる** LayerStack
//! architecture の quantised `.bin` save / load。
//!
//! ## `layerstack` (V2 / [`crate::layerstack_weights`]) との違い
//!
//! - `ft_out` は本モジュールも共通 (bucket 間で共有)。
//! - `l1_out` / `l2_out` は **bucketごとに個別サイズ** を持てる (`[usize; 9]`)。
//!   `layerstack_weights` は全bucketが同一の `l1_out`/`l2_out` を共有する前提の
//!   flat 配列 (`Vec<f32>` を `num_buckets * dim` で割った固定 stride) だったが、
//!   本モジュールは bucketごとに長さが異なる `[Vec<f32>; 9]` (ragged array) を使う。
//! - bucket数は **常に9固定** ([`NUM_BUCKETS_V3`])。progress8kpabs (0..7 の8bucket) /
//!   progress9kpabs (0..8 の9bucket) / kingrank9 のどのbucket割り当てを使うかは
//!   engine 側の実行時オプション (yaneuraou の `LS_BUCKET_MODE`) で選ぶので、本
//!   format / architecture 名には bucket割り当て方式のsuffixを含めない
//!   (`_k3k3` 等は付けない)。
//! - L1f (shared factorized L1 の学習時トリック、[`crate::layerstack_weights`] 参照)
//!   は持たない。bucketごとにサイズが異なるため素直な shared factorizer が
//!   組みにくく、本 v1 実装ではスコープ外にした (per-bucket L1 を直接学習する)。
//!   PSQT shortcut / Threat feature も同様にスコープ外 (将来拡張の余地として
//!   残すが、現時点では未対応 = save/load 側で reject する)。
//!
//! ## file layout (top-level)
//!
//! **この layout は yaneuraou の generic loader
//! (`evaluate_nnue.cpp::LoadAndShare` / `ReadHeader` /
//! `Detail::ReadParameters<T>`) が実際に読む byte 列と 1:1 で対応するように
//! 書く。ここに書いていないフィールドを追加したり、書いてある通りの順序・個数
//! から外れたりすると、hard error にはならず (yaneuraou 側の hash/version
//! mismatch はいずれも warning のみ) **読み込みだけ成功してゴミ weight で
//! 動く** という、気づきにくい壊れ方をする (実際にこの bug で発生した:
//! `--ft-out`/`--l1-per-bucket`/`--l2-per-bucket` の値を header に埋めて
//! 自己記述的にしようとして余計な bytes を挟んでいた版があった。dims は
//! (yaneuraou 側が `nnue_arch_gen.py` でコンパイル時に決め打ちするのと同様)
//! **完全に呼び出し側の `expected_*` 引数から来るもので、file には一切
//! 含めない**。)**
//!
//! 1. header ([`crate::layerstack_weights`] と共通の `ReadHeader`/`WriteHeader`
//!    contract): version (4 LE u32, [`NNUE_V3_VERSION`]) + hash_value
//!    (4 LE u32、yaneuraou 側は info string warning のみに使う cosmetic な
//!    値、値そのものの一致は要求されない) + arch_len (4 LE u32) + arch_str
//!    (人間可読、[`build_arch_str_v3`] 参照。load 時の構造化照合は
//!    feature set 名の prefix のみ、他は呼び出し側の `expected_*` 引数で
//!    行う。**ここに `num_buckets`/`ft_out`/`l1_out`/`l2_out` 等の追加
//!    フィールドを絶対に挟まないこと** — yaneuraou 側はこの直後
//!    ノーマージンで FT の hash を読みにいく)。
//! 2. `Detail::ReadParameters<FeatureTransformer>` wrapper: ft_hash
//!    (4 LE u32、cosmetic) + ft_biases LEB128 + ft_weights LEB128
//!    (`base_ft_in * ft_out` 個、`crate::layerstack_weights` と同じ
//!    magic/形式)。
//! 3. `Detail::ReadParameters<Network>` wrapper (V3 は 9 bucket 分を
//!    `Network` 1個に集約しているのでこの wrapper は **1回だけ**):
//!    network_hash (4 LE u32、cosmetic) に続けて、9 bucket 分を
//!    ノーマージンで並べる。**bucket ごとの hash は無い**
//!    (`NetworkBucket::ReadParameters` は hash を読まない、
//!    `nnue_arch_gen.py` 生成コード参照) — bucket ごとに hash を書くと
//!    4 bytes × 9 だけ file がずれ、以降すべてのデータがゴミになる。
//!    各 bucket `i` の内容:
//!    `L1 (bias i32×l1_out[i] + weight i8×l1_out[i]×pad32(ft_out)),
//!     L2 (bias i32×l2_out[i] + weight i8×l2_out[i]×pad32(l2_in[i])),
//!     L3 (bias i32×1 + weight i8×pad32(l2_out[i]))`
//!    (`l2_in[i] = (l1_out[i] - 1) * 2`)。
//!
//! 量子化スケールは [`crate::layerstack_weights`] と同一 (`QA` = FT scale, `QB` =
//! dense weight scale, bias は `QA*QB` または `127*QB`)。`pad32` も同じ (32 の
//! 倍数 pad、SIMD load align 要求)。dense 層の weight の file 上のバイト順は
//! yaneuraou 側の SIMD 用 scramble 済み内部レイアウトとは別物の、素直な
//! row-major (`weight[out][in]`) で良い — `AffineTransform{,SparseInput}Explicit
//! ::ReadParameters` が file 上は row-major な列を読みながら内部の scramble
//! 済み位置に書き込む (`get_weight_index`) ので、file 側が scramble を意識
//! する必要は無い。
//!
//! この byte layout は yaneuraou の `nnue_arch_gen.py --l1 --l2` が生成する
//! `SFNNwoP_V3` architecture header (`NetworkBucket<L1,L2,Hash>` を9個集約した
//! `Network` 構造体) が読む1bucket分の layout (`fc_0`=L1 sparse affine,
//! `ac_0`+`ac_sqr_0`=活性化 (パラメータ無し), `fc_1`=L2 affine, `ac_1`=活性化,
//! `fc_2`=L3 affine) と一致するように書く。

use std::io::{self, Read, Write};

use shogi_features::FeatureSetSpec;

use crate::layerstack_weights::{
    compute_fc_hash, features_token, ft_hash, pad32, LayerStackWeights, QA, QB,
};

/// layerstack_v3 の bucket 数。常に固定 (bucket 割り当て方式は engine 側の実行時
/// オプションで選ぶため、architecture 側は bucket 数のみ知っていればよい)。
pub const NUM_BUCKETS_V3: usize = 9;

/// 現行 `.bin` format version (layerstack_v3 専用の magic)。
pub const NNUE_V3_VERSION: u32 = 0x7AF3_3301;

/// 既定 FT 出力次元 (`--ft-out` 未指定時)。[`crate::layerstack_weights::DEFAULT_FT_OUT`]
/// と同じ値 (bucket 間で共通の軸なので既定値も揃えておく)。
pub const DEFAULT_FT_OUT: usize = 1536;
/// 既定 L1 出力次元 (bucketごとに同一値を敷き詰めた既定配列)。
pub const DEFAULT_L1_OUT: usize = 16;
/// 既定 L2 出力次元 (同上)。
pub const DEFAULT_L2_OUT: usize = 32;

/// カンマ区切り文字列 (9個の自然数) を `[usize; NUM_BUCKETS_V3]` に parse する。
/// `--l1` / `--l2` CLI オプション共通の parser。
pub fn parse_bucket_dims_csv(csv: &str, opt_name: &str) -> Result<[usize; NUM_BUCKETS_V3], String> {
    let parts: Vec<&str> = csv.split(',').map(str::trim).collect();
    if parts.len() != NUM_BUCKETS_V3 {
        return Err(format!(
            "{opt_name} must be {NUM_BUCKETS_V3} comma-separated natural numbers, got {} in `{csv}`",
            parts.len()
        ));
    }
    let mut out = [0usize; NUM_BUCKETS_V3];
    for (i, p) in parts.iter().enumerate() {
        let v: usize = p
            .parse()
            .map_err(|_| format!("{opt_name} entry {i} is not a natural number: `{p}`"))?;
        if v == 0 {
            return Err(format!("{opt_name} entry {i} must be > 0, got 0"));
        }
        out[i] = v;
    }
    Ok(out)
}

/// layerstack_v3 の arch 文字列 (人間可読、debug / ログ用)。load 時の構造化照合は
/// 別途 explicit header field (ft_out / l1_out\[9\] / l2_out\[9\]) で行うため、本
/// 文字列自体は feature set 名の prefix 照合にのみ使う (`features_token` と同じ
/// 規約)。
pub fn build_arch_str_v3(
    feature_name: &str,
    input_size: usize,
    ft_out: usize,
    l1_out: &[usize; NUM_BUCKETS_V3],
    l2_out: &[usize; NUM_BUCKETS_V3],
    fv_scale: i32,
) -> String {
    let l1_list = l1_out
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let l2_list = l2_out
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{},ModelType=SFNNwoP_V3,L1=[{l1_list}],L2=[{l2_list}],fv_scale={fv_scale}",
        features_token(feature_name, input_size, ft_out),
    )
}

/// layerstack_v3 の network_hash。`ft_hash` と、bucketごとの `compute_fc_hash`
/// (bucket index で `rotate_left` して同一サイズ列でも順序を区別できるようにした
/// もの) を XOR-fold した値。engine 側 (yaneuraou) はこの hash を warning-only の
/// 整合性チェックにしか使わない (mismatch でも load は継続する) が、`.bin` 側の
/// reject policy (本 crate の `load_quantised`) は hard reject に使う。
pub fn network_hash_v3(feature_hash: u32, ft_out: usize, l2_out: &[usize; NUM_BUCKETS_V3]) -> u32 {
    let mut h = ft_hash(feature_hash, ft_out);
    for (i, &l2) in l2_out.iter().enumerate() {
        h ^= compute_fc_hash(ft_out, l2).rotate_left(i as u32 + 1);
    }
    h
}

/// layerstack_v3 の全 weight (f32, host 側保持)。bucketごとに `l1_out[i]` /
/// `l2_out[i]` が異なるため、per-bucket テンソルは `[Vec<f32>; NUM_BUCKETS_V3]`
/// (ragged array) で持つ。
///
/// layout (bucket `i` 内は [`crate::layerstack_weights::LayerStackWeights`] の
/// 対応テンソルと同じ row-major 規約):
/// - `ft_w`: `(ft_in, ft_out)` row-major、`ft_w[feat * ft_out + out]`
/// - `ft_b`: `(ft_out)`
/// - `l1_w[i]`: `(l1_out[i], ft_out)` row-major
/// - `l1_b[i]`: `(l1_out[i])`
/// - `l2_w[i]`: `(l2_out[i], l2_in[i])` row-major、`l2_in[i] = (l1_out[i]-1)*2`
/// - `l2_b[i]`: `(l2_out[i])`
/// - `l3_w[i]`: `(l2_out[i])` (out_dim=1 なので out 軸省略)
/// - `l3_b[i]`: scalar
#[derive(Debug, Clone)]
pub struct LayerStackV3Weights {
    pub feature_set: FeatureSetSpec,
    pub ft_out: usize,
    pub l1_out: [usize; NUM_BUCKETS_V3],
    pub l2_out: [usize; NUM_BUCKETS_V3],
    pub ft_w: Vec<f32>,
    pub ft_b: Vec<f32>,
    pub l1_w: [Vec<f32>; NUM_BUCKETS_V3],
    pub l1_b: [Vec<f32>; NUM_BUCKETS_V3],
    pub l2_w: [Vec<f32>; NUM_BUCKETS_V3],
    pub l2_b: [Vec<f32>; NUM_BUCKETS_V3],
    pub l3_w: [Vec<f32>; NUM_BUCKETS_V3],
    pub l3_b: [f32; NUM_BUCKETS_V3],
}

/// bucket `i` の L2 入力次元 (`(l1_out[i] - 1) * 2`)。
#[inline]
pub fn l2_in_for(l1_out_i: usize) -> usize {
    (l1_out_i - 1) * 2
}

impl LayerStackV3Weights {
    /// 全 buffer を 0 で初期化した新規 instance。
    pub fn zeroed(
        feature_set: FeatureSetSpec,
        ft_out: usize,
        l1_out: [usize; NUM_BUCKETS_V3],
        l2_out: [usize; NUM_BUCKETS_V3],
    ) -> Self {
        for i in 0..NUM_BUCKETS_V3 {
            assert!(
                l1_out[i] >= 2,
                "l1_out[{i}] must be >= 2 (1 skip dim + >=1 main dim), got {}",
                l1_out[i]
            );
            assert!(l2_out[i] >= 1, "l2_out[{i}] must be >= 1, got {}", l2_out[i]);
        }
        Self {
            feature_set,
            ft_out,
            l1_out,
            l2_out,
            ft_w: vec![0.0; feature_set.ft_in() * ft_out],
            ft_b: vec![0.0; ft_out],
            l1_w: std::array::from_fn(|i| vec![0.0; l1_out[i] * ft_out]),
            l1_b: std::array::from_fn(|i| vec![0.0; l1_out[i]]),
            l2_w: std::array::from_fn(|i| vec![0.0; l2_out[i] * l2_in_for(l1_out[i])]),
            l2_b: std::array::from_fn(|i| vec![0.0; l2_out[i]]),
            l3_w: std::array::from_fn(|i| vec![0.0; l2_out[i]]),
            l3_b: [0.0; NUM_BUCKETS_V3],
        }
    }

    /// layerstack_v3 quantised `.bin` を `writer` に書き出す。
    pub fn save_quantised<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        if self.feature_set.threat_profile().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "layerstack_v3 does not support threat features yet",
            ));
        }
        let ft_out = self.ft_out;
        if self.ft_b.len() != ft_out {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ft_b length {} != ft_out {ft_out}", self.ft_b.len()),
            ));
        }
        let feature_hash = self.feature_set.feature_hash();

        writer.write_all(&NNUE_V3_VERSION.to_le_bytes())?;
        writer.write_all(&network_hash_v3(feature_hash, ft_out, &self.l2_out).to_le_bytes())?;

        let arch_str = build_arch_str_v3(
            self.feature_set.arch_feature_name(),
            self.feature_set.ft_in(),
            ft_out,
            &self.l1_out,
            &self.l2_out,
            crate::layerstack_weights::FV_SCALE,
        );
        let arch_bytes = arch_str.as_bytes();
        writer.write_all(&(arch_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(arch_bytes)?;

        // NOTE: ここで num_buckets/ft_out/l1_out[9]/l2_out[9] 等を書いては
        // いけない。yaneuraou 側 (`evaluate_nnue.cpp::LoadAndShare`) は
        // `ReadHeader` (version + hash + arch_len + arch_str) の直後、
        // 一切追加フィールドを挟まずに
        // `Detail::ReadParameters<FeatureTransformer>` (= ft_hash (4 LE u32)
        // + FT の LEB128 bias/weight) を読みにいく。ここで余計な bytes を
        // 挟むと、そのぶん FT の LEB128 ストリームの先頭がずれて全体が
        // 破壊される (かつ hard error にはならず、ゴミ weight のまま
        // "info string Warning: NNUE file has trailing data (ignored)" と
        // 共に読み込みだけは成功してしまうので発見しづらい)。
        writer.write_all(&ft_hash(feature_hash, ft_out).to_le_bytes())?;

        // ---- FT biases / weights (LEB128 i16, scale=QA) ----
        let qa_f = QA as f64;
        let ft_b_i16: Vec<i16> = self
            .ft_b
            .iter()
            .map(|&v| quantise_i16(v, qa_f))
            .collect();
        crate::layerstack_weights::write_leb128_tensor_i16(writer, &ft_b_i16)?;

        let base_ft_w_n = self.feature_set.base_ft_in() * ft_out;
        if self.ft_w.len() != self.feature_set.ft_in() * ft_out {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ft_w length {} != ft_in {} * ft_out {ft_out}",
                    self.ft_w.len(),
                    self.feature_set.ft_in(),
                ),
            ));
        }
        let ft_w_base = &self.ft_w[..base_ft_w_n];
        let ft_w_i16: Vec<i16> = ft_w_base.iter().map(|&v| quantise_i16(v, qa_f)).collect();
        crate::layerstack_weights::write_leb128_tensor_i16(writer, &ft_w_i16)?;

        // ---- LayerStacks (9 buckets、per-bucket サイズ) ----
        // ここで書く1個の hash が yaneuraou 側 `Detail::ReadParameters<Network>`
        // wrapper が読む hash (= `Network::GetHashValue()` との比較、mismatch でも
        // warning のみ) に対応する。V2 の per-bucket と違い、V3 は9bucket分を
        // Network 1個に集約しているので、bucket ごとの hash は無い (bucket 単体の
        // `NetworkBucket::ReadParameters` は hash を読まない、
        // yaneuraou 側 `nnue_arch_gen.py` 生成コード参照)。
        writer.write_all(&network_hash_v3(feature_hash, ft_out, &self.l2_out).to_le_bytes())?;
        let qb_f = QB as f64;
        let l1_bias_scale = (QA * QB) as f64; // = 8128
        let l2_bias_scale = 127.0 * qb_f; // = 8128 (QA == 127 前提)
        let l3_bias_scale = 127.0 * qb_f;

        for buc in 0..NUM_BUCKETS_V3 {
            let l1_out = self.l1_out[buc];
            let l2_out = self.l2_out[buc];
            let l2_in = l2_in_for(l1_out);
            check_len(&self.l1_b[buc], l1_out, "l1_b", buc)?;
            check_len(&self.l1_w[buc], l1_out * ft_out, "l1_w", buc)?;
            check_len(&self.l2_b[buc], l2_out, "l2_b", buc)?;
            check_len(&self.l2_w[buc], l2_out * l2_in, "l2_w", buc)?;
            check_len(&self.l3_w[buc], l2_out, "l3_w", buc)?;

            // --- L1 ---
            for &b in &self.l1_b[buc] {
                let val = (l1_bias_scale * b as f64).round() as i32;
                writer.write_all(&val.to_le_bytes())?;
            }
            let l1_padded_in = pad32(ft_out);
            for out in 0..l1_out {
                for in_idx in 0..l1_padded_in {
                    let q: i8 = if in_idx < ft_out {
                        let w = self.l1_w[buc][out * ft_out + in_idx];
                        clamp_i8((qb_f * w as f64).round())
                    } else {
                        0
                    };
                    writer.write_all(&[q as u8])?;
                }
            }

            // --- L2 ---
            for &b in &self.l2_b[buc] {
                let val = (l2_bias_scale * b as f64).round() as i32;
                writer.write_all(&val.to_le_bytes())?;
            }
            let l2_padded_in = pad32(l2_in);
            for out in 0..l2_out {
                for in_idx in 0..l2_padded_in {
                    let q: i8 = if in_idx < l2_in {
                        let w = self.l2_w[buc][out * l2_in + in_idx];
                        clamp_i8((qb_f * w as f64).round())
                    } else {
                        0
                    };
                    writer.write_all(&[q as u8])?;
                }
            }

            // --- L3 (output, 1 dim) ---
            let val = (l3_bias_scale * self.l3_b[buc] as f64).round() as i32;
            writer.write_all(&val.to_le_bytes())?;
            let l3_padded_in = pad32(l2_out);
            for in_idx in 0..l3_padded_in {
                let q: i8 = if in_idx < l2_out {
                    clamp_i8((qb_f * self.l3_w[buc][in_idx] as f64).round())
                } else {
                    0
                };
                writer.write_all(&[q as u8])?;
            }
        }

        Ok(())
    }

    /// layerstack_v3 quantised `.bin` を parse する。`expected_l1_out` /
    /// `expected_l2_out` は要求する bucketごとの出力次元。file 側の値と完全一致
    /// しなければ `InvalidData` で reject する。
    pub fn load_quantised<R: Read>(
        reader: &mut R,
        expected: FeatureSetSpec,
        expected_ft_out: usize,
        expected_l1_out: [usize; NUM_BUCKETS_V3],
        expected_l2_out: [usize; NUM_BUCKETS_V3],
    ) -> io::Result<Self> {
        if expected.threat_profile().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "layerstack_v3 does not support threat features yet",
            ));
        }
        let mut buf4 = [0u8; 4];

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != NNUE_V3_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown layerstack_v3 version: {version:#x} (expected {NNUE_V3_VERSION:#x})"),
            ));
        }

        reader.read_exact(&mut buf4)?;
        let file_network_hash = u32::from_le_bytes(buf4);
        let expected_network_hash =
            network_hash_v3(expected.feature_hash(), expected_ft_out, &expected_l2_out);
        if file_network_hash != expected_network_hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "network_hash mismatch: file {file_network_hash:#x}, expected \
                     {expected_network_hash:#x} for feature set {}",
                    expected.canonical_name()
                ),
            ));
        }

        reader.read_exact(&mut buf4)?;
        let arch_len = u32::from_le_bytes(buf4) as usize;
        if arch_len == 0 || arch_len > 16384 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid arch_len: {arch_len}"),
            ));
        }
        let mut arch_bytes = vec![0u8; arch_len];
        reader.read_exact(&mut arch_bytes)?;
        let arch_str = String::from_utf8_lossy(&arch_bytes);
        let expected_features_prefix =
            features_token(expected.arch_feature_name(), expected.ft_in(), expected_ft_out);
        if !arch_str.starts_with(&expected_features_prefix) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "feature set mismatch: expected `{expected_features_prefix}` (feature set {}), \
                     file arch_str = `{arch_str}`",
                    expected.canonical_name()
                ),
            ));
        }

        reader.read_exact(&mut buf4)?;
        let file_ft_hash = u32::from_le_bytes(buf4);
        let expected_ft_hash = ft_hash(expected.feature_hash(), expected_ft_out);
        if file_ft_hash != expected_ft_hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ft_hash mismatch: file {file_ft_hash:#x}, expected {expected_ft_hash:#x} \
                     for feature set {}",
                    expected.canonical_name()
                ),
            ));
        }

        let ft_out = expected_ft_out;
        let qa_f = QA as f32;
        let ft_b_i16 =
            crate::layerstack_weights::read_leb128_tensor_i16(reader, Some(ft_out))?;
        let ft_b: Vec<f32> = ft_b_i16.iter().map(|&v| v as f32 / qa_f).collect();

        let base_ft_w_n = expected.base_ft_in() * ft_out;
        let ft_w_i16 =
            crate::layerstack_weights::read_leb128_tensor_i16(reader, Some(base_ft_w_n))?;
        let ft_w: Vec<f32> = ft_w_i16.iter().map(|&v| v as f32 / qa_f).collect();

        let qb_f = QB as f32;
        let l1_bias_scale = (QA * QB) as f32;
        let l2_bias_scale = 127.0 * qb_f;
        let l3_bias_scale = 127.0 * qb_f;

        let mut l1_w: [Vec<f32>; NUM_BUCKETS_V3] = std::array::from_fn(|_| Vec::new());
        let mut l1_b: [Vec<f32>; NUM_BUCKETS_V3] = std::array::from_fn(|_| Vec::new());
        let mut l2_w: [Vec<f32>; NUM_BUCKETS_V3] = std::array::from_fn(|_| Vec::new());
        let mut l2_b: [Vec<f32>; NUM_BUCKETS_V3] = std::array::from_fn(|_| Vec::new());
        let mut l3_w: [Vec<f32>; NUM_BUCKETS_V3] = std::array::from_fn(|_| Vec::new());
        let mut l3_b = [0.0_f32; NUM_BUCKETS_V3];

        // yaneuraou 側 `Detail::ReadParameters<Network>` wrapper が読む hash
        // (save_quantised の対応するコメント参照)。bucket 単体の
        // `NetworkBucket::ReadParameters` は hash を読まないので、ここで
        // 1回だけ読む (bucket ごとには読まない)。
        reader.read_exact(&mut buf4)?;
        let file_network_level_hash = u32::from_le_bytes(buf4);
        let expected_network_level_hash =
            network_hash_v3(expected.feature_hash(), expected_ft_out, &expected_l2_out);
        if file_network_level_hash != expected_network_level_hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "network hash (Detail::ReadParameters<Network> wrapper) mismatch: file \
                     {file_network_level_hash:#x}, expected {expected_network_level_hash:#x}"
                ),
            ));
        }

        for buc in 0..NUM_BUCKETS_V3 {
            let l1_out = expected_l1_out[buc];
            let l2_out = expected_l2_out[buc];
            let l2_in = l2_in_for(l1_out);

            let mut bucket_l1_b = vec![0.0_f32; l1_out];
            for out in 0..l1_out {
                reader.read_exact(&mut buf4)?;
                bucket_l1_b[out] = i32::from_le_bytes(buf4) as f32 / l1_bias_scale;
            }
            let l1_padded_in = pad32(ft_out);
            let mut bucket_l1_w = vec![0.0_f32; l1_out * ft_out];
            for out in 0..l1_out {
                for in_idx in 0..l1_padded_in {
                    let mut byte = [0u8; 1];
                    reader.read_exact(&mut byte)?;
                    if in_idx < ft_out {
                        bucket_l1_w[out * ft_out + in_idx] = byte[0] as i8 as f32 / qb_f;
                    }
                }
            }

            let mut bucket_l2_b = vec![0.0_f32; l2_out];
            for out in 0..l2_out {
                reader.read_exact(&mut buf4)?;
                bucket_l2_b[out] = i32::from_le_bytes(buf4) as f32 / l2_bias_scale;
            }
            let l2_padded_in = pad32(l2_in);
            let mut bucket_l2_w = vec![0.0_f32; l2_out * l2_in];
            for out in 0..l2_out {
                for in_idx in 0..l2_padded_in {
                    let mut byte = [0u8; 1];
                    reader.read_exact(&mut byte)?;
                    if in_idx < l2_in {
                        bucket_l2_w[out * l2_in + in_idx] = byte[0] as i8 as f32 / qb_f;
                    }
                }
            }

            reader.read_exact(&mut buf4)?;
            l3_b[buc] = i32::from_le_bytes(buf4) as f32 / l3_bias_scale;
            let l3_padded_in = pad32(l2_out);
            let mut bucket_l3_w = vec![0.0_f32; l2_out];
            for in_idx in 0..l3_padded_in {
                let mut byte = [0u8; 1];
                reader.read_exact(&mut byte)?;
                if in_idx < l2_out {
                    bucket_l3_w[in_idx] = byte[0] as i8 as f32 / qb_f;
                }
            }

            l1_b[buc] = bucket_l1_b;
            l1_w[buc] = bucket_l1_w;
            l2_b[buc] = bucket_l2_b;
            l2_w[buc] = bucket_l2_w;
            l3_w[buc] = bucket_l3_w;
        }

        Ok(Self {
            feature_set: expected,
            ft_out,
            l1_out: expected_l1_out,
            l2_out: expected_l2_out,
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
        })
    }
}

fn quantise_i16(v: f32, scale: f64) -> i16 {
    (scale * v as f64)
        .round()
        .clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

fn clamp_i8(v: f64) -> i8 {
    if v < -128.0 {
        -128
    } else if v > 127.0 {
        127
    } else {
        v as i8
    }
}

fn check_len(v: &[f32], expected: usize, name: &str, bucket: usize) -> io::Result<()> {
    if v.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{name}[{bucket}] length {} != expected {expected}",
                v.len()
            ),
        ));
    }
    Ok(())
}

/// `layerstack` (V2, 全 bucket 同一サイズ) の [`LayerStackWeights`] を、
/// bucket ごとに小さい実効サイズを持つ [`LayerStackV3Weights`] へ変換する。
///
/// # 前提 ("bucket dim freeze" 学習方式)
///
/// `uniform` は「`l1_out`/`l2_out` を `l1_out_per_bucket`/`l2_out_per_bucket`
/// の **最大値** で学習し、bucket ごとに `[real_size, max_size)` の余剰行/列
/// (skip 行を除く) を 0 に固定した」ネットワークであることを前提とする
/// (`GpuTrainer::apply_bucket_dim_freeze` / `GpuTrainer::set_freeze_l1f`
/// 参照)。L1f (`uniform.l1f_w`/`l1f_b`) は寄与が厳密 0 であることを前提に
/// 加算するので (0 を足しても無害)、`freeze_l1f` が正しく機能していれば
/// 結果は変わらない。PSQT (`uniform.psqt_w`) は非対応 (`Some` なら error)。
///
/// bucket `g` の L1 の実際の行は「main 行 `[0, l1_out_per_bucket[g]-1)`」+
/// 「skip 行 (padded 側の `l1_out_max - 1` 番目)」で、後者を小さい側の
/// 最終行 (`l1_out_per_bucket[g]-1` 番目) に詰め直す (skip は常に「その
/// bucket の l1_out の最後の行」という `layerstack_v3` 側の規約に合わせる
/// ため)。L2 の入力列も対応する prefix だけを取る。
pub fn repack_from_uniform(
    uniform: &LayerStackWeights,
    l1_out_per_bucket: [usize; NUM_BUCKETS_V3],
    l2_out_per_bucket: [usize; NUM_BUCKETS_V3],
) -> Result<LayerStackV3Weights, String> {
    if uniform.num_buckets != NUM_BUCKETS_V3 {
        return Err(format!(
            "repack_from_uniform requires num_buckets == {NUM_BUCKETS_V3}, got {}",
            uniform.num_buckets
        ));
    }
    if uniform.psqt_w.is_some() {
        return Err("repack_from_uniform does not support PSQT (layerstack_v3 has no PSQT block)".to_string());
    }
    let ft_out = uniform.ft_b.len();
    if uniform.ft_w.len() != uniform.feature_set.base_ft_in() * ft_out {
        return Err(format!(
            "ft_w length {} != base_ft_in {} * ft_out {ft_out}",
            uniform.ft_w.len(),
            uniform.feature_set.base_ft_in(),
        ));
    }
    let l1_max = uniform.l1_b.len() / uniform.num_buckets;
    let l2_max = uniform.l2_b.len() / uniform.num_buckets;
    if l1_max == 0 || l1_max * uniform.num_buckets != uniform.l1_b.len() {
        return Err(format!(
            "l1_b length {} is not a multiple of num_buckets {}",
            uniform.l1_b.len(),
            uniform.num_buckets
        ));
    }
    if l2_max == 0 || l2_max * uniform.num_buckets != uniform.l2_b.len() {
        return Err(format!(
            "l2_b length {} is not a multiple of num_buckets {}",
            uniform.l2_b.len(),
            uniform.num_buckets
        ));
    }
    let l1_eff_max = l1_max - 1;
    let l2_in_max = l1_eff_max * 2;
    if uniform.l1_w.len() != uniform.num_buckets * l1_max * ft_out {
        return Err(format!(
            "l1_w length {} != num_buckets {} * l1_max {l1_max} * ft_out {ft_out}",
            uniform.l1_w.len(),
            uniform.num_buckets
        ));
    }
    if uniform.l1f_w.len() != ft_out * l1_max || uniform.l1f_b.len() != l1_max {
        return Err(format!(
            "l1f_w/l1f_b length ({}, {}) does not match ft_out {ft_out} * l1_max {l1_max} / l1_max",
            uniform.l1f_w.len(),
            uniform.l1f_b.len(),
        ));
    }
    if uniform.l2_w.len() != uniform.num_buckets * l2_max * l2_in_max {
        return Err(format!(
            "l2_w length {} != num_buckets {} * l2_max {l2_max} * l2_in_max {l2_in_max}",
            uniform.l2_w.len(),
            uniform.num_buckets
        ));
    }
    if uniform.l3_w.len() != uniform.num_buckets * l2_max || uniform.l3_b.len() != uniform.num_buckets {
        return Err(format!(
            "l3_w/l3_b length ({}, {}) does not match num_buckets {} * l2_max {l2_max} / num_buckets",
            uniform.l3_w.len(),
            uniform.l3_b.len(),
            uniform.num_buckets
        ));
    }
    for g in 0..NUM_BUCKETS_V3 {
        if !(2..=l1_max).contains(&l1_out_per_bucket[g]) {
            return Err(format!(
                "l1_out_per_bucket[{g}] = {} must be in [2, {l1_max}]",
                l1_out_per_bucket[g]
            ));
        }
        if !(1..=l2_max).contains(&l2_out_per_bucket[g]) {
            return Err(format!(
                "l2_out_per_bucket[{g}] = {} must be in [1, {l2_max}]",
                l2_out_per_bucket[g]
            ));
        }
    }

    let mut out = LayerStackV3Weights::zeroed(
        uniform.feature_set,
        ft_out,
        l1_out_per_bucket,
        l2_out_per_bucket,
    );
    out.ft_w = uniform.ft_w.clone();
    out.ft_b = uniform.ft_b.clone();

    for g in 0..NUM_BUCKETS_V3 {
        let l1_eff_g = l1_out_per_bucket[g] - 1;
        let l2_out_g = l2_out_per_bucket[g];

        // ---- L1: main 行 [0, l1_eff_g) をそのまま、skip 行 (padded 側の
        //      l1_eff_max 番目) を最終行 (l1_eff_g 番目) に詰め直す。L1f を
        //      加算してから (0 のはずだが、防御的に常に加算する) 抽出する。
        for row in 0..l1_eff_g {
            let src_base = (g * l1_max + row) * ft_out;
            let dst_base = row * ft_out;
            for k in 0..ft_out {
                let l1f = uniform.l1f_w[k * l1_max + row];
                out.l1_w[g][dst_base + k] = uniform.l1_w[src_base + k] + l1f;
            }
            out.l1_b[g][row] = uniform.l1_b[g * l1_max + row] + uniform.l1f_b[row];
        }
        {
            let skip_row = l1_eff_max; // padded 側の skip 行 index
            let src_base = (g * l1_max + skip_row) * ft_out;
            let dst_base = l1_eff_g * ft_out;
            for k in 0..ft_out {
                let l1f = uniform.l1f_w[k * l1_max + skip_row];
                out.l1_w[g][dst_base + k] = uniform.l1_w[src_base + k] + l1f;
            }
            out.l1_b[g][l1_eff_g] = uniform.l1_b[g * l1_max + skip_row] + uniform.l1f_b[skip_row];
        }

        // ---- L2: 出力行 [0, l2_out_g)、入力列は sqr 半分 [0, l1_eff_g) +
        //      main 半分 [l1_eff_max, l1_eff_max + l1_eff_g) の prefix だけ。
        let l2_in_g = l2_in_for(l1_out_per_bucket[g]);
        debug_assert_eq!(l2_in_g, l1_eff_g * 2);
        for row in 0..l2_out_g {
            let src_base = (g * l2_max + row) * l2_in_max;
            let dst_base = row * l2_in_g;
            for c in 0..l1_eff_g {
                out.l2_w[g][dst_base + c] = uniform.l2_w[src_base + c];
            }
            for c in 0..l1_eff_g {
                out.l2_w[g][dst_base + l1_eff_g + c] = uniform.l2_w[src_base + l1_eff_max + c];
            }
            out.l2_b[g][row] = uniform.l2_b[g * l2_max + row];
        }

        // ---- L3: 出力(=入力)行 [0, l2_out_g)。
        for row in 0..l2_out_g {
            out.l3_w[g][row] = uniform.l3_w[g * l2_max + row];
        }
        out.l3_b[g] = uniform.l3_b[g];
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(v: usize) -> [usize; NUM_BUCKETS_V3] {
        [v; NUM_BUCKETS_V3]
    }

    #[test]
    fn parse_bucket_dims_csv_accepts_nine_naturals() {
        let csv = "15,15,15,20,20,20,25,25,25";
        let parsed = parse_bucket_dims_csv(csv, "--l1").unwrap();
        assert_eq!(parsed, [15, 15, 15, 20, 20, 20, 25, 25, 25]);
    }

    #[test]
    fn parse_bucket_dims_csv_rejects_wrong_count() {
        assert!(parse_bucket_dims_csv("1,2,3", "--l1").is_err());
    }

    #[test]
    fn parse_bucket_dims_csv_rejects_zero() {
        assert!(parse_bucket_dims_csv("0,1,1,1,1,1,1,1,1", "--l1").is_err());
    }

    #[test]
    fn network_hash_v3_is_order_sensitive() {
        let l2_a: [usize; NUM_BUCKETS_V3] = [32, 32, 32, 32, 32, 32, 32, 32, 40];
        let mut l2_b = l2_a;
        l2_b.reverse();
        let ha = network_hash_v3(0x1234, 1536, &l2_a);
        let hb = network_hash_v3(0x1234, 1536, &l2_b);
        assert_ne!(ha, hb, "reordering per-bucket l2_out should change the hash");
    }

    #[test]
    fn save_then_load_round_trips_heterogeneous_buckets() {
        let feature_set = shogi_features::FeatureSet::HalfKp.spec();
        let ft_out = 128;
        let mut l1_out = uniform(16);
        let mut l2_out = uniform(32);
        l1_out[3] = 20;
        l2_out[6] = 48;

        let mut w = LayerStackV3Weights::zeroed(feature_set, ft_out, l1_out, l2_out);
        // 適当な非ゼロ値を詰めて quantise の丸めが破綻しないことも合わせて確認する。
        for (feat_row, v) in w.ft_w.chunks_mut(ft_out).zip(0..) {
            for (o, x) in feat_row.iter_mut().enumerate() {
                *x = ((v * 31 + o) % 7) as f32 * 0.01 - 0.03;
            }
        }
        for buc in 0..NUM_BUCKETS_V3 {
            for (i, x) in w.l1_b[buc].iter_mut().enumerate() {
                *x = (i as f32 * 0.1) - 0.2;
            }
            for (i, x) in w.l1_w[buc].iter_mut().enumerate() {
                *x = ((i % 5) as f32 * 0.02) - 0.04;
            }
            for (i, x) in w.l2_b[buc].iter_mut().enumerate() {
                *x = (i as f32 * 0.05) - 0.1;
            }
            for (i, x) in w.l2_w[buc].iter_mut().enumerate() {
                *x = ((i % 3) as f32 * 0.03) - 0.03;
            }
            for (i, x) in w.l3_w[buc].iter_mut().enumerate() {
                *x = (i as f32 * 0.01) - 0.02;
            }
            w.l3_b[buc] = buc as f32 * 0.01;
        }

        let mut buf = Vec::new();
        w.save_quantised(&mut buf).expect("save_quantised failed");

        let loaded =
            LayerStackV3Weights::load_quantised(&mut buf.as_slice(), feature_set, ft_out, l1_out, l2_out)
                .expect("load_quantised failed");

        assert_eq!(loaded.l1_out, l1_out);
        assert_eq!(loaded.l2_out, l2_out);
        for buc in 0..NUM_BUCKETS_V3 {
            assert_eq!(loaded.l1_b[buc].len(), l1_out[buc]);
            assert_eq!(loaded.l1_w[buc].len(), l1_out[buc] * ft_out);
            assert_eq!(loaded.l2_b[buc].len(), l2_out[buc]);
            assert_eq!(loaded.l2_w[buc].len(), l2_out[buc] * l2_in_for(l1_out[buc]));
            assert_eq!(loaded.l3_w[buc].len(), l2_out[buc]);
        }
    }

    #[test]
    fn load_with_mismatched_l1_out_fails_via_byte_misalignment() {
        // l1_out/l2_out はもう file に自己記述されていない (yaneuraou 側も
        // 同様に dims を file から読まず、コンパイル時の `nnue_arch_gen.py`
        // 引数から決め打ちする — 詳細はこの module の doc 冒頭のコメント参照)。
        // そのため呼び出し側が間違った `expected_l1_out` を渡しても、file
        // フォーマット的に「file が壊れている」ことを検出する仕組みは無く、
        // 単純にバイト列の消費位置がずれて読み込みが破綻する (典型的には
        // 途中で EOF に達して `UnexpectedEof` になる)。この test は
        // 「必ず何らかの error になる (=garbage を静かに読み込んで
        // 成功したことにはならない)」ことだけを確認する。
        let feature_set = shogi_features::FeatureSet::HalfKp.spec();
        let ft_out = 128;
        let l1_out = uniform(16);
        let l2_out = uniform(32);
        let w = LayerStackV3Weights::zeroed(feature_set, ft_out, l1_out, l2_out);
        let mut buf = Vec::new();
        w.save_quantised(&mut buf).unwrap();

        let mut bad_l1 = l1_out;
        bad_l1[0] = 24;
        let result = LayerStackV3Weights::load_quantised(
            &mut buf.as_slice(),
            feature_set,
            ft_out,
            bad_l1,
            l2_out,
        );
        assert!(
            result.is_err(),
            "loading with a mismatched l1_out must fail (byte stream misalignment), not silently succeed"
        );
    }
}
