# layerstack_v3: bucketごとに個別サイズの L1/L2 を持つ LayerStack

## 背景

`layerstack` (V2) は `--ft-out` / `--l1` / `--l2` / `--num-buckets` がいずれも
**全 bucket 共通の単一値**だった (2026-05-22 / 2026-05-23 の decision で configurable
になったのもこの「全bucket同一サイズ」の範囲内)。本 decision は、yaneuraou の
`SFNNwoP1536_V2` (`sfnnwop-1536-v2.h`) 相当のアーキテクチャに対して

1. yaneuraou 側に `LS_BUCKET_MODE=progress9kpabs` (9bucket全部を使う progress-kpabs
   binning。既存の `progress8kpabs` は歴史的経緯で9bucket中0..7の8bucketしか
   使っていなかった) を追加する。
2. `ft_out` は bucket 間で共通のまま、`l1_out` / `l2_out` を **bucketごとに個別
   サイズ** で指定できるようにする (yaneuraou 側 `SFNNwoP_V3`、tatara 側
   `layerstack_v3`)。

を実装するにあたっての、実装済み範囲と明示的なスコープ外範囲を記録する。

## 実装済み

### yaneuraou (C++ 推論エンジン、engine 側)

- `evaluate_nnue.cpp`: `LSBucketMode::Progress9KPAbs` を追加。9個の閾値
  (`ln(k/(9-k))`, k=1..8) で `floor(sigmoid(progress_sum) * 9)` 相当の9bucket
  binning を行う。`LS_BUCKET_MODE` USI オプションに `progress9kpabs` 文字列を
  追加。
- `SFNNwoP_V3` architecture: `nnue_arch_gen.py --l1 <csv9> --l2 <csv9>` で
  bucketごとに異なる `NetworkBucket<L1,L2,Hash>` (template) を9個生成し、単一の
  `Network` 構造体に集約する。`NnueNetworks::network[]` は (既存の homogeneous
  array 前提を壊さないよう) 要素数1のまま (`LayerStacks = 1`) にして、実際の9
  bucket分は集約された`Network`内部に持たせ、`Network::Propagate(features,
  buffer, bucket)` の引数でbucket選択する設計にした (`evaluate_nnue.h` /
  `NnueNetworks` 側の変更は不要、`evaluate_nnue.cpp` の呼び出し部のみ
  `#if defined(SFNNwoPSQT_V3)` で分岐)。
  - `Makefile`: `YANEURAOU_ENGINE_NNUE_SFNNwoP_V3-<ft>` エディション、
    `NNUE_L1` / `NNUE_L2` (カンマ区切り9個の自然数) make変数を追加。
    `SFNNWOP` を禁止する既存 guard から `SFNNWOP_V3` prefix のみ例外化。
  - **動作確認**: このセッション内で実際に `make normal` を実行し (g++
    13.3.0、AVX2)、
    - `NNUE_L1`/`NNUE_L2` が全bucket同一値 (V2互換) の場合
    - bucketごとに異なる値 (`15,15,15,20,20,20,25,25,25` /
      `32,32,32,40,40,40,48,48,48`) の場合
    の両方で `YaneuraOu-by-gcc` がリンクまで成功することを確認した。また
    `setoption name LS_BUCKET_MODE value progress9kpabs` が USI 経由で
    エラー無く受理されることも確認した (実際の`nn.bin`が無いため評価計算
    自体は未検証)。

### tatara (`nnue-format` crate、`.bin` フォーマット層)

- 新規 `crates/nnue-format/src/layerstack_v3_weights.rs`:
  `LayerStackV3Weights` (bucketごとに `l1_out[i]` / `l2_out[i]` が異なる
  ragged array 表現) の `zeroed` / `save_quantised` / `load_quantised`。
  yaneuraou の `SFNNwoP_V3` header が読む1bucket分の byte layout
  (`fc_0`=L1 sparse affine, `ac_0`+`ac_sqr_0`=活性化, `fc_1`=L2 affine,
  `ac_1`=活性化, `fc_2`=L3 affine) と一致するように書く。
  `--l1` / `--l2` のカンマ区切り9個の自然数を parse するユーティリティ
  ([`parse_bucket_dims_csv`]) も同モジュールに置いた。
  round-trip (save → load) の unit test と、bucket間でサイズが異なる場合の
  load-time reject (サイズ不一致 → `InvalidData`) の unit test を含む。
- `ArchKind::LayerStackV3` (`canonical_name = "layerstack_v3"`) を追加し、
  artifact identity として区別できるようにした。

## スコープ外 (未実装、フォローアップが必要)

**GPU trainer (`bins/nnue_train`) 側で実際に bucketごと異なるサイズを
"学習" することは、本 decision の範囲では実装していない。**

理由: `arch.rs` のコメント通り、現行の per-bucket backward kernel
(`dense_mm_bwd_weight_bucket_tiled_{l2,l3}` 等) は「**固定9レジスタ
アキュムレータ (`a0..a8`) による register fan-out**」で9bucket分を1回の
kernel launchで処理する設計になっている。この設計は **全bucketが同一の
出力次元 (`l1_out`/`l2_out`)** であることを前提にしており (同じ tile幅・
同じレジスタ幅で9bucket分をまとめて計算する)、bucketごとに出力次元が
異なる場合はこの前提が成立しない。

これを正しく解決するには、以下のいずれかの kernel 側の作り直しが必要:

1. **bucketごとに別kernel launchにする** (`blockIdx.z` 等でbucketを
   分離し、各launchが自分のbucketの出力次元だけを知っていればよい形に
   register fan-outをやめる)。実装は比較的素直だが、kernel launch回数が
   9倍になるオーバーヘッドの実測が要る。
2. **全bucketをmax(l1_out)/max(l2_out)にpaddingして学習し、export時に
   truncateする** — 一見簧単に見えるが **不健全**: skip dim (`l1_out-1`
   番目、既存 `L1_SKIP` 定数) が常に「そのbucketの出力次元の最後の
   index」である前提と矛盾する (padding後の末尾はpaddingの0であって
   skip dimではない)。加えて、小さい`l1_out`のbucketも実際にはmax
   サイズで学習されてしまい、「小さいネットワークとして学習された」
   ことにならない (単なる係数を後から切り詰めるだけ)。**本 decision
   ではこの方式は採用しない** (このファイルで明示的に否定しておく:
   将来 "手っ取り早い実装" として再提案されたときに、上記の不健全さを
   再発見する手間を省くため)。

いずれの方式でも、実際の学習が数値的に妥当であることの検証 (小さい
bucketが本当に小さい容量で収束するか、kernel の実測性能) が必要で、
GPU が無いこの変更のsessionでは検証しようがない。そのため:

- **フォワードプラン**: 上記 (1) の「bucketごとに別kernel launch」を
  次の変更で実装する。`GpuWorkspace` に `l1_out: [usize;9]` /
  `l2_out: [usize;9]` を持たせ、forward/backwardの各kernelを
  `for bucket in 0..9 { launch(..., l1_out[bucket], l2_out[bucket]) }`
  にする形が現実的な最小変更になる見込み (dense_mm_bucket.rs の
  1bucket分のtiled kernelをそのまま複数回呼ぶだけで済む可能性がある —
  現行がまとめて9bucket分を1回のlaunchでregister fan-outしているのを
  「1bucketずつ9回launch」に分解するだけなら、kernelのコード自体は
  大きく変えずに済むかもしれない。要調査)。
- それまでの間、`layerstack_v3_weights.rs` の format は
  「手で組んだ (あるいは他ツールで学習した) per-bucket 重みを
  yaneuraou `SFNNwoP_V3` 向けに書き出す」用途には今すぐ使える。

## 非対応機能 (V3 の初期実装で意図的に外したもの)

- L1f (shared factorized L1、`layerstack_weights` 参照): bucketごとに
  サイズが異なると素直な shared factorizer が組みにくいため、V1実装では
  省略した。per-bucket L1 を直接学習する前提。
- PSQT shortcut / Threat feature: `layerstack_v3_weights` は現状どちらも
  `Unsupported` で reject する。必要になった時点で `layerstack_weights`
  の該当ロジックを移植する。
