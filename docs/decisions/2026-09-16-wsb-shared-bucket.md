# `wsb` (WithSharedBucket): 常に選ばれる共有バケットの追加

- **Status**: Implemented (2026-09-16 追記) — `bins/nnue_train/src/trainer_layerstack.rs`
  (forward/backward) と `bins/nnue_train/src/kernels/layerstack.rs`
  (`average_inplace`/`add_inplace`/`scale_inplace` の3個の小さい新規kernel) に
  実装済み。**cuda-oxide (nightly rustc + 独自PTX codegenバックエンド) が使える
  環境が無く、GPU実機での動作確認・`local-ci.sh`実行はできていない** —
  マージ前に必ず `local-ci.sh` を通すこと (下記「実装の検証状況」参照)。

## Context

YaneuraOu `architectures/nnue_arch_gen.py` (SFNN layer-stack 生成器) に、
バケット名の最後に `wsb` トークンを置くと、hand/king/progress/router の
合成バケットに加えて「局面に依らず常に評価される共有バケット」を1個追加し、
評価値を「選択された1バケットの出力」と「共有バケットの出力」の平均にする
機能を追加した (engine 側: `evaluate_nnue.h`/`.cpp` の
`NNUE_SFNN_USE_SHARED_BUCKET`)。tatara 側もこの net を実際に学習できるように
する必要がある。

## Decision

### 1. バケット合成・フォーマット層 (実装済み)

`wsb` を「hand/king/progress/routerの合成バケット数 (`selectable_buckets()`)
に対して常に index = `selectable_buckets()` (= 合成バケット数そのもの、必ず
最後の1個) を割り当てる、常時選択の追加バケット」として
[`shogi_features::bucket_mode::BucketMode`] に実装した:

- `BucketMode::shared_bucket: bool` フィールドを追加。
- `BucketMode::parse` は末尾トークン `wsb` (大文字小文字不問) を認識する。
  **文字列上、必ず最後のトークンでなければならない** (YaneuraOu
  `nnue_arch_gen.py` の検証と同じ規則、`wsb_k3k3` や `k3k3_wsb_progress4` は
  reject)。
- `BucketMode::selectable_buckets()`: hand×king×progress×router の積
  (router併用時は router が最後の桁、`wsb` を含まない)。
- `BucketMode::total_buckets()`: `selectable_buckets() + (shared_bucket as u32)`。
- `BucketMode::shared_bucket_index()`: `wsb` 有効時 `Some(selectable_buckets())`、
  無効時 `None`。
- `BucketMode::canonical_token()`: 末尾に `"wsb"` を追加
  (例: `"hand64z_k9k9_progress4_wsb"`)。

`combine_bucket_index()` (位置→バケットindexの合成関数) 自体は**変更していない**
— 常に `0..selectable_buckets()` の範囲を返す。共有バケットの index
(`shared_bucket_index()`) は合成規則の一部ではなく、常に選ばれる別枠の
定数インデックスとして扱う。

**フォーマット層への波及は自動**:

- `crates/nnue-format/src/yaneuraou.rs` の `save_yaneuraou`/`save_yaneuraou_combined`
  は `bucket_mode.total_buckets() == weights.num_buckets` を検証してから
  `for bucket in 0..arch.num_buckets` で全 bucket の重み (L1/L2/L3) を書く
  だけなので、`total_buckets()` が既に +1 を含む以上、共有バケット用の
  weight section (index = `selectable_buckets()`、常に最後) は既存コードの
  ままで正しく書き出される。特別な分岐は不要だった。
- `arch_string()` も `bucket_mode.canonical_token()` をそのまま
  `Network=SFNN_..._WSB` のアーキ名に埋め込むので、YaneuraOu
  `nnue_arch_gen.py` が生成する `GetStructureString()` と自動的に一致する
  (router/progress と同様、`wsb` 専用の `{...}` トークンは追加していない —
  `evaluate_nnue.cpp::GetArchitectureString()` 側も同じ理由で追加していない)。
- `crates/nnue-format/src/layerstack_weights.rs` (tatara 独自の checkpoint /
  量子化フォーマット) は `BucketMode` を一切知らず `num_buckets: usize` を
  そのまま扱うだけなので変更不要 (`docs/decisions/2026-05-23-num-buckets-configurable.md`
  の「任意の N をサポートする」設計のおかげで `wsb` の +1 もそのまま通る)。

これで **`--bucket-mode ..._wsb` を CLI に渡して重み配列を
`total_buckets()` 個確保し、YaneuraOu 形式で保存する所まで**は
既存コードパスにそのまま乗る。残っているのは学習時の forward/backward
で実際に「選択バケット + 共有バケットの平均」を計算する部分。

### 2. GPU forward/backward (未実装 — 設計のみ)

`bins/nnue_train/src/trainer_layerstack.rs` の `GpuTrainer::forward`
(L1886〜) / `backward` (L2713〜) は、feature transformer (FT) の出力
(`self.ws.combined`、bucket に依存しない) を1回計算した後、L1/L2/L3 を
**position ごとに1個の bucket** (`batch.bucket_idx`、host が
`dataloader::bucket_board()` で事前に確定した値) だけで forward/backward
する設計になっている (`l1_bucket_segments` が `batch.bucket_idx` を
sorted-by-bucket の tiled 行列積用にセグメント化する)。

好都合なことに、この「FT forward を再利用しつつ、L1以降だけを別の
`bucket_idx` でやり直す」経路は **`GpuTrainer::validate_reuse_ft`
としてforward方向にはすでに存在する** (router の E-step: 同じ batch を
`--num-buckets` 通りの固定 bucket で forward-only sweep するための機構、
`self.ws.combined` に前回の FT-post 出力が残っている前提で `bucket_idx`
だけ差し替える)。`wsb` の forward はこれと全く同じパターン
(「同じ FT 出力を、2種類の bucket_idx で L1 以降だけ2回通す」) だが、
backward + optimizer まで含めた学習ループ向けにこの機構を拡張する必要が
ある。

**設計**:

1. `bucket_mode.shared_bucket_index()` が `Some(k)` のとき、host 側
   (`dataloader` から渡す `BatchData`、または `step_impl` 冒頭) で、
   `batch.bucket_idx` (選択バケット、既存のまま) に加えて、全行を `k` に
   固定した `shared_bucket_idx: Vec<i32>` (長さ `batch.n_pos`、全要素 `k`)
   を用意する。全要素が同じ値なので `l1_bucket_segments` 相当の
   セグメント化は既存の1-bucketケースと同じ (1個のセグメント、全行)。
2. **Forward**: FT forward (`sparse_ft_forward` × 2 + `ft_post_perspective_fwd`)
   を1回だけ実行し `self.ws.combined` を作る (変更なし)。その後 L1→L2→L3 の
   forward を **2回**実行する: 1回目は `batch.bucket_idx` (選択バケット、
   既存のまま `self.ws.*` に書く)、2回目は `shared_bucket_idx` (専用の
   workspace buffer、例えば `self.ws.combined_shared_out` 的な別領域に書く
   — `self.ws.combined` 自体は使い回すだけで上書きしない)。最終出力は
   `net_output = 0.5 * (out_selected + out_shared)` (elementwise、batch分)。
   loss kernel はこの平均済み `net_output` に対して通常通り評価する。
3. **Backward**: loss kernel から得られる `d(loss)/d(net_output)`
   (既存の `dl1_total`/`loss_acc` 相当の初期勾配) を **0.5倍したもの**を、
   選択バケット経路・共有バケット経路の**両方**に独立して流し、
   それぞれ L3→L2→L1 backward を行う (per-bucket weight の勾配は
   sorted + split-K kernel の atomicAdd で書くので、選択バケット経路と
   共有バケット経路が異なる bucket group への書き込みである限り
   衝突しない。まれに選択バケット自体が共有バケットの index と一致する
   ことは `combine_bucket_index()` の値域が `0..selectable_buckets()`
   であり `shared_bucket_index() == selectable_buckets()` なので原理的に
   起こらない)。両経路それぞれが `d(loss)/d(combined)` (FT出力に対する勾配)
   を生成するので、FT backward に渡す前に **この2つを elementwise 加算**
   する (FT出力は2回の forward で共有して読まれているため、backward は
   両方からの寄与の和になる — 通常の「同じ activation を2回使う」ノードの
   backprop則そのもの)。
4. Optimizer step は変更不要 (per-bucket weight の勾配 buffer は bucket
   ごとに独立な領域のままで、共有バケットも「total_buckets 個の bucket の
   うちの1個」として同じ optimizer state 配列に乗る)。

**実装上の注意**:

- 上記 2. の「2回目の forward 用の別 workspace 領域」「3. の2経路分の
  勾配を加算してから FT backward に渡す」処理は、`forward`/`backward`
  内の bucket 依存部分 (L1 以降) を丸ごと関数として切り出し、
  `(bucket_idx, out_buffer, grad_scale)` を引数にして2回呼べるようにする
  リファクタが前提になる (現状は forward/backward 1本の巨大関数に
  インライン展開されている)。
- `wsb` 無効時 (`bucket_mode.shared_bucket == false`、既存の全 net) は
  この2回目の forward/backward 呼び出しを完全にスキップし、既存の1回だけの
  経路をそのまま通ることで、既存挙動を1bit も変えない (`if
  self.bucket_mode.shared_bucket { ... }` で分岐)。
- `l1_bucket_segments` は `batch.bucket_idx` を引数に取る形に既になって
  いるので、`shared_bucket_idx` を渡せばそのまま使える (変更不要)。
- validation (`GpuTrainer::validate`) 側も同じ2回forward+平均を通す必要が
  ある (loss/accuracy が学習時と整合するように)。`validate_reuse_ft`
  ベースの E-step (router) とは独立な経路なので、`wsb` と `router` の
  併用時は「E-step の N-way sweep の**各** bucket で、さらに共有バケットとの
  平均を取る」形になる (`combine_bucket_index` の router 部分と `wsb` は
  独立な軸なので両立できる設計にはなっている)。

### 2b. 「batch doubling」不要 — 実装レシピ (2026-09-16 追記、数値検証済み)

上記 §2 の初稿では forward/backward を丸ごと2回通す設計を提案したが、
実際に `gpu-kernels::layerstack::dense_mm_bucket::*_cpu` (本番 GPU kernel
`dense_mm_fwd_bucket` / `dense_mm_bwd_input_bucket` /
`dense_mm_bwd_weight_bucket` / `bias_grad_bucket` の CPU reference。
`bins/nnue_train` の `gpu_cpu_equivalence_tests` がこの CPU reference と
本番 kernel の数値一致を保証している) と `crelu_fwd_cpu`/`crelu_grad_cpu`
を使い、トイ2層ネット (FT出力 x → L1 per-bucket affine → CReLU → L2
per-bucket affine(out=1) → loss) で `wsb` の forward 平均 + backward
(0.5分割 + FT勾配の和) を実装し、有限差分と比較する数値検証を行った
(`gpu-kernels` は GPU/cuda-oxide 非依存の pure Rust crate なので stable
rustc で単体ビルド・実行できる)。結果: forward は完全一致
(`0.25116065 == 0.25116065`)、全パラメータ (w1/b1/w2/b2) と共有入力 x の
勾配が有限差分と一致 (max relative error ≈ 3.8e-4、f32 有限差分の
ノイズ床相当)。この検証を通じて、当初想定より単純で低リスクな実装方式が
判明した:

**L2 / L3 は「plain kernel を2回呼ぶ」だけでよい。** `dense_mm_fwd_bucket` /
`dense_mm_bwd_*_bucket` / `bias_grad_bucket` は `bucket_idx` (per-row) と
`num_buckets` を runtime 引数に取る汎用 kernel で、`trainer_layerstack.rs`
では L2/L3 forward/backward に**既にこの plain 版がそのまま**使われている
(L1 だけが sort/tile 最適化版 `dense_mm_fwd_bucket_tiled_l1_sorted` 等を使う
— 理由は L1 の in_dim (`ft_out`、既定 1536) が L2/L3 よりずっと大きく
sort によるタイル局所性が効くため)。したがって WSB の L2/L3 は:

- forward: 選択bucket用に**既存のまま** `dense_mm_fwd_bucket` を呼ぶのに加え、
  共有bucket用にもう1回、同じ kernel を「全行 = `shared_bucket_index()` の
  定数 `bucket_idx` バッファ」で呼ぶ (出力は別バッファ `l2_dense_out_shared`
  等)。
- backward: 同様に bwd_input/bwd_weight/bias_grad をそれぞれもう1回、
  共有bucket用の `bucket_idx`・別バッファで呼ぶ。`dense_mm_bwd_weight_bucket`
  はbucket群ごとに書き込み先を分離する (§Context の
  `dense_mm_bwd_weight_bucket_cpu` docstring 参照) ので、選択bucket呼出し
  (対象 index `0..selectable_buckets()`) と共有bucket呼出し (対象 index
  `selectable_buckets()` のみ) が**同じ `grad_w` 配列全体**に書いても
  (num_buckets = `total_buckets()` を渡す限り) 互いの担当 bucket 以外を
  破壊しない ── ただし両呼出しとも「担当外の全 bucket を 0 埋めする」
  overwrite 実装であるべき点に注意 (今回検証した `*_cpu` reference は
  「対象 bucket のみ書く」形なので、本番 kernel も同じ契約かどうかは
  実装時に kernel 側の実際のコードで確認すること — 異なる場合は grad_w を
  `selectable_buckets()` 分と共有分の**別バッファ**に分けて後で
  memcpy/結合する方が安全)。

**L1 は共有bucket側だけ sort/tile を丸ごとスキップできる。** 共有bucketの
`bucket_idx` は全行が同じ定数なので、そもそも sort する意味が無い —
選択bucket側の複雑な sort/permute/tiled-L1/tf32-segment 経路
(`bucket_counts_dev` 等の scratch 一式) は**選択bucket側のみ既存のまま**
使い、共有bucket側は L2/L3 と同じ plain `dense_mm_fwd_bucket` /
`dense_mm_bwd_*_bucket` を `in_dim=ft_out, out_dim=l1_out` で直接呼べば
よい (`l1_bucket_shared` 等、新規 scratch は不要)。

**L1f (`l1f_w`、bucket 非依存の共有 dense head) は複製しない。** `l1_total
= l1_bucket + l1f_out` の `l1f_out` は `combined` のみに依存し bucket に
依らないので、選択branch/共有branchで**同じ値を1回だけ計算**すればよい。
ただし backward はその逆で、`l1f_w` は**両branchから勾配を受け取る**:
`d(loss)/d(l1f_out) = dl1_total_selected + dl1_total_shared` (elementwise_add
で1回summeすればよい、`l1_total`の加算がbranch内で恒等写像だから)。この
summeした `dl1_total_summed` を使って `l1f_w` の weight-grad
(`sgemm_xt_y_rowmajor(combined, dl1_total_summed, l1f_w_grad)`) と
`dcombined_from_l1f` (`dl1_total_summed @ l1f_w^T`) を**それぞれ1回だけ**
計算する (l1f 関連のcuBLAS呼出しはoverwrite契約なので、2回呼ぶのではなく
先に `dl1_total` を2branch分足してから1回呼ぶ)。

**最終的な `d(combined)`** (FT backwardへ渡す勾配) は現行コードの
`dcombined_from_l1 + dcombined_from_l1f` (2項) から
`dcombined_from_l1_selected + dcombined_from_l1_shared + dcombined_from_l1f`
(3項、最後の1項は上記の通りbranch合算後の1回計算) になる。

**必要な新規 buffer** (すべて `b × dim` サイズ、`bucket_mode.shared_bucket`
時のみ確保、既存 buffer の再サイズは不要 — L1 sort scratch も含め既存
buffer は一切拡張しない):

```
l1_bucket_shared, l1_total_shared, l1_main_shared, l1_skip_shared,
l1_sqr_shared, l2_pre_shared, l2_input_shared, l2_dense_out_shared,
l2_acted_shared, l3_out_shared,
dl2_acted_shared, dl2_out_shared, dl2_input_shared, dl2_pre_shared,
dl1_sqr_shared, dl1_main_from_concat_shared, dl1_main_from_sqr_shared,
dl1_main_shared, dl1_total_shared,
dl1_total_summed (= dl1_total + dl1_total_shared、l1f 用),
shared_bucket_idx_dev (b要素、全行 shared_bucket_index() の定数、
  step開始時に1回 memset 的に埋めればよい — 各stepで値は変わらない)
```

forward 末尾: `net_output[i] = 0.5 * (l3_out[i] + l3_out_shared[i])`
(新規 elementwise kernel、または既存 `elementwise_add` + `abs_pow2_scale_fwd`
相当のscaleで代用可)。backward 先頭: `dy = 0.5 * dy_net_output` を選択
branch・共有branch**両方**の初期勾配として使う (現行の `dl2_acted` 生成
loss kernel の出力を単純に半分にして両方に配る)。

この設計は L1 の sort/tile/tf32 経路の**サイズ計算に一切触れない**
(`padded_sort_batch`、`n_out_tiles`、`bucket_counts_dev` 等はすべて
選択branch用のまま不変)。触れるのは (1) 新規 `*_shared` buffer の追加、
(2) L2/L3/共有L1 の kernel launch をコピー＆bucket_idx/出力先だけ差し替え、
(3) L1f 部分の「2 branch の dl1_total を先に足してから1回呼ぶ」という
呼び出し順の変更、(4) forward末尾の平均・backward先頭の0.5分配、の4点に
限定できる。

この検証は `crates/gpu-kernels/src/layerstack/wsb_design_check.rs`
(`#[cfg(test)]`、stable rustc でも実行可能) として恒久テスト化した:
`wsb_forward_matches_naive_branch_average` / `wsb_backward_matches_finite_differences`。

### 3. 実装 (2026-09-16、`trainer_layerstack.rs` / `kernels/layerstack.rs`)

上記 §2b のレシピに沿って実装した。ユーザーの要望により、cuda-oxideが
使えない (実行検証できない) 制約を承知の上で実装まで進めた。以下、実装の
要点と検証状況を記録する。

**`GpuWorkspace` への追加**: `bucket_mode.shared_bucket` が true のときのみ
確保される `Option<DeviceBuffer<f32>>` フィールド群 (`l1_bucket_shared` から
`dcombined_from_l1_shared`、`dl1_total_summed`、`shared_bucket_idx_dev` まで
約20個)。全て既存の `b × dim` サイズ (batch doubling はしない)。
`GpuWorkspace::new` に `shared_bucket_index: Option<usize>` 引数を追加し、
`GpuTrainer::new` から `bucket_mode.shared_bucket_index().map(|v| v as usize)`
を渡す。既存呼び出し (wsb無効) は `None` で従来通り (メモリ・挙動とも無変更)。

**新規 kernel (3個、`kernels/layerstack.rs` の cuda-oxide 版 +
`crates/cuda-native-runtime/kernels/native_kernels.cu` の native-cuda-host 版、
両方に実装が必要)**: 既存の `elementwise_add`
(`c[i]=a[i]+b[i]`、出力 `c: DisjointSlice` が入力と host 側で同時に別borrow
必要) では in-place 更新 (`a`自身を読みながら`a`に書く) ができないため、
最小限の in-place 版を3個追加した:
- `average_inplace(mut a, b, n)`: `a[i] = 0.5*(a[i]+b[i])`。forward末尾の
  `net_output` 平均に使う。
- `add_inplace(mut a, b, n)`: `a[i] += b[i]`。`dcombined_from_l1` に共有branch
  分を畳み込むのに使う。
- `scale_inplace(mut a, scale, n)`: `a[i] *= scale`。forward直後に
  `dy_net_output` を0.5倍し、以降backwardの選択branch・共有branch両方が
  この同じ (既に0.5倍済の) buffer をそのまま読めるようにする (既存の選択
  branch側backwardコードを一切変更せずに済む鍵)。

いずれも `#[kernel]` 属性を付けるだけで自動的にPTX artifactに載る設計
(`kernels/mod.rs` のdoc参照、手動registration不要)。**cuda-oxide 版は
kernel source 編集後に `bash scripts/build-kernels.sh` (= `cargo-oxide
build` + `.ll`→`.ptx`) を再実行しないと反映されない** (`cargo build` では
自動生成されない、cuda-oxideの既知の落とし穴。実際にこれが原因で
`average_inplace` が `CUDA_ERROR_NOT_FOUND` になった)。**native-cuda-host
版は `native_kernels.cu` を丸ごと nvcc でコンパイルするだけ
(`build.rs` に `cargo:rerun-if-changed=kernels/native_kernels.cu` 済) なので
通常の `cargo build --features native-cuda-host` で自動的に反映される**。
両バックエンドの呼び出し規約は同一 (`&[f32]`/`DisjointSlice<f32>` の
slice引数は native 側で `(const/非const float* ptr, unsigned long long
<未使用長>)` の2引数に展開される、`elementwise_add`等の既存native kernelと
同じ形)。

**forward**: 「Forward step 14」(選択branchの`net_output`計算) の直後に、
`bucket_mode.shared_bucket` のとき、共有branch用の同じ計算列
(L1→L1total→slice→sqr→concat→CReLU→L2→CReLU→L3→net_output_shared) を
`shared_bucket_idx_dev` (全行同一の定数bucket_idx) で実行する。L1は選択
branch側のsort/tile機構 (`bucket_counts_dev`等) を一切使わず、L2/L3と同じ
`dense_mm_fwd_bucket` (plain kernel) をそのまま使う。`l1f_out` は選択branchで
計算済のものをそのまま再利用 (再計算しない)。最後に `average_inplace` で
`net_output` (選択branch) に共有branchの寄与を0.5ずつ織り込む。

**backward**: loss kernel直後に `scale_inplace(dy_net_output, 0.5)` を1回。
これにより既存の選択branch側backward (L3→L2→L1eff) は無変更のまま正しい
(0.5倍済の) 初期勾配を使う。「Backward 6 reverse」の直後 (L1fの直前) に
共有branchのL3→L2→L1eff backwardを追加し、`dl1_total_shared` を計算後、
`dl1_total_summed = dl1_total + dl1_total_shared` を作る。直後のL1f
backward (3箇所: cuBLAS入力bwd、cuBLAS重みbwd、`bias_grad_shared_l1f`) は
`dl1_total` の代わりに (wsb有効時のみ) `dl1_total_summed` を読むよう分岐する
(`let dl1_total_for_l1f = if shared_bucket {..summed..} else {..dl1_total..}`)。
「Backward 4 reverse」(選択branch自身のL1 weight/input/bias backward、
sorted機構、無変更) の直後に、共有branch自身のL1 backwardを追加する:
共有bucketは全行が同一bucketなので、選択branch側のsorted kernel
(`dense_mm_bwd_weight_bucket_tiled_l1_sorted`) や9-bucket固定accumulatorの
旧 `_tiled_l1` は使わず、l1fと全く同じパターンの単純なcuBLAS matmul
(`sgemm_xt_y_rowmajor`で重み勾配、`sgemm_fwd_rowmajor`で入力勾配、
`bias_grad_bucket`で bias勾配) で計算する。書き込み先は
`l1_w_grad`/`l1_b_grad` の共有bucket専用セル (offset
`shared_bucket_index() * l1_out * ft_out` 等) で、選択branch側が書く他
bucketのセルとは完全に独立。最後に `add_inplace(dcombined_from_l1,
dcombined_from_l1_shared)` で選択branchの `dcombined_from_l1` に共有branchの
寄与を畳み込んでから、既存の `ft_post_perspective_grad_fused[_fp16]`
(`dcombined_from_l1 + dcombined_from_l1f` の2項を読む契約、無変更) に渡す。

**`--psqt` との併用は reject**: PSQTショートカット (`psqt_diff_sparse_*`)
側にも共有branch分のforward/backwardが必要だが未実装のため、
`training.rs::validate_bucket_mode` で `wsb` + `--psqt` の組合せを明示的に
reject する。

### 実装の検証状況 (重要)

- **検証済み**: バケット合成/フォーマット層 (`bucket_mode.rs`、13
  ユニットテスト)。学習アルゴリズムの数学的正しさ (`wsb_design_check.rs`、
  本番kernelの CPU reference を使った forward一致 + 有限差分による
  backward勾配検証、2テスト)。全 cuBLAS 呼び出しの引数 (m/n/k/A/B/C の
  対応) は `trainer_common.rs` の `sgemm_*_rowmajor` の doc に書かれた
  厳密な数式と1つずつ照合して確認した。全 kernel 呼び出しの引数の個数・順序は
  同じ kernel の既存呼び出し箇所と1つずつ比較して確認した (`Option`
  field は本ADR作成時に grep で「宣言した全フィールドが使われ、使った
  全フィールドが宣言されている」ことを機械的に確認済み)。
- **未検証 (できない)**: 本セッションには `cuda-oxide` (nightly rustc +
  独自 PTX codegen backend) の実行環境が無く、`bins/nnue_train`
  そのもののコンパイルは一度も通していない。新規kernel3個 (`average_inplace`
  等) が実際にPTXへ正しくlowerされるか、`GpuWorkspace`の新規フィールド追加が
  既存コードとメモリレイアウト的に整合するか、cuBLASのポインタオフセット
  計算 (`grad_base.add(shared_idx * l1_out * ft_out)` 等) が実際のバッファ
  サイズと整合するか、は静的な読み合わせでのみ確認しており、実機・
  コンパイラによる検証を経ていない。

**マージ前に必ず行うこと**: GPU環境で `local-ci.sh` を通す。可能であれば
`bins/nnue_train/src/tests/gpu_cpu_equivalence_tests.rs` に `wsb` 用の
end-to-endケース (`--bucket-mode k3k3_wsb` 等で数stepのforward/backwardを
実行し、`net_output`が2branchの単純平均と一致すること、
`--bucket-mode k3k3` (wsb無し) の既存学習結果に一切変化が無いこと (回帰
確認) の両方) を追加してから実運用の学習に使うこと。
