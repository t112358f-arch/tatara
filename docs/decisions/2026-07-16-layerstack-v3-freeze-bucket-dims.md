# layerstack_v3 training via "bucket dim freeze" (既存 uniform trainer 再利用)

## 背景・方針転換

2026-07-15 時点でいったん「bucket ごとに専用 buffer + cuBLAS で独立に
L1/L2/L3 を計算する」新規 trainer (`trainer_layerstack_v3.rs`、新規 kernel
`gather_by_perm_offset`/`scatter_by_perm_offset` 追加) を実装したが、
**新しいカーネルを追加せず、既存の `layerstack` (uniform bucket size)
trainer をそのまま使う**方針に変更した。当該 file・kernel は削除し、
本 decision の方式に置き換えた。

## 方式: 最大サイズで学習 + 余剰パラメータを 0 に固定 + export 時に repack

1. `--l1-per-bucket`/`--l2-per-bucket` (カンマ区切り9個の自然数) を
   `nnue-train layerstack` サブコマンドに追加。指定すると:
   - 既存の `GpuTrainer` (kernel は一切変更なし) を `l1 = max(l1_per_bucket)`,
     `l2 = max(l2_per_bucket)` (全 bucket 共通、既存 API のまま) で構築する。
   - 学習開始前に一度、`GpuTrainer::apply_bucket_dim_freeze(l1_out_per_bucket,
     l2_out_per_bucket)` を呼び、bucket ごとに「実際に使わない」余剰の
     L1 行・L2 行/列を 0 に初期化する。
   - 併せて `GpuTrainer::set_freeze_l1f(true)` 相当 (`apply_bucket_dim_freeze`
     内部で呼ぶ) で、L1f (shared factorized L1) の寄与を毎 step 明示的に
     0 へ再上書きする ([`GpuTrainer::zero_l1f`])。
2. 学習は普通に (kernel 変更なしで) 進む。
3. checkpoint 書き出し (`.bin`) だけ、`GpuTrainer::to_layerstack_weights()`
   (padding された uniform 形式) → 新設
   `nnue_format::layerstack_v3_weights::repack_from_uniform(..)` (host-side、
   pure Rust の repack。GPU 計算なし) → `LayerStackV3Weights::save_quantised`
   という経路に差し替える (`BucketDimFreezeExport` という
   `TrainerBackend` の薄い decorator で実現、`train_step`/`validate_step`/
   `save_resume_checkpoint` は `GpuTrainer` の既存実装にそのまま委譲)。
   resume 用の raw checkpoint (`--resume`) は引き続き padding 済みの uniform
   形式のまま (v1 では `--resume`/`--init-from`/`--eval-only` と
   `--l1-per-bucket` の併用は非対応、明示 error)。

## 数学的な正しさ (なぜ 0 固定が「動かない」のか)

CReLU / SqrCReLU は「0 入力 → 0 出力」の乗算的なゲートなので、**ある重みが
厳密に 0 で、かつそこから先の経路が全部その重みの出力を読む箇所も 0 なら、
その重みへ流れ込む gradient も厳密に 0 になる** という自己無矛盾な不動点が
ある (IEEE754 の `0 * finite = 0`, `0 + 0 = 0` は丸め誤差なしに厳密に成り立つ
ため、一度この不動点に入れば数値誤差でドリフトしない)。

具体的に 0 固定する集合 (bucket `g`、`l1_max`/`l2_max` は padding 後の
共通サイズ、`l1_eff_g = l1_out_per_bucket[g] - 1`, `l1_eff_max = l1_max - 1`):

- `l1_w[g, row, :]` / `l1_b[g, row]` (`row` は excess main 行、skip 行
  (`l1_max - 1`) は対象外・常に real)。
- `l2_w[g, r, c]`: `r >= l2_out_per_bucket[g]` (excess 出力行、**全 c**) の
  OR `c` が上記 excess L1 main 行に対応する入力列 (sqr 半分・main 半分
  両方、**全 r**) の **和集合**。
- `l2_b[g, r]` (`r` は excess 出力行)。
- `l3_w[g, r]` (`r` は excess L2 出力位置)。

「L2 の列方向 0 固定は excess 行だけでなく **全出力行** に対して必要」な点が
ポイント: L1 の excess main 行の backward gradient (`dl1_main[:,excess]`) は
`Σ_out dl2_pre[:,out] * L2_w[out,excess_col]` で決まるので、excess でない
出力行 `out` の `L2_w` もその列については 0 にしておかないと、L1 の excess
行に非 0 gradient が流れ込み、0 で固定したはずの重みが動いてしまう。

**L1f (shared, bucket 非依存の加算項) は自己無矛盾な不動点を作れない**:
`l1_total = l1_w[bucket]の寄与 + l1f の寄与` という加算分岐は、CReLU 等の
乗算ゲートと違って「0 入力 → 0 gradient」が成り立たない (加算は微分すると
定数 1 が両方の枝に伝わるだけなので、一方の枝が 0 でも他方の gradient には
無関係)。そのため L1f だけは 0 初期化しても学習が進むにつれて 0 から
ずれていき、それが L1 の "excess" 行にも (bucket 非依存に) 混入してしまい、
上記の不動点全体を壊す。よって L1f は自己無矛盾な固定では止められず、
`GpuTrainer::step()` の末尾で毎 step 明示的に 0 へ再上書きする方式にした
(計算自体は無駄になるが、正しさを保つには最も単純で確実)。

PSQT / threat feature は同様の理由 (per-bucket freeze との整合を個別に
検証していない) で `--l1-per-bucket` と併用不可にした (CLI で明示 error)。

## repack (export) の変換

`repack_from_uniform` (`crates/nnue-format/src/layerstack_v3_weights.rs`):

- L1: bucket `g` の実行の main 行 `[0, l1_eff_g)` (そのまま) + skip 行
  (padded 側の `l1_max - 1` 番目を、小さい側の最終行 `l1_eff_g` 番目に
  詰め直す) → `l1_out_per_bucket[g]` 行の行列にする。
- L2: 出力行 `[0, l2_out_per_bucket[g])`、入力列は sqr 半分・main 半分
  それぞれの prefix `[0, l1_eff_g)` だけを取る (concat 構造なので prefix
  切り出しだけで済む、並べ替え不要)。
- L3: `[0, l2_out_per_bucket[g])` の prefix。
- L1f は (`freeze_l1f` が正しく効いていれば厳密 0 のはずだが) 念のため
  `l1_w[bucket][out][in] + l1f_w[in][out]` の式で常に加算してから
  抽出する (0 を足すだけなので無害、防御的)。
- PSQT が `Some` なら error (layerstack_v3 は PSQT 非対応)。

## 検証状況

**GPU 環境が無いこのセッションでは、`apply_bucket_dim_freeze` /
`zero_l1f` / `repack_from_uniform` を含め、ビルドも実行もできていない。**
kernel を新規追加していないため、既存の (すでに動作実績のある)
`dense_mm_*_bucket` / `radam_step` / `ranger_lookahead_lerp` 等はそのまま
なので前回 (per-bucket cuBLAS trainer) 案より検証負荷は小さいはずだが、
以下は実機での確認が必須:

1. `apply_bucket_dim_freeze` の host-side index 計算 (`l1_w`/`l2_w` の
   flat index 変換) に off-by-one が無いか。
2. 0 固定した重みが実際に学習を通じて厳密 0 のまま留まるか (数 superbatch
   学習後に `to_layerstack_weights()` して該当 range が全部 0 であることを
   assert する統合テストを書いて確認すること)。
3. `repack_from_uniform` が `apply_bucket_dim_freeze` で 0 固定した range
   と exact に対応した prefix/skip-row 変換になっているか (round-trip
   テスト: 0 固定した uniform weights を手で作り、repack して
   `LayerStackV3Weights::save_quantised` → yaneuraou 側 `SFNNwoP_V3` で
   読み込んで、同じ入力に対して uniform 版 (`layerstack` 通常 export) と
   同じ評価値になることを確認するのが理想)。
4. `--num-buckets` を 9 以外にした場合の挙動 (`apply_bucket_dim_freeze`
   は `num_buckets == 9` を assert しており、それ以外は reject する)。

## resume / init-from 対応

`--resume` / `--init-from` はどちらも `--l1-per-bucket`/`--l2-per-bucket` と
併用できる (`--eval-only` のみ未対応のまま)。適用順序が重要:

- **fresh** (どちらも無し): `GpuTrainer::new` の直後に
  `apply_bucket_dim_freeze` を呼ぶ (今まで通り)。
- **`--init-from`**: 読み込んだ (量子化 `.bin` の) weight を
  `trainer.load_layerstack_weights(&weights)` で trainer に上書きした
  **後** に `apply_bucket_dim_freeze` を呼ぶ。先に呼ぶと
  `load_layerstack_weights` の上書きで freeze (0 固定) が消えてしまう。
  読み込む `.bin` の `--l1`/`--l2` は (per-bucket ではなく) `--l1-per-bucket`/
  `--l2-per-bucket` の最大値 (padding 済みサイズ) と一致している必要がある。
- **`--resume`**: raw checkpoint (`GpuTrainer::save_raw_checkpoint`) には
  重みと Ranger optimizer state (`m`/`v`/`slow`) がそのまま保存されている。
  freeze が正しく機能していれば、保存された重みは既に余剰 dims が 0 の
  はずなので、**`apply_bucket_dim_freeze` は呼び直さない** (呼ぶと
  `m`/`v`/`slow` を丸ごと 0 にリセットしてしまい、resume の意味が無くなる)。
  `freeze_l1f` は raw checkpoint に含まれない in-memory only の flag なので、
  `GpuTrainer::set_freeze_l1f(true)` で明示的に立て直すだけでよい。
  **注意**: resume 時は元の学習と同じ `--l1-per-bucket`/`--l2-per-bucket`
  を渡すこと (checkpoint 自体にはどの bucket が何次元だったかの記録が
  無いため、export 時の repack サイズは resume 実行時の CLI 引数がそのまま
  使われる。異なる値を渡すと、まだ 0 のはずの位置を「real」として export
  してしまったり、逆に real な位置を切り捨てたりする)。

## 検証状況 (追記)

上記 resume/init-from 分岐もビルドは通ったが (ユーザー環境で確認済み)、
実際に「学習を中断 → resume → 学習再開 → export」の一連の流れを実機で
確認したわけではない。特に以下は要確認:

5. resume 直後、`freeze_l1f` を再度 `true` にしただけで (checkpoint の
   重み自体は既に 0 のはずなので) 追加の drift が起きないこと。
6. `--init-from` で読み込んだ `.bin` が (通常の uniform layerstack として
   学習された、frozen でない) 任意の網であっても、`apply_bucket_dim_freeze`
   一発で正しく freeze された状態に変換できること (これは fresh init と
   同じロジックなので、3. の検証で概ねカバーされるはず)。

## フォローアップ

- `apply_bucket_dim_freeze` の正しさを固定する unit test
  (`crates/gpu-kernels` 相当の CPU reference で host-side index 計算だけ
  切り出してテストするのが現実的、GPU 無しで検証できる部分)。
- 学習の無駄 (L1f を毎 step 計算してから捨てる) を減らしたければ、
  forward/backward に `skip_l1f: bool` 分岐を追加する案があるが、kernel
  呼び出し自体は変えず「呼ぶかどうか」の条件分岐だけなので、必要になれば
  改めて decision を切ってから着手する。
- `--eval-only` との併用対応 (現状明示 error)。
- resume 時に `--l1-per-bucket`/`--l2-per-bucket` の一致を checkpoint 側で
  検証する仕組み (現状は CLI 引数を都度正しく揃えることをユーザーに要求
  しているだけで、機械的な整合性チェックが無い)。
