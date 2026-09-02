# `--router-arch ft-by-ft`: 設計と実装状況

`spec.md` で要求された、Router を評価net自身の Feature Transformer (FT) に
統合する新しい Router Architecture (`ft-by-ft`) の設計確定版と、tatara /
YaneuraOu 両リポジトリでの実装状況をまとめる。

既存の `--bucket-mode router` (= `--router-arch kpabs`, default) は
**一切変更していない**。KP-absolute 疎入力上の独立な線形モデル
(`RouterKPAbsWeights` / `router_kpabs.rs`) のままで、100% 後方互換。

## 1. 確定したレイアウト

`ft_out` = `--ft-out` (片視点、pairwise 後 = 従来通り L1 に渡る次元)、
`R = sqrt(--num-buckets)` (偶数であることを要求)、
`H = ft_out / 2`、`r = R / 2` として、片視点の **activation 前**
(= accumulator の raw な) FT 出力を次の順で並べる:

```text
[ normal_a(H) | router_a(r) | normal_b(H) | router_b(r) ]
合計 = 2H + 2r = ft_out + R  (= 片視点の物理 FT 幅)
```

既存の CReLU→pairwise-multiply (`ELEMENT_WISE_MULTIPLY`) は「前半
`[0, half)` × 後半 `[half, 2*half)`」の対応 index 同士の積
(`half = (ft_out+R)/2 = H+r`) なので、この配置だと:

- `j ∈ [0, H)`: `normal_a[j] * normal_b[j]` → **採用** (L1 入力、次元は
  `ft_out` のまま不変)
- `j ∈ [H, H+r)`: `router_a[j-H] * router_b[j-H]` → **破棄**
  (STM 用 j=0 は `router_a[0]*router_b[0]` — 正常な FT 値とは無関係な
  ゴミなので、L1 には絶対に渡さない)

Router の bucket 選択は **pairwise/CReLU を経由しない、生の accumulator**
から直接読む: `s[0..R) = router_a ++ router_b` (STM)、`n[0..R) = router_a
++ router_b` (NSTM) を取り、`bucket = argmax(s) * R + argmax(n)`。

この配置の最大の利点: **既存の FT accumulator 差分更新 (incremental
update) が一切変更不要** — 単に幅が `ft_out` から `ft_out + R` に増える
だけで、既存コードは列の意味を知らないまま正しく動作する。

具体例 (`spec.md` の例、`--num-buckets 64`, `--ft-out 2296`):
`R=8, H=1148, r=4`, 物理 FT 幅 `2304`。`normal_a=[0,1148)`,
`router_a=[1148,1152)`, `normal_b=[1152,2300)`, `router_b=[2300,2304)`。
pairwise 後、片視点 1152 個のうち 1148 個を採用、4 個 (router 由来) を破棄
— 両視点で 8 個破棄 (`spec.md` 15 節の図と一致)。

## 2. Router の loss / 勾配

`P(i,j) = softmax(s)[i] * softmax(n)[j]` という設計そのものが、任意の目的
分布 `T(i,j)` に対する cross entropy を

```text
L = -sum_i marg_stm(i) log softmax(s)[i] - sum_j marg_nstm(j) log softmax(n)[j]
```

(`marg_stm(i)=sum_j T(i,j)`, `marg_nstm(j)=sum_i T(i,j)`) に厳密分解する。
つまり **STM 側・NSTM 側で完全に独立な 2 本の softmax cross entropy** に
なり、勾配は通常の softmax-CE と同じ形:

```text
dL/ds[i] = softmax(s)[i] - marg_stm(i)
dL/dn[j] = softmax(n)[j] - marg_nstm(j)
```

joint な `R*R` 分布を陽に持ち回る必要がない。hard-EM oracle 由来の
one-hot target ならこれは自明な特殊ケース (marginal がそのまま one-hot)。

実装・検証済み: `bins/nnue_train/src/router_ftbyft.rs::loss` module
(softmax, argmax, `hard_target_grad`, `soft_target_grad`)。
`hard_target_grad` は有限差分と突き合わせて数値検証済み
(`loss_hard_target_grad_matches_finite_difference` test)。

## 3. 学習パイプラインへの統合(続報)

section 1 の設計に加え、spec.md 16節の指示 (「保存時にのみ結合すればよい」)
に従い、学習中は評価net本体の FT と router を **完全に分離** した設計で
実装した:

- `crates/shogi-features/src/router_ftbyft.rs`: `FtByFtLayout` (レイアウト計算・検証) /
  `loss` module (softmax・argmax・hard/soft target 勾配、有限差分で数値検証済み) /
  `RouterFtByFtWeights` (`(ft_in, R)` 重み行列、評価net本体の FT と**同じ
  sparse 入力** `Batch::stm_indices`/`nstm_indices` 上で forward/backward) /
  `RouterFtByFtAdamState` / `RouterFtByFt` (process-global、`RouterKPAbs` と
  対をなす API) / `combine_ft_by_ft_columns` (保存直前にのみ呼ぶ列結合関数)。
- `crates/nnue-train/src/trainer.rs`: `RouterTrainingConfig` に `arch:
  RouterArch` を追加。E step (bucket 0..N の N 通り forward で誤差を求める
  oracle sweep) は kpabs と完全に共用 (bucket の "意味" に依存しないロジック
  のため)。M step だけを `RouterArch::FtByFt` 用に分岐し、
  `RouterFtByFtWeights::train_oracle_batch` (hard-EM のみ対応、backprop
  mode・balance loss は今回未対応、`training.rs` が明示的に reject する) を
  呼ぶ。
- `bins/net_to_yo`: `--router-ft-by-ft <path>` を追加。`RouterFtByFtWeights`
  の sidecar (`{net_id}-{sb}.router-ftbyft.ckpt`) を読み、入力 `.bin` の
  `(num_buckets, ft_out)` から `FtByFtLayout` を再構築した上で
  `nnue_format::save_yaneuraou_combined` を呼ぶ。
- `crates/nnue-format/src/yaneuraou.rs`: `save_yaneuraou_combined` を新設
  (既存 `save_yaneuraou` は無変更)。評価net本体の `LayerStackWeights`
  (nominal `ft_out` のまま、L1/L2/L3 は一切変更不要) と router 重みを
  `combine_ft_by_ft_columns` で結合してから、`arch_string`/`ft_hash` は
  **nominal** `ft_out` を使い、edition suffix だけ `_ROUTER_FT{R}FT{R}`
  にする (`ft_by_ft_arch_string`)。FT 重み/bias ブロックだけが物理的に
  `ft_out+R` 幅になる。

これで CLI (`nnue_train --bucket-mode router --router-arch ft-by-ft`) から
学習を実行でき、`net_to_yo --router-ft-by-ft <ckpt>` で YaneuraOu が読める
単一ファイルに変換できる状態になった。`--router-balance-weight` (Switch
Transformer 式の負荷分散補助損失) も STM/NSTM 側それぞれの `R`-way softmax
に独立に適用する形で対応済み (下記 5 章)。未対応: `--router-resume`
(ft-by-ft)、`--router-mode backprop` (ft-by-ft)、GPU-resident 学習 (現状
CPU のみ、kpabs の CPU fallback と同じ速度感)。

## 4. Balance loss (`--router-balance-weight`) の ft-by-ft 対応

`RouterKPAbsWeights::train_oracle_batch` の Switch Transformer 式負荷分散
補助損失を、STM 側・NSTM 側それぞれの `R`-way softmax に**独立に**適用する
形で `RouterFtByFtWeights::train_oracle_batch` にも実装した:

```text
L_balance_stm  = R · Σ_i f_stm_i  · P_stm_i
L_balance_nstm = R · Σ_j f_nstm_j · P_nstm_j
L_balance = (L_balance_stm + L_balance_nstm) / 2   (報告値、`RouterFtByFtTrainStats::balance_loss`)
```

`f_stm`/`f_nstm` は batch 内での router 自身の argmax (dispatch) 分布
(stop-gradient)、`P_stm`/`P_nstm` はこの batch での softmax 確率の平均
(微分可能)。

**実装時に発見したバグ**: `balance_loss` を STM/NSTM の平均として報告する
ため `/2.0` しているにもかかわらず、勾配側にはこの `/2.0` を反映し忘れて
いた (`d(L_stm/2 + L_nstm/2)/dw` であるべきところ `d(L_stm + L_nstm)/dw`
を計算していた)。これは有限差分テスト
(`router_ftbyft_balance_loss_matches_finite_difference`) で検出・修正した
— 数値勾配 (収束値 `0.35753230354...`、eps を `1e-3`〜`1e-6` で振っても
安定) に対し、修正前の解析的勾配は正確に **2 倍** の値になっており
(`-0.027257` vs 正しい `-0.013628`)、`/2.0` を掛け忘れている症状と完全に
一致した。修正後は 10 桁以上一致する。

なお、この手のテストは argmax (dispatch) が有限差分の微小摂動で反転しない
ことが前提になる — `RouterFtByFtWeights::random` のような小さい scale
(0.01) のランダム初期値だと softmax がほぼ一様になり、argmax が容易に反転
してしまい誤って「不一致」と判定されるので、テストでは明示的に分離の効い
た重み値を使っている (`router_ftbyft_balance_loss_matches_finite_difference`
のコメント参照)。

`--router-balance-weight` 以外の decay 系オプション
(`--router-balance-weight-gamma`/`--router-balance-weight-min`) も kpabs
と全く同じ CLI フラグ・decay スケジュールをそのまま共用する
(`training.rs` の `RouterTrainingConfig` 構築、`trainer.rs` の
per-superbatch decay ループは元々 arch に依存しない共通コードだった)。

`experiment.json` (`ExperimentDoc::router_history`) にも kpabs と全く同じ
`RouterHistoryEntry` 形式で記録する。`usage` フィールドは STM/NSTM の独立
性を仮定しない実測の joint 分布 (`RouterFtByFtTrainStats::joint_bucket_usage`、
`bucket_index = stm_idx * R + nstm_idx` ごとの頻度、長さ `num_buckets`) を
使う — kpabs の `bucket_usage` と全く同じ形なので、`nnue-lab` 側の
downstream 消費コードを変更する必要がない。

## 5. 修正履歴

- **SIMD 幅チェックの対象を修正** (spec.md §11 準拠): 当初
  `nnue_architecture.h` の `static_assert(kTransformedFeatureDimensions %
  kMaxSimdWidth == 0, "")` を変更せずに残していたため、`ft-by-ft` ビルド
  (例: `ft_out=2296, R=8`) で `2296 % 32 == 24 != 0` となりビルドが失敗して
  いた。spec.md §11 の通り、SIMD 幅チェックは **物理 FT 幅
  (`ft_out + R` = `kAccumulatorDimensions`)** に対して行うべきであり、`ft_out`
  単体 (`kTransformedFeatureDimensions`) に対して行うべきではない。
  `static_assert` を `kAccumulatorDimensions % kMaxSimdWidth == 0` に修正
  (`kAccumulatorDimensions` 定義後に移動)。`kRouterFtByFtR == 0` の通常ビル
  ド/kpabs では `kAccumulatorDimensions == kTransformedFeatureDimensions`
  なので完全に無変更 (regression なし、`ls9`/`k3k3` 等で再確認済み)。
  `SFNN_halfkahm2_2296-15-64-router_ft8ft8` (`2296+8=2304`, `2304%32==0`)
  で再生成し、`kAccumulatorDimensions` ベースの assert が通ることを確認済
  み。



### 完了・動作確認済み

| リポジトリ | ファイル | 内容 |
|---|---|---|
| tatara | `bins/nnue_train/src/cli.rs` | `--router-arch kpabs\|ft-by-ft` |
| tatara | `crates/shogi-features/src/router_ftbyft.rs` | レイアウト計算 (`FtByFtLayout`) + 検証 + loss/勾配 (`loss` module、有限差分検証済み) + `RouterFtByFtWeights`/Adam/process-global (`RouterFtByFt`) + `combine_ft_by_ft_columns` (保存時結合)。unit test 有り (combine のレイアウト一致、Adam 学習で loss 減少、を含む) |
| tatara | `bins/nnue_train/src/router_ftbyft.rs` | `shogi_features::router_ftbyft` の re-export (旧 `crate::router_ftbyft::...` 参照を壊さないための薄いラッパー) |
| tatara | `bins/nnue_train/src/training.rs` | `--router-arch` 検証を本線接続。`ft-by-ft` 選択時は `RouterFtByFt::init_random` でランダム初期化し、`RouterTrainingConfig{ arch: FtByFt, .. }` を構築 (kpabs と並行する分岐、後方互換) |
| tatara | `crates/nnue-train/src/trainer.rs` | `RouterTrainingConfig.arch: RouterArch` 追加。E step (oracle sweep) は kpabs と共用、M step だけ `RouterFtByFtWeights::train_oracle_batch` (CPU, hard-EM) に分岐。superbatch ごとに `{net_id}-{sb}.router-ftbyft.ckpt` を保存 |
| tatara | `crates/nnue-format/src/yaneuraou.rs` | `save_yaneuraou_combined` 新設 (既存 `save_yaneuraou` は無変更)。nominal `ft_out` で L1/L2/L3 を検証しつつ、FT だけ `combine_ft_by_ft_columns` で結合、edition suffix を `_ROUTER_FT{R}FT{R}` にする。unit test 2件 (arch string 検証、bucket 数不一致の reject) |
| tatara | `bins/net_to_yo/src/main.rs` | `--router-ft-by-ft <path>` 追加 (`--assume-kingrank9`/`--router` と排他)。sidecar から `RouterFtByFtWeights` を読み、`FtByFtLayout` を入力 `.bin` から再構築して `save_yaneuraou_combined` を呼ぶ |
| YaneuraOu | `nnue_arch_gen.py` | `_router_kpabs_{N}` (rename) / `_router_ft{R}ft{R}` (new) の検出・生成。python3 で全パターン (正常系/異常系) 実行検証済み |
| YaneuraOu | `Makefile` | 新命名からの `TANUKI_ROUTER_FIXED_BUCKETS` 抽出。単体テストで検証済み |
| YaneuraOu | `nnue_architecture.h` | `kAccumulatorDimensions` (=`kTransformedFeatureDimensions + kRouterFtByFtR`) 新設。SIMD 幅 assert もこちらに対して実施 (spec.md §11)。非 ft-by-ft では無変更 (R=0 fallback) |
| YaneuraOu | `nnue_accumulator.h` | accumulator 配列幅を `kAccumulatorDimensions` に |
| YaneuraOu | `nnue_feature_transformer.h` | VECTOR パスは `kPairSplit` (SIMD 整列済み) でループし `kKeptHalf` だけ出力先にコピー、scalar パスは直接 `kKeptHalf` まで書く。router 由来の組み合わせを自動破棄。`kKeptHalf == kPairSplit` (通常ビルド/kpabs) は `if constexpr` で従来と同一コード |
| YaneuraOu | `evaluate_nnue.cpp` | `router_ftbyft_index_for_nnue` (argmax ベース bucket 選択) を追加し、`TANUKI_ROUTER_ARCH_FTBYFT` で kpabs パスと分岐 |

### 未対応 (既知の制限、今後の課題)

1. **GPU-resident 学習**: `ft-by-ft` の router M step は CPU 実装のみ
   (`RouterKPAbs` の CPU fallback と同じ設計・同程度の速度感)。kpabs の
   `bins/nnue_train::router_gpu` に相当する GPU-resident 実装は無い。
2. `--router-resume` (ft-by-ft router の resume) は未実装 (`training.rs` が
   明示的に reject する)。
3. `--router-mode backprop` は `ft-by-ft` では未対応 (`training.rs` が明示
   的に reject する)。`hard-em` (default) の oracle-target M step のみ動作
   する。`--router-balance-weight` は対応済み (4章)。
4. `net_to_yo --router-ft-by-ft` は sidecar の `(ft_in, R)` が入力 `.bin`
   の `(feature_set, num_buckets, ft_out)` と整合するかだけを検証する —
   同じ学習 run の成果物同士を渡す前提 (別 run の sidecar を取り違えても
   検出できるとは限らない次元一致以外の不整合がありうる)。

### コンパイル未検証

本サンドボックス環境に Rust ツールチェイン (`cargo`/`rustc`) が
インストールできず (ネットワークで `crates.io` 等には到達できるが
コンパイラ本体が無い)、tatara 側の変更は **コンパイル未確認**。
YaneuraOu 側は `nnue_arch_gen.py` の実行と `Makefile` ロジックの抽出部分は
実行確認したが、C++ 本体のフルビルドはしていない (`g++` はあるが依存関係
込みのビルドは本セッションでは実施していない)。マージ前に必ず
`cargo build --workspace` および対象 edition での YaneuraOu ビルド・
`unit_test`/`gensfen` 等での動作確認を行うこと。
