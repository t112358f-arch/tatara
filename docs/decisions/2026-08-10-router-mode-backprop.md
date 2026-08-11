# router bucket mode の M step 選択 (`--router-mode`)

- **Status**: Accepted

## Context

`--bucket-mode router` の router (`shogi_features::router_kpabs::RouterKPAbs`)
は、GPU 側 LayerStack 学習ループが per-position の `bucket_idx` を batch 開始前
に host 側で確定させる設計であるため、CUDA kernel 内で router の勾配を直接
引き戻す経路を持たない。そのため router は EM 系の手続き (E step: N 個の
固定 bucket で forward-only を回し bucket ごとの誤差を求める、M step: その
誤差から作った oracle ターゲット分布との cross entropy を router の重みに
backprop する) で学習してきた (`RouterKPAbsWeights::train_oracle_batch`)。
`--top-k` で hard-EM (`K=1`) / soft-EM (`K=num_buckets`) / Top-K Hard Routing
(中間) を切り替えられる。

この EM 手続きは目的関数 (実際の評価誤差) を router の重みについて直接
微分しているわけではなく、「誤差最小 (または上位 K 個) の bucket を教師
ラベルとして router を分類器として fit する」代理タスクを解いている。LLM の
Mixture-of-Experts で一般的な、ゲートネットワークの重みを本体と同じ計算
グラフの一部として実際の task loss から誤差逆伝播で学習する方式とは異なる。

## Decision

E step (N-way forward-only sweep で bucket ごとの誤差 `errs[k]` を求める部分)
は共有したまま、M step を `--router-mode` で選べるようにした:

- **`hard-em`** (既定、従来動作): `oracle_targets_from_errors` (`--top-k`
  依存) で oracle ターゲット分布を作り、そのラベルとの cross entropy を
  `RouterKPAbsWeights::train_oracle_batch` で backprop する。
- **`backprop`**: oracle ターゲット分布を経由せず、router 自身の softmax
  分布 `P = softmax(logits)` の下での期待損失
  `L = Σ_k P_k · errs[k]` を router の重みについて直接微分する
  (`RouterKPAbsWeights::train_backprop_batch`)。softmax の jacobian から
  `dL/dlogits_j = P_j · (errs[j] − L)` (期待値の softmax 微分の標準形)。
  `--top-k` / `--top-k-reduction-interval` はこのモードでは無視する
  (oracle ターゲットという概念自体が無いため)。

両モードとも、負荷分散補助損失 (Switch Transformer 式
`L_balance = N · Σ_i f_i · P_i`、`--router-balance-weight`) は同じ式を加算
する。E step (N 回の forward-only sweep) のコスト・`--router-refresh-interval`
による頻度制御・Adam optimizer state・checkpoint (`--router-resume`) 形式は
モードに依らず共通。

`--router-mode backprop` を選んだときに `--top-k` に既定と異なる値を渡しても
（無視される値のため）弾かない — `--top-k` の `[1, --num-buckets]` 範囲検証は
`hard-em` のときのみ行う。

## Update (2026-08-11): `backprop` でも `--top-k` / `--top-k-reduction-interval` を使う

上記の初版では `backprop` モードは常に全 `num_buckets` 個の softmax を使い、
`--top-k` は無視していた。実際には Top-K MoE routing (Switch Transformer の
Top-1、Mixtral の Top-2 等) は「上位 K 個の expert だけを選び、その K 個に
softmax を制限してから backprop する」のが標準的な形であり、`backprop` に
`--top-k` を使わせない方が実際の LLM MoE から乖離していた。そこで
`RouterKPAbsWeights::train_backprop_batch` を、`hard-EM` と同じ誤差昇順の
上位 `top_k` 個選択 (`top_k_error_ranked_indices`、`oracle_targets_from_errors`
と共有) で候補集合 `S` を絞り、`S` の中だけで renormalize した router 自身の
分布 `Q` (`Q_k = P_k / Σ_{j∈S} P_j`) による期待損失を微分するよう変更した:

```text
L = Σ_{k∈S} Q_k · errs[k]
d L / d logits_j = Q_j · (errs[j] − L)   (j∈S)
d L / d logits_j = 0                     (j∉S、この loss 項からは勾配なし)
```

`top_k = num_buckets` なら `S` = 全 bucket で従来の (制限なし) 版と一致する。
`top_k = 1` は `S` が 1 点になり `Q` が one-hot に退化するため
`d L/d logits ≡ 0` — softmax の支持集合が 1 点しかないと自由度が無く、勾配が
恒等的に消える。これは `backprop` モードの `--top-k` の既定値 (`1`、`hard-em`
との統一のため) では学習が完全に止まってしまうことを意味する。

これに対応するため `--top-k-min` を追加した。`--top-k-reduction-interval` に
よる anneal は (`hard-em` / `backprop` 共通で) `top_k` を `1` ずつ減らしていく
が、今までは下限が `1` に固定されていた。`--top-k-min` (既定 `1`、`[1,
--top-k]` の範囲) でその下限を明示的に指定できるようにし、`backprop` で
anneal を使う場合は `2` 以上を指定することで学習停止を避けられるようにした
(`--top-k-min` が `2` 未満で `backprop` + anneal が有効な場合は起動時に
warning を出す)。

`--top-k` / `--top-k-min` の範囲検証 (`validate_router_top_k`) は `hard-em` /
`backprop` 共通で常に行うようにした (初版の「`backprop` では検証しない」は
`--top-k` が両モードで意味を持つようになったため撤回)。負荷分散補助損失
(`--router-balance-weight`) は引き続き `top_k` 制限とは無関係に全
`num_buckets` 個の使用率に対してかける (Top-K で選ばれなかった bucket も
全体の load balance には数える)。
