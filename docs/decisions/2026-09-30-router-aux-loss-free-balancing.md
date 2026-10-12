# router の負荷分散方式の選択 (`--router-balance-mode`)

- **Status**: Accepted

## Context

`--bucket-mode ...routerkpabs<N>` の router は、あるバケットの評価関数が弱い間
そのバケットへ dispatch されにくくなり、学習データが減ってさらに弱くなる偏り
(expert collapse) を起こしうる。これまでは Switch Transformer 式の補助損失
`L_balance = N · Σ_i f_i · P_i` (`--router-balance-weight`) を router の損失に
足して抑えてきた。

補助損失は勾配として本来の目的 (バケット選択の精度) と干渉する。これを避ける
方式として Auxiliary-Loss-Free Load Balancing (Wang et al. 2024、DeepSeek-V3 で
採用) がある。補助損失を使わず、バケットごとの bias を dispatch (argmax) の
選択にだけ加え、bias を使用率に応じて更新する。

## Decision

`--router-balance-mode` で 2 方式を切り替える。

- **`aux-loss`** (既定、従来動作): 補助損失。`--router-balance-weight` /
  `--router-balance-weight-gamma` / `--router-balance-weight-min` を使う。bias は
  更新しない。
- **`bias-free`**: 補助損失を使わない。router の M step ごとに、この batch の
  dispatch 使用率 `f_k` から `bias[k] += u * sign(1/N - f_k)` を適用する
  (`u` = `--router-bias-update-rate`)。更新後に平均を引いて中心化する
  (argmax は変わらず、bias が一方向にドリフトしない)。

bias は dispatch の argmax `argmax_k (logits[k] + bias[k])` にだけ影響する。
softmax・cross entropy・期待損失の勾配は bias を含まない `logits` から計算し、
bias 自体は勾配を受け取らない。dataloader のバケット割当と、使用率 `f` の計測は
どちらも bias 込みの dispatch を使う。

`--router-balance-mode bias-free` は `routerkpabs<N>` のみ対応。`routerft<R>ft<R>`
で指定するとエラーにする。

### 推論側 (YaneuraOu) への受け渡し

推論側の `RouterKPAbs` は bias 項を持たず、重みの和の argmax だけでバケットを選ぶ。
読み込み側を変えないため、export 時に bias を重みへ畳み込む。

推論側が 1 局面で足す重みの個数は常に一定である。KP-absolute の active index は
玉以外の駒 1 枚 (盤上・持ち駒を問わない) につき先手視点・後手視点の 2 個で、玉以外の
駒は必ず 38 枚あるため `38 * 2 = 76` 個になる
(`ROUTER_ACTIVE_INDICES_PER_POSITION`)。そこでバケット `k` の全 index の
`w[idx][k]` へ `bias[k] / 76` を加算すると、重み和は `logits[k] + bias[k]` に
なる。量子化 (Q16.16) は加算後の値に対して行う。先に量子化してから足すと丸め誤差が
76 倍に積もるため。

bias が全 0 (`aux-loss`) なら出力は従来と同一。

### 保存形式

router の raw 保存形式 (`RouterKPAbsWeights::write_to`、`--router-resume` の sidecar
を含む) は `w` の後ろに bias (f64 × N) を続ける新しい magic で書く。読み込みは
bias の無い旧 magic も受け付け、その場合 bias は 0 とする。

`--router-balance-mode aux-loss` で bias 付きの sidecar を resume した場合、bias は
更新されないがそのまま dispatch に使われ、export にも畳み込まれる。学習時の
dispatch と推論時のバケット選択が常に一致することを優先した。

## Consequences

- `--router-bias-update-rate` は router の logit 単位のステップ幅で、大きいほど
  早く均等化するが dispatch が揺れる。既定値 `0.001` は初期値の重み和 (標準偏差
  0.6 程度) に対して控えめな値で、データ・`--num-buckets` に応じた調整が要る。
- GPU-resident 学習では bias は host が保持する (数 KB 未満)。dataloader が読む
  process-global への反映は重みと同じ `--router-refresh-interval` / checkpoint
  保存前の同期で行う。
- 畳み込みは「1 局面で常に同じ個数の重みを足す」ことが前提。玉が盤上に無い局面
  (active index が 0 個) は学習データに存在しない前提で扱っている。
