# layerstack_v3 GPU trainer (`bins/nnue_train layerstack-v3`)

## 背景

`2026-07-14-layerstack-v3-per-bucket-dims.md` で `layerstack_v3` の `.bin`
フォーマット (`nnue_format::layerstack_v3_weights`) を実装した際、GPU trainer
(実際に学習を回す部分) は「per-bucket 別 kernel launch にする設計が必要」と
スコープ外にしていた。本 decision はその GPU trainer (`nnue-train layerstack-v3`
サブコマンド) を実装した記録。

## 設計

`bins/nnue_train/src/trainer_layerstack_v3.rs` の module doc 参照。要点:

- FT (feature transformer) は bucket 非依存なので `GpuTrainer` ("V2") と完全に
  同一のコード (kernel 呼び出し列) を流用する。
- L1/L2/L3 の per-bucket dense 計算は、V2 の「9 bucket まとめて 1 kernel
  launch (register fan-out)」設計をやめ、**bucket ごとに独立した dense stack**
  として扱う: bucket ごとに専用 (0-indexed, `batch_size` 分確保) の
  weight/activation buffer を持ち、cuBLAS `Sgemm` (`SimpleGpuTrainer` の
  L1f 相当と同じパターン) で計算する。
- `combined`/`combined_sorted` (ft_out 幅、bucket 間で共通の大きい buffer) は
  bucket ごとに複製せず、1 本の共有 buffer + 生ポインタ offset (cuBLAS の
  `a_ptr`/`c_ptr` に `cu_deviceptr() + byte_offset` を渡す) でメモリ増大を
  避ける。
- bucket ごとの 0-indexed buffer と、共有の original-order buffer
  (`net_output`/`dy_net_output`) の橋渡しに、新設の 2 kernel
  (`gather_by_perm_offset` / `scatter_by_perm_offset`,
  `kernels/layerstack.rs`) を使う。V2 の `permute_rows_f32` /
  `inverse_permute_rows_f32` と同じ形だが、`perm` 配列の読み出し開始位置
  (`perm_offset`) を追加引数に取る点だけが違う。

## スコープ外 (V1 実装で意図的に外したもの)

- L1f (shared factorized L1) / PSQT shortcut / threat feature / FT
  factorizer / FP16 fast path (`--ft-fp16` 等)。FP32 のみ (TF32 は
  `--tf32` で opt-in)。
- WRM (win-rate-model) loss。`--win-rate-model` は明示 error にした
  (`loss_wrm` / `wrm_weight_sum` kernel の正確な引数列をこの codebase の
  実装で確認しないまま配線するリスクを避けるため)。既定の sigmoid-MSE
  loss のみ対応。
- `--resume` / `--init-from`。V1 は常に xavier-init からの新規学習のみ。
  resume 用の raw checkpoint 書き出し自体 (`save_raw_checkpoint`) は独自の
  簡易バイナリ形式で実装した (`ckpt.rs` の generic 形式とは非互換) が、
  読み込み側は未実装。
- held-out validation (`--test-data` / `--test-tail-positions`)。
  `TrainingConfig` には `test_data: None` を渡している。
- Async loss readback (`AsyncLossRing`)。`step()` は `loss_acc` を毎回
  同期 (`to_host_vec`) で読み戻す単純な実装 (V2 の 2-step-lag async
  pipeline より遅いが、実装・検証コストを下げるための意図的な単純化)。

## 検証状況 — 重要 (繰り返しになるが記録として明記する)

**本実装はこのセッションの開発サンドボックスに CUDA / GPU が無いため、
一度もビルド/実行されていない。** YaneuraOu 側 (`SFNNwoP_V3` architecture、
別 decision "2026-07-14" 付近参照ではなく、この日の C++/Python 側の変更) は
実際に `make` でビルド・リンクし USI 経由の動作確認までできたが、本 Rust
GPU trainer は cuda-oxide / cuBLAS の実 API 呼び出しを含み、それらを
このセッションの環境から検証する手段が無かった。

設計判断そのもの (cuBLAS の shape 引数と backprop 公式の対応) は手計算で
複数箇所 (L1/L2/L3 の fwd/bwd 双方) 確認済みで、既存コードの
`sgemm_fwd_rowmajor` / `sgemm_x_yt_rowmajor` / `sgemm_xt_y_rowmajor` の
実装 (`trainer_common.rs`) を直接読んで対応させた。多くの再利用 kernel
(`loss_wdl` / `sparse_ft_forward` / `ft_post_perspective_fwd` /
`ft_post_perspective_grad_fused` / `radam_step` 等) も実際の signature を
`kernels/*.rs` から確認して合わせた。

それでも次の点は実機での確認が必須:

1. `cargo build --features gpu` が通ること (借用チェック・型不一致など、
   目視レビューでは拾いきれないミスがありうる)。
2. `gather_by_perm_offset` / `scatter_by_perm_offset` の2 kernelの動作
   (単体テストで正しさを確認すること)。
3. FT backward の inverse-index パイプライン (`build_feature_counts` /
   `prefix_sum_block_local` / `exclusive_prefix_sum_small` /
   `prefix_sum_add_block_offset` / `scatter_positions` /
   `gather_and_sum_per_feature_{overwrite,add}`) の引数を `GpuTrainer`
   から機械的に移植したが、1つ1つの grid/block 設定を再検証すること。
4. `DeviceBuffer::cu_deviceptr()` の戻り値型に対する整数オフセット演算
   (`+ byte_offset as u64`) が意図通りかの確認 (このコードベースの既存
   箇所はどこも offset 無しの直接 cast しかしておらず、本実装が最初の
   offset 付き使用箇所)。
5. 小規模データでの smoke test → loss が実際に下がることの確認 → 既存
   `layerstack` (V2, 全 bucket 同一サイズ) 学習との比較で同等の収束を
   することの確認。

## フォローアップ

- WRM loss 対応。
- resume (`--resume`) の読み込み側。
- 性能: 現状は bucket ごとに 9 回の cuBLAS 呼び出し (小さい GEMM) に
  分解しているため、V2 の「1 launch で 9 bucket 分」に比べて kernel
  launch オーバーヘッドが増える。実測してから、必要なら
  `cublasSgemmBatched` / `cublasGemmGroupedBatchedEx` へ置き換える
  (bucket ごとに shape が違うため batched といっても "grouped" API が
  要る点に注意)。
