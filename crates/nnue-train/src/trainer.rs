//! Training-loop driver — host-side superbatch loop for the NNUE trainer。
//!
//! GPU 非依存の trait ([`TrainerBackend`]) 越しに 1 batch 分の forward / backward
//! / optimizer step を呼び出す superbatch loop を提供する。`bins/nnue_train::
//! GpuTrainer` が `TrainerBackend` を impl し、本 module の [`run`] がそれを
//! 駆動する (`nnue-train` crate は `gpu-runtime` に依存せず、kernel launch は
//! bin 側に置く設計)。
//!
//! ## ループ構造
//!
//! ```text
//! for sb in start_superbatch..=end_superbatch:
//!     for batch_idx in 0..batches_per_superbatch:
//!         lr  = lr_scheduler.lr(batch_idx, sb)
//!         wdl = wdl_scheduler.blend(batch_idx, sb, end_superbatch)
//!         fill Batch + per-position bucket from the PSV stream
//!           (EOF → 同 file を開き直す = 次 epoch)
//!         loss += backend.train_step(batch, buckets, lr, wdl, loss_kind)
//!     report(sb, loss / positions, pos/s, ETA)
//!     if sb % save_rate == 0 || sb == end_superbatch:
//!         backend.save_checkpoint("{output_dir}/{net_id}-{sb}.bin")          # 量子化 (推論用)
//!         backend.save_resume_checkpoint("{output_dir}/{net_id}-{sb}.ckpt", sb, run_id, lr_horizon)  # raw f32 + optimizer state (resume 用)
//!         if keep_raw_checkpoints == Some(n): 直近 n 個より古い *.ckpt を削除
//! ```
//!
//! `start_superbatch != 1` で呼ぶと resume になる: lr/wdl scheduler は superbatch
//! index 駆動 (`StepLR` は `(sb-1)/step` を使う) なので `start_superbatch` を
//! 渡せば lr が自動で正しい値に戻る。weight + optimizer state の復元自体は
//! backend 側で別途行う必要がある (`bins/nnue_train --resume` 経路)。
//!
//! ## per-position output bucket
//!
//! `ShogiProgressKPAbs::bucket` が `floor(sigmoid(Σ w·x) * num_buckets)` を
//! `0..num_buckets-1` に clamp する (LayerStack `--num-buckets` で N を選ぶ、
//! [`TrainingConfig::num_buckets`])。`progress.bin` 未指定時は重み 0 で全局面
//! が `floor(0.5 * N) = N/2` (N=8/9 ともに bucket 4) に collapse する。
//!
//! ## score-drop-abs の近似
//!
//! `--score-drop-abs t` は本来「`|score| >= t` の position の per-position
//! loss weight を 0 にする」semantics。本実装は **batch に push しない (skip)**
//! で近似する。loss/gradient へ寄与しない点は同じだが、batch の構成 (slot
//! 割当・順序) は厳密一致しない。
//!
//! ## 決定論性
//!
//! `cfg.threads >= 2` のとき [`BucketedPrefetchedLoader`] は 1 epoch 内の
//! position 順序が非決定的になる (loader doc 参照)。lr / wdl は `batch_idx`
//! 駆動で順序非依存なので training には影響しない。決定論的順序が必要なら
//! `cfg.threads = 1` を指定する。

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use shogi_features::FeatureSetSpec;
#[cfg(test)]
use shogi_features::progress_kpabs::ShogiProgressKPAbs;
use shogi_features::router_kpabs::{RouterAdamState, RouterKPAbs, RouterMode};

use crate::dataloader::{Batch, BucketMode, BucketedPrefetchedLoader, PSV_RECORD_BYTES};
use crate::experiment::{ExperimentLogger, RouterHistoryEntry};
use crate::schedule::{LrScheduler, WdlScheduler};

// =============================================================================
// LossKind — どの loss kernel で 1 step を回すか
// =============================================================================

/// training step で使う loss の種別と固定パラメータ。
///
/// backend ([`TrainerBackend::train_step`]) はこの enum で分岐して対応する
/// loss kernel (`loss_wdl` / `loss_wrm`) を起動する。本 enum 自体は GPU には
/// 触らず CPU-only crate に置ける。
///
/// - [`LossKind::Sigmoid`] — plain sigmoid-MSE (`p = sigmoid(out * scale)`,
///   target = `lambda*wdl + (1-lambda)*sigmoid(score * scale)`)。net_output が
///   cp 単位 (`out ≈ cp`) で収束する。
/// - [`LossKind::Wrm`] — win-rate-model loss。prediction / target 双方に WRM 変換を
///   適用するため net_output が `out ≈ cp / nnue2score` (O(1)) で収束する。
///   `nnue2score` / `in_scaling` / `in_offset` / `target_offset` / `target_scaling` は
///   いずれも CLI から本 enum field 経由で渡る。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossKind {
    /// plain sigmoid-MSE。`scale = 1.0 / --scale` (典型値 `1/290`)。
    Sigmoid { scale: f32 },
    /// win-rate-model loss。
    ///
    /// - `nnue2score` = `--wrm-nnue2score` (既定 600)
    /// - `in_scaling` = `--wrm-in-scaling` (既定 340、prediction 側のみ)
    /// - `in_offset` = `--wrm-in-offset` (既定 270、WRM prediction sigmoid の中心)
    /// - `target_offset` = `--wrm-target-offset` (既定 270、WRM target sigmoid の中心)
    /// - `target_scaling` = `--wrm-target-scaling` (既定 380、WRM target sigmoid の
    ///   入力スケール)
    ///
    /// 以下は nnue-pytorch `calculate_sf_loss` の一般化 loss パラメータ。既定値
    /// (`pow_exp=2 / qp_asymmetry=0 / weight_boost_w1=0`) では plain な二乗誤差に
    /// 帰着し、loss kernel の bit-identical 経路を通る。
    /// - `pow_exp` = `--loss-pow-exp` (既定 2.0、誤差の冪)
    /// - `qp_asymmetry` = `--loss-qp-asymmetry` (既定 0.0、`qf > target` の過大評価
    ///   ペナルティ係数)
    /// - `weight_boost_w1` / `weight_boost_w2` = `--loss-weight-boost-w1` /
    ///   `--loss-weight-boost-w2` (既定 0.0 / 0.5、決着寄り局面の loss 重み増幅。
    ///   `w1=0` で weight≡1)
    Wrm {
        nnue2score: f32,
        in_scaling: f32,
        in_offset: f32,
        target_offset: f32,
        target_scaling: f32,
        pow_exp: f32,
        qp_asymmetry: f32,
        weight_boost_w1: f32,
        weight_boost_w2: f32,
    },
}

impl LossKind {
    /// extended (nnue-pytorch 一般化) loss 経路が必要か。既定の拡張パラメータ
    /// (`pow_exp=2 / qp_asymmetry=0 / weight_boost_w1=0`) では二乗誤差に帰着するため
    /// `false` を返し、loss kernel は bit-identical な default 経路を通る。
    pub fn wrm_extended(&self) -> bool {
        match *self {
            LossKind::Sigmoid { .. } => false,
            LossKind::Wrm {
                pow_exp,
                qp_asymmetry,
                weight_boost_w1,
                ..
            } => pow_exp != 2.0 || qp_asymmetry != 0.0 || weight_boost_w1 != 0.0,
        }
    }
}

impl LossKind {
    /// CLI / config から渡されたパラメータが loss kernel に流せる値か検証する。
    fn validate(&self) -> io::Result<()> {
        match *self {
            LossKind::Sigmoid { scale } => {
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(io::Error::other(format!(
                        "loss scale must be finite and > 0 (got {scale})"
                    )));
                }
            }
            LossKind::Wrm {
                nnue2score,
                in_scaling,
                in_offset,
                target_offset,
                target_scaling,
                pow_exp,
                qp_asymmetry,
                weight_boost_w1,
                weight_boost_w2,
            } => {
                if !nnue2score.is_finite() || nnue2score <= 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm nnue2score must be finite and > 0 (got {nnue2score})"
                    )));
                }
                if !in_scaling.is_finite() || in_scaling <= 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm in_scaling must be finite and > 0 (got {in_scaling})"
                    )));
                }
                if !in_offset.is_finite() {
                    return Err(io::Error::other(format!(
                        "wrm in_offset must be finite (got {in_offset})"
                    )));
                }
                if !target_offset.is_finite() {
                    return Err(io::Error::other(format!(
                        "wrm target_offset must be finite (got {target_offset})"
                    )));
                }
                if !target_scaling.is_finite() || target_scaling <= 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm target_scaling must be finite and > 0 (got {target_scaling})"
                    )));
                }
                // pow_exp は誤差 |err| の冪。grad は |err|^(pow_exp-1) を含むので
                // pow_exp >= 1 でないと err→0 で発散する (既定 2.0)。
                if !pow_exp.is_finite() || pow_exp < 1.0 {
                    return Err(io::Error::other(format!(
                        "wrm pow_exp must be finite and >= 1 (got {pow_exp})"
                    )));
                }
                // qp_asymmetry は過大評価 (qf > target) の追加ペナルティで >= 0。負値は
                // asym = 1 + qp_asymmetry を 1 未満にし、<= -1 では asym <= 0 で当該局面の
                // loss が負・勾配符号が反転する (最適化が逆方向に進む) ため reject する。
                if !qp_asymmetry.is_finite() || qp_asymmetry < 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm qp_asymmetry must be finite and >= 0 (got {qp_asymmetry})"
                    )));
                }
                // weight boost は決着寄り局面の重みを増幅する用途で w1/w2 >= 0。w1 < 0
                // は重み < 1 の de-emphasis で boost の意図に反し、w2 < 0 は weight base
                // が 0 (pf=0.5 や飽和) のとき `0^負 = inf` を生むため reject する。
                // w1,w2 >= 0 なら weight >= 1、Σw >= n > 0 が保証され loss_wrm の 1/Σw が
                // 安全になる。
                if !weight_boost_w1.is_finite() || weight_boost_w1 < 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm weight_boost_w1 must be finite and >= 0 (got {weight_boost_w1})"
                    )));
                }
                if !weight_boost_w2.is_finite() || weight_boost_w2 < 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm weight_boost_w2 must be finite and >= 0 (got {weight_boost_w2})"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for LossKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            LossKind::Sigmoid { scale } => write!(f, "sigmoid-MSE(scale={scale:.6})"),
            LossKind::Wrm {
                nnue2score,
                in_scaling,
                in_offset,
                target_offset,
                target_scaling,
                pow_exp,
                qp_asymmetry,
                weight_boost_w1,
                weight_boost_w2,
            } => write!(
                f,
                "wrm(nnue2score={nnue2score}, in_scaling={in_scaling}, \
                 in_offset={in_offset}, target_offset={target_offset}, \
                 target_scaling={target_scaling}, pow_exp={pow_exp}, \
                 qp_asymmetry={qp_asymmetry}, weight_boost_w1={weight_boost_w1}, \
                 weight_boost_w2={weight_boost_w2})"
            ),
        }
    }
}

// =============================================================================
// ValidationStepOutput — forward + loss のみの 1 batch 出力
// =============================================================================

/// [`TrainerBackend::validate_step`] の結果 — held-out 1 batch の forward + loss
/// 出力。weight 更新は伴わない。
#[derive(Debug, Clone)]
pub struct ValidationStepOutput {
    /// batch 全体の二乗誤差和 (`Σ err²`、position 数で割る前)。
    /// [`TrainerBackend::train_step`] の戻り値と同じ単位。
    pub sum_sq_err: f64,
    /// position ごとの net 出力スカラ (`batch.n_positions` 個、batch の position
    /// 順)。sign-agreement accuracy の計算に使う。
    pub net_output: Vec<f32>,
}

// =============================================================================
// TrainerBackend — 1 batch 分の forward → loss → backward → optimizer step
// =============================================================================

/// 1 batch 分の training step を実行する backend。
///
/// `bins/nnue_train::GpuTrainer` が impl する。本 trait を介すことで loop driver
/// を GPU 非依存に保ち (CPU-only crate に置ける)、mock backend で単体テストできる。
pub trait TrainerBackend {
    /// 1 batch 分 (forward → loss kernel → backward → optimizer step) を実行し、
    /// batch 全体で累積した二乗誤差 (`Σ err²`、まだ position 数で割っていない値)
    /// を返す。caller が報告時に position 数で割って平均 loss にする。
    ///
    /// - `batch`: HalfKA_hm sparse + score/wdl/norm (`batch.n_positions` が有効件数)
    /// - `bucket_idx`: `batch.n_positions` 個の output bucket index
    ///   (`0..num_buckets-1`、N は LayerStack `--num-buckets` で決まる runtime 値)
    /// - `lr`: learning rate (`LrScheduler` 由来)
    /// - `wdl_lambda`: WDL blend lambda (`WdlScheduler` 由来、loss kernel の `lambda`)
    /// - `loss`: どの loss kernel を起動するか (sigmoid-MSE / WRM) + 固定パラメータ
    fn train_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> io::Result<f64>;

    /// backend が前 step の loss を pipeline 経由で遅延報告する場合 (例えば pinned
    /// host ring を使った async loss readback) に、内部に滞留している未報告分の
    /// `Σ err²` を drain して返す。default 実装は `0.0` を返す (同期 readback 実装
    /// 向け)。
    ///
    /// caller (本 crate の [`run`]) は **superbatch 末尾** で呼び出すこと。背景:
    /// async pipeline では `train_step` の N 番目の呼出が `step N-1` の loss を
    /// 返し、最後の batch (`step N_per_sb - 1`) の loss は呼出されないまま残る。
    /// `flush_pending_loss` を sb 末で 1 回呼んでこの残量を sb_loss に加算することで、
    /// 1 sb の loss 集計が正確になる。
    fn flush_pending_loss(&mut self) -> io::Result<f64> {
        Ok(0.0)
    }

    /// 1 batch 分の **forward + loss のみ** を実行する (backward / optimizer step
    /// なし、weight は一切更新しない)。held-out validation 用。
    ///
    /// 引数は [`TrainerBackend::train_step`] と同じ意味 (ただし `lr` は不要)。
    /// 戻り値は batch 全体の `Σ err²` と position ごとの net 出力スカラ。
    /// caller ([`crate::validation::HeldoutSet::evaluate`]) が前者を position 数で
    /// 割って平均 loss に、後者を sign-agreement accuracy にする。
    ///
    /// `wdl_lambda` / `loss` は同 superbatch の training step と同じ値を渡すこと
    /// (test_loss を training loss と比較可能にするため)。
    fn validate_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        wdl_lambda: f32,
        loss: LossKind,
    ) -> io::Result<ValidationStepOutput>;

    /// 現在の weight を量子化 NNUE binary として `path` に書き出す (推論用
    /// artifact、`nnue-format` の `save_quantised` 相当を backend 内で実行する)。
    fn save_checkpoint(
        &mut self,
        path: &Path,
        fv_scale: Option<i32>,
        output_format: OutputFormat,
    ) -> io::Result<()>;

    /// resume 用 **raw f32 checkpoint** を `path` に書き出す。
    ///
    /// 量子化 `.bin` ([`TrainerBackend::save_checkpoint`]) と違い、全 weight
    /// group の raw f32 値に加えて optimizer state (`m` / `v` / `slow`)
    /// と step counter、および現在の `superbatch` 番号を保存する。これを
    /// `--resume` で読み戻すと optimizer state ごと学習を再開できる
    /// (`--init-from` の weight だけ注入する経路と違い、optimizer 状態も
    /// 復元される真の resume)。
    ///
    /// backend 側は device → host download → file 書き出し (`.tmp` へ書いてから
    /// `rename` で atomic に置換) を行う。本 crate は GPU 非依存なので device I/O は
    /// backend 任せ。`run_id` はこの checkpoint を書き出す学習 run の experiment.json
    /// `id` で、`*.ckpt` に producer run id として埋め込まれ、resume 時に lineage の
    /// 親参照になる (空文字列なら埋め込まない)。
    ///
    /// `lr_horizon` は LR schedule の終端 superbatch ([`LrScheduler::horizon`])。
    /// `*.ckpt` に埋め込まれ、resume 時に curve を `--superbatches` から独立に
    /// 再現するのに使う。horizon を持たない schedule では `None`。
    fn save_resume_checkpoint(
        &mut self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
        lr_horizon: Option<usize>,
    ) -> io::Result<()>;

    /// 累積 FP16 clamp event count + 累積処理要素数を device から読み出す。
    ///
    /// 戻り値 `(clamp_count, elems_processed)`:
    /// - `clamp_count`: `--ft-fp16-out` 経路で `dft_scale * grad` を `±65504`
    ///   (f16 有限域) に cap した要素数の合計 (累積)。
    /// - `elems_processed`: 同 kernel が処理した FP16 書き込み要素数の合計
    ///   (累積)。clamp ratio の分母。
    ///
    /// `--ft-fp16-out` 無し / clamp 計装を持たない backend では両方 `0`。caller
    /// ([`run`]) は sb 末で読み、前回値との差分から sb 内 ratio を出して
    /// `[fp16-clamp]` log line にする。
    fn read_fp16_clamp_count(&mut self) -> io::Result<(u64, u64)> {
        Ok((0, 0))
    }
}

// =============================================================================
// TrainingConfig
// =============================================================================

/// 推論用 checkpoint の出力形式。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// tatara LayerStack / Simple quantised binary。
    #[default]
    Tatara,
    /// YaneuraOu SFNNWithoutPsqt evaluation file。
    Yaneuraou,
}

/// 1 回の [`run`] に渡す training hyper-parameter 一式。
///
/// `BucketMode::Router` 用の hard-EM router 学習パラメータ
/// (`bins/nnue_train --router-*` CLI 由来)。
///
/// `run()` は `bucket_mode` が `Router` のときだけこれを参照する。
/// `router_kpabs::RouterKPAbs` は呼び出し前に `init_random` / `init_with_weights`
/// 済であること (`run()` はランダム初期化しない — resume との対称性のため
/// 初期化は呼び出し側の責務)。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouterTrainingConfig {
    /// router Adam の learning rate (`--router-lr`)。`RouterKPAbsWeights` は
    /// `progress.bin` と同じ f64 精度で持つため、この学習率も f64。
    pub lr: f64,
    /// router weight (`RouterKPAbsWeights::w`、bias は無い) への L2 weight
    /// decay (`--router-weight-decay`)。
    pub weight_decay: f64,
    /// 負荷分散補助損失の係数 `λ` (`--router-balance-weight`)。Switch
    /// Transformer (Fedus et al. 2021) と同じ形の補助損失
    /// (`L_balance = N · Σ_i f_i · P_i`) を cross entropy に足して router を
    /// 学習する。`0.0` で無効化 (cross entropy のみ、expert collapse の
    /// リスクあり)。詳細は [`shogi_features::router_kpabs::RouterKPAbsWeights::train_oracle_batch`]
    /// のドキュメントを参照。
    pub balance_weight: f64,
    /// router の M step の切替 (`--router-mode`)。[`RouterMode::HardEm`]
    /// (既定) は oracle ターゲット EM 手続き、[`RouterMode::Backprop`] は
    /// oracle ターゲットを経由しない通常の誤差逆伝播。`top_k` /
    /// `top_k_reduction_interval` / `top_k_min` はどちらのモードでも使う
    /// (詳細は [`shogi_features::router_kpabs`] モジュール doc、および
    /// `top_k` フィールドの doc を参照)。
    pub mode: RouterMode,
    /// hard-EM oracle refresh を何 batch ごとに行うか (`--router-refresh-interval`)。
    /// `1` で毎 batch (評価関数と "一緒に" 学習する度合いが最大、その分 forward
    /// を batch あたり 9 回余計に (bucket 数分) 実行するので低速)。`N > 1` で
    /// `batch_idx % N == 0` の batch でのみ oracle sweep + router 更新を行い、
    /// それ以外の batch は直前の router 出力をそのまま bucket 割当に使う
    /// (dataloader は毎 batch 独立に `RouterKPAbs::bucket_board` を呼ぶため、
    /// router 更新頻度を落としても bucket 割当自体は常に最新の重みを使う)。
    pub refresh_interval: usize,
    /// `lr` の superbatch ごとの減衰率 (`--router-lr-gamma`)。1 superbatch
    /// 終える度に `lr *= lr_gamma` する (`1.0` で減衰無し、従来動作)。
    pub lr_gamma: f64,
    /// `balance_weight` の superbatch ごとの減衰率 (`--router-balance-weight-gamma`)。
    /// 1 superbatch 終える度に `balance_weight *= balance_weight_gamma` する
    /// (`1.0` で減衰無し)。学習序盤は expert collapse を防ぐために強めに、
    /// 終盤は cross entropy (oracle 予測精度) を優先させたい場合に `< 1.0` を
    /// 指定する。
    pub balance_weight_gamma: f64,
    /// `balance_weight` の下限 (`--router-balance-weight-min`)。減衰させても
    /// これを下回らない (`balance_weight_gamma < 1.0` で完全に 0 まで落として
    /// しまうのを防ぐ安全弁)。
    pub balance_weight_min: f64,
    /// `--router-mode` 共通 (`hard-EM` / `backprop` どちらでも使う): E step で
    /// 求めた N 個 (N = `TrainingConfig::num_buckets`) の bucket 誤差のうち、
    /// 誤差が小さい方から `top_k` 個だけを M step の対象にする Top-K 選択。
    /// - [`RouterMode::HardEm`]: 選ばれた `top_k` 個を `softmax(-err)` で重み
    ///   付けした分布を router の教師信号にする
    ///   (`shogi_features::router_kpabs::oracle_targets_from_errors`)。`1`
    ///   (既定) で従来の hard-EM (one-hot)、`num_buckets` で古典的な soft-EM
    ///   (Jacobs & Jordan 1991 の responsibility)、その中間で Top-K Hard
    ///   Routing。
    /// - [`RouterMode::Backprop`]: 選ばれた `top_k` 個に softmax を制限した
    ///   router 自身の分布 `Q` の下での期待損失を router の重みについて直接
    ///   backprop する (`shogi_features::router_kpabs::RouterKPAbsWeights::train_backprop_batch`、
    ///   実際の LLM MoE で使われる Top-K routing と同じ考え方)。`top_k = 1`
    ///   は softmax の支持集合が 1 点に固定され勾配が恒等的に 0 になる
    ///   (学習が完全に止まる) ため、`backprop` では `top_k >= 2` が必要
    ///   ── `top_k_reduction_interval` で anneal する場合は `top_k_min` を
    ///   `2` 以上にしておくこと。
    ///
    /// `[1, num_buckets]` の範囲でなければならない (`run()` が検証する)。
    pub top_k: usize,
    /// `top_k` の superbatch ごとの減衰間隔 (`--top-k-reduction-interval`、
    /// `hard-EM` / `backprop` 共通)。`N` superbatch 終える度に `top_k -= 1`
    /// する (`1` の場合は毎 superbatch、`lr_gamma` / `balance_weight_gamma` と
    /// 同じタイミングで適用する)。`top_k` は `top_k_min` を下回らない。`0`
    /// (既定) で無効化 (`--top-k` を最後まで一定に保つ、従来動作)。
    pub top_k_reduction_interval: usize,
    /// `top_k` を anneal (`top_k_reduction_interval`) させるときの下限
    /// (`--top-k-min`)。既定は `1` (`--top-k-reduction-interval` を使わない
    /// 従来動作と同じ)。[`RouterMode::Backprop`] で `top_k_reduction_interval`
    /// を使う場合、`top_k` が `1` まで落ちると勾配が恒等的に 0 になり学習が
    /// 止まってしまう (`top_k` の doc 参照) ため、`2` 以上を指定すること。
    /// `[1, top_k]` の範囲でなければならない (`run()` が検証する)。
    pub top_k_min: usize,
}

/// LayerStack (bucket-aware) / Simple (bucket-less) どちらの backend で学習する
/// にも要る subset。bucket 数 N は [`TrainingConfig::num_buckets`] で持つ (既定 9、
/// LayerStack `--num-buckets` 由来)。learning rate / WDL schedule は別に
/// [`LrScheduler`] / [`WdlScheduler`] を渡す。
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// network id — checkpoint file 名にのみ使う (`{net_id}-{sb}.bin`)。
    pub net_id: String,
    /// 入力 feature set spec。dataloader の sparse index 化に使う
    /// (どの feature set で学習するかの単一の真実源)。
    pub feature_set: FeatureSetSpec,
    /// checkpoint の出力先 directory (呼び出し側で作成しておくこと)。
    pub output_dir: PathBuf,
    /// 開始 superbatch (1-indexed, inclusive)。
    pub start_superbatch: usize,
    /// 終了 superbatch (inclusive)。
    pub end_superbatch: usize,
    /// 1 superbatch あたりの batch 数。
    pub batches_per_superbatch: usize,
    /// 1 batch あたりの position 数。
    pub batch_size: usize,
    /// `save_rate` superbatch ごと (および末尾) に checkpoint を書き出す。
    pub save_rate: usize,
    /// LayerStack arch string に書く `fv_scale`。`None` は token を省略する。
    pub fv_scale: Option<i32>,
    /// 推論用 checkpoint の serialization format。
    pub output_format: OutputFormat,
    /// `Some(n)` のとき、新しい raw checkpoint (`{net_id}-{sb}.ckpt`) を書いた
    /// 後、直近 `n` 個より古い raw checkpoint を削除する (`--keep-checkpoints`)。
    /// `None` は全 raw checkpoint を保持。raw state は ~1.8GB/個 なので
    /// save-rate × superbatches が大きい長期ランでは明示指定推奨。量子化 `.bin`
    /// (~116MB) は本設定に関わらず常に全保持する (推論 artifact なので保守的に)。
    pub keep_raw_checkpoints: Option<usize>,
    /// どの loss kernel で学習するか (sigmoid-MSE / WRM) + 固定パラメータ。
    pub loss: LossKind,
    /// `Some(t)` のとき `|score| >= t` の position を skip する (`--score-drop-abs`)。
    pub score_drop_abs: Option<i32>,
    /// `Some(c)` のとき drop を生き残った position の score を `[-c, c]` に飽和
    /// させる (`--score-clamp-abs`)。教師の encode 変種 (clip 上限の違い等) を
    /// 単一の上限へ正規化する用途。i16 なのは PSV score の表現域に合わせ、
    /// 消費側の縮小 cast (wrap) を型で不可能にするため。
    pub score_clamp_abs: Option<i16>,
    /// dataloader の prefetch worker 数 (`--threads`)。`0` は `1` 扱い。
    /// `1` で決定論的逐次 read 相当、`>= 2` で並列パース (1 epoch 内の
    /// position 順序は非決定的になる; [`BucketedPrefetchedLoader`] doc 参照)。
    pub threads: usize,
    /// `Some` のとき held-out validation 用 PSV / HCPE file。各 superbatch 末に
    /// forward-only 検証を走らせ test_loss / test_accuracy を report する。
    pub test_data: Option<PathBuf>,
    /// held-out validation 1 回あたりの検証局面数 (`batch_size` 単位に切り上げて
    /// 満タン batch を作る)。`test_data` / `test_tail_positions` がいずれも
    /// `None` のときは無視される。
    pub test_positions: usize,
    /// `Some(n)` のとき `data_path` 末尾 `n` 局面を training stream から除外し
    /// held-out 専用に分離する (`--test-tail-positions`)。training は
    /// `[0, file_size - n * PSV_RECORD_BYTES)`、validation は
    /// `[file_size - n * PSV_RECORD_BYTES, file_size)` を読む。`test_data` と
    /// 同時指定は `validate()` で error。
    pub test_tail_positions: Option<u64>,
    /// dataloader worker で選択された bucket mode を計算して per-position
    /// bucket を `Batch` と共に返すか。LayerStack (bucket-aware) は `true`、Simple
    /// (bucket-less) は `false` で worker CPU 仕事を ~1 board 推論分削減できる。
    /// `false` のとき backend に渡る `bucket_idx` は空 slice になるので backend は
    /// 参照してはならない (Simple `TrainerBackend::train_step` は元から bucket_idx
    /// を使わない契約)。
    pub compute_bucket: bool,
    /// LayerStack の output bucket 数 (`--num-buckets`)。dataloader worker が
    /// bucket mode が `0..num_buckets-1` を emit
    /// するために必要。`compute_bucket = false` の Simple 経路では bucket 計算
    /// 自体が skip されるため値は参照されない (任意の `>= 1` で可)。
    pub num_buckets: usize,
    /// `true` のとき各 superbatch 末で backend の累積 FP16 clamp event count を
    /// 読んで `[fp16-clamp] sb=N total=X delta=Y elems=Z ratio=R` line を stderr に
    /// 出す (`--monitor-fp16-clamps`)。`false` は no-op (D2H read も skip、kernel
    /// 側 atomic 計数は常時有効だが host 報告のみ gate)。`--ft-fp16-out` 無しの
    /// run では FP16 clamp kernel 自体が launch されないため total は常に 0。
    pub monitor_fp16_clamps: bool,
    /// `true` のとき dataloader worker が各 position の実 active feature 数
    /// (`Batch::push_decoded_counting` の `written`) を histogram に集計し、各
    /// superbatch 末に `[active-hist] sb=N positions=T mean=M p50=.. p90=.. p99=..
    /// max=..` line を、run 終了時に non-zero bin の full dump (`[active-hist]
    /// count[k]=n`) を stderr に出す (`--monitor-active-features`)。stats は累積
    /// histogram に対して計算する。`false` では histogram を確保せず worker の
    /// ホットパスに計装コードを通さない (feature-set 非依存、threat off でも動く)。
    pub monitor_active_features: bool,
    /// `bucket_mode == Router` のときの hard-EM router 学習設定。`Some`
    /// でなければならない (`run()` が bucket_mode を見て検証する)。他の
    /// bucket mode では無視される (`None` を渡すのが自然)。
    pub router: Option<RouterTrainingConfig>,
}

impl TrainingConfig {
    fn validate(&self) -> io::Result<()> {
        if self.start_superbatch == 0 {
            return Err(io::Error::other(
                "start_superbatch must be >= 1 (1-indexed)",
            ));
        }
        if self.end_superbatch < self.start_superbatch {
            return Err(io::Error::other(format!(
                "end_superbatch ({}) < start_superbatch ({})",
                self.end_superbatch, self.start_superbatch
            )));
        }
        if self.batches_per_superbatch == 0 {
            return Err(io::Error::other("batches_per_superbatch must be >= 1"));
        }
        if self.batch_size == 0 {
            return Err(io::Error::other("batch_size must be >= 1"));
        }
        if self.save_rate == 0 {
            return Err(io::Error::other("save_rate must be >= 1"));
        }
        if let Some(0) = self.keep_raw_checkpoints {
            return Err(io::Error::other(
                "keep_raw_checkpoints must be >= 1 when set (0 would delete every raw checkpoint)",
            ));
        }
        self.loss.validate()?;
        if let Some(t) = self.score_drop_abs
            && t < 1
        {
            return Err(io::Error::other(format!(
                "score_drop_abs must be >= 1 (got {t}); a non-positive threshold would drop every position"
            )));
        }
        if let Some(c) = self.score_clamp_abs
            && c < 1
        {
            return Err(io::Error::other(format!(
                "score_clamp_abs must be >= 1 (got {c}); a non-positive bound would saturate \
                 every score to it"
            )));
        }
        if let (Some(t), Some(c)) = (self.score_drop_abs, self.score_clamp_abs)
            && i64::from(c) >= i64::from(t)
        {
            return Err(io::Error::other(format!(
                "score_clamp_abs ({c}) must be < score_drop_abs ({t}); every position surviving \
                 the drop filter already has |score| < {t}, so the clamp would never fire"
            )));
        }
        if self.test_data.is_some() && self.test_positions == 0 {
            return Err(io::Error::other(
                "test_positions must be >= 1 when test_data is set",
            ));
        }
        if self.test_data.is_some() && self.test_tail_positions.is_some() {
            return Err(io::Error::other(
                "test_data and test_tail_positions are mutually exclusive \
                 (pick one held-out source: external PSV/HCPE file or training-data tail)",
            ));
        }
        if let Some(n) = self.test_tail_positions {
            if n == 0 {
                return Err(io::Error::other(
                    "test_tail_positions must be >= 1 when set",
                ));
            }
            if self.test_positions == 0 {
                return Err(io::Error::other(
                    "test_positions must be >= 1 when test_tail_positions is set",
                ));
            }
        }
        if self.num_buckets == 0 {
            return Err(io::Error::other(
                "num_buckets must be >= 1 (LayerStack uses `--num-buckets` in [2, 9])",
            ));
        }
        Ok(())
    }
}

// =============================================================================
// ActiveHistStats — 実 active feature 数 histogram の要約統計
// =============================================================================

/// 実 active feature 数 histogram (`bin[k]` = active 数 `k` の position 数) から
/// 計算した分布要約 (`--monitor-active-features` の superbatch 末 line 用)。
struct ActiveHistStats {
    /// histogram に入っている総 position 数 (= bin の総和)。
    positions: u64,
    /// active 数の算術平均。
    mean: f64,
    /// 50 / 90 / 99 percentile (nearest-rank: 累積が `ceil(p * N)` に達する最小
    /// bin index)。
    p50: usize,
    p90: usize,
    p99: usize,
    /// non-zero だった最大 bin index (= 観測された最大 active 数)。
    max: usize,
}

impl ActiveHistStats {
    /// `histogram` (index = active 数、値 = 出現 position 数) から統計を計算する。
    /// 総和が 0 (= 1 position も入っていない) なら `None`。
    fn from_histogram(histogram: &[u64]) -> Option<Self> {
        let positions: u64 = histogram.iter().sum();
        if positions == 0 {
            return None;
        }
        let weighted: u128 = histogram
            .iter()
            .enumerate()
            .map(|(active, &count)| active as u128 * count as u128)
            .sum();
        let mean = weighted as f64 / positions as f64;

        // nearest-rank percentile: 累積 count が threshold 以上になる最小 bin。
        let percentile = |p: f64| -> usize {
            let threshold = (p * positions as f64).ceil() as u64;
            let mut cumulative = 0u64;
            for (active, &count) in histogram.iter().enumerate() {
                cumulative += count;
                if cumulative >= threshold {
                    return active;
                }
            }
            histogram.len().saturating_sub(1)
        };

        let max = histogram.iter().rposition(|&count| count > 0).unwrap_or(0);

        Some(Self {
            positions,
            mean,
            p50: percentile(0.50),
            p90: percentile(0.90),
            p99: percentile(0.99),
            max,
        })
    }
}

// =============================================================================
// run — superbatch loop
// =============================================================================

/// superbatch training loop を実行し、`cfg.output_dir` 配下に checkpoint を書き出す。
///
/// - `backend`: GPU step を実行する backend (`bins/nnue_train::GpuTrainer`)
/// - `data_path`: PSV file (`PackedSfenValue` × N、40 bytes 固定)
/// - `bucket_mode`: dataloader と held-out validation に共通の bucket 算出方式。
///   progress8kpabs の重みは process-global なので呼び出し前に load 済であること
/// - `lr_scheduler` / `wdl_scheduler`: superbatch / batch index から lr / wdl lambda を返す
/// - `cfg`: hyper-parameter (superbatch 範囲、batch 構成、save 間隔、loss scale、score-drop-abs、`threads`)
/// - `experiment`: `Some` のとき、run 開始時・superbatch ごと・正常終了時に
///   experiment.json を atomic に書き出す。書き込み失敗は warning のみで
///   training は止めない。`None` なら構造化ログを残さない
///
/// PSV stream は [`BucketedPrefetchedLoader`] で `cfg.threads` 本の worker から
/// `decode()` 1 回 / position の bucket-aware 先読み + ring-buffer 再利用される。
/// worker 数 ≥ 2 では 1 epoch 内の position 順序が非決定的になる点に注意
/// (training では問題ない)。
pub fn run<B, L, W, M>(
    backend: &mut B,
    data_path: &Path,
    bucket_mode: &M,
    lr_scheduler: &L,
    wdl_scheduler: &W,
    cfg: &TrainingConfig,
    mut experiment: Option<&mut ExperimentLogger>,
) -> io::Result<()>
where
    B: TrainerBackend,
    L: LrScheduler,
    W: WdlScheduler,
    M: Copy + Into<BucketMode>,
{
    cfg.validate()?;
    let resolved_bucket_mode: BucketMode = (*bucket_mode).into();
    if resolved_bucket_mode == BucketMode::Router && cfg.router.is_none() {
        return Err(io::Error::other(
            "TrainingConfig::router must be Some(..) when bucket_mode is Router",
        ));
    }
    if resolved_bucket_mode != BucketMode::Router && cfg.router.is_some() {
        return Err(io::Error::other(
            "TrainingConfig::router must be None unless bucket_mode is Router",
        ));
    }

    // data file の byte サイズを取って PSV alignment を確認。
    // `--test-tail-positions` の split 計算がここで PSV record 境界に揃うか
    // 決まるため、tail 経路に入る前に確実に reject する。
    let file_size = std::fs::metadata(data_path)?.len();
    if !file_size.is_multiple_of(PSV_RECORD_BYTES) {
        return Err(io::Error::other(format!(
            "data file {} size {file_size} is not a multiple of PSV record size \
             ({PSV_RECORD_BYTES} bytes); the file is corrupted or not a PackedSfenValue stream",
            data_path.display(),
        )));
    }

    let train_end_offset = match cfg.test_tail_positions {
        Some(n) => {
            let tail_bytes = n.checked_mul(PSV_RECORD_BYTES).ok_or_else(|| {
                io::Error::other(format!(
                    "test_tail_positions ({n}) * PSV record size ({PSV_RECORD_BYTES}) \
                     overflows u64"
                ))
            })?;
            if tail_bytes >= file_size {
                return Err(io::Error::other(format!(
                    "test_tail_positions ({n}) leaves no training records \
                     (data file {} has {} records)",
                    data_path.display(),
                    file_size / PSV_RECORD_BYTES,
                )));
            }
            file_size - tail_bytes
        }
        None => file_size,
    };

    let mut loader = BucketedPrefetchedLoader::spawn(
        data_path,
        cfg.batch_size,
        cfg.score_drop_abs,
        cfg.score_clamp_abs,
        cfg.threads,
        (*bucket_mode).into(),
        cfg.feature_set,
        cfg.compute_bucket,
        cfg.num_buckets,
        train_end_offset,
        cfg.monitor_active_features,
    )?;

    println!(
        "[train] data={} | net_id={} | superbatches {}..={} | {} batches/sb x bs {} \
         | lr-sched: {lr_scheduler} | wdl-sched: {wdl_scheduler} | loss: {} | score-drop-abs {:?} | score-clamp-abs {:?} | dataloader threads {}",
        data_path.display(),
        cfg.net_id,
        cfg.start_superbatch,
        cfg.end_superbatch,
        cfg.batches_per_superbatch,
        cfg.batch_size,
        cfg.loss,
        cfg.score_drop_abs,
        cfg.score_clamp_abs,
        cfg.threads.max(1),
    );

    // held-out validation 集合を起動時に固定読み込みする
    // (`--test-data` または `--test-tail-positions` 指定時)。毎 superbatch 末に
    // 同じ集合で test_loss / test_accuracy を測る。`validate()` で両者の同時
    // 指定は reject 済みなので exhaustive な 4-arm match で意図を明示。
    let heldout = match (&cfg.test_data, cfg.test_tail_positions) {
        (Some(test_path), None) => {
            let set = crate::validation::HeldoutSet::load(
                test_path,
                cfg.batch_size,
                cfg.score_drop_abs,
                cfg.score_clamp_abs,
                cfg.test_positions,
                bucket_mode,
                cfg.feature_set,
                cfg.num_buckets,
            )?;
            println!(
                "[train] held-out validation (external file): data={} | {} batches x bs {} ({} positions)",
                test_path.display(),
                set.n_batches(),
                cfg.batch_size,
                set.n_positions(),
            );
            Some(set)
        }
        (None, Some(_)) => {
            let set = crate::validation::HeldoutSet::load_from_range(
                data_path,
                train_end_offset,
                file_size,
                cfg.batch_size,
                cfg.score_drop_abs,
                cfg.score_clamp_abs,
                cfg.test_positions,
                bucket_mode,
                cfg.feature_set,
                cfg.num_buckets,
            )?;
            println!(
                "[train] held-out validation (training tail): data={} | range [{}, {}) | \
                 {} batches x bs {} ({} positions)",
                data_path.display(),
                train_end_offset,
                file_size,
                set.n_batches(),
                cfg.batch_size,
                set.n_positions(),
            );
            Some(set)
        }
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!("validate() rejects test_data + test_tail_positions"),
    };

    // experiment.json を run 開始時点 (`status: "running"`) で一度書く。以降は
    // superbatch ごとに incremental に上書きする。
    if let Some(log) = experiment.as_mut() {
        write_experiment_log(log);
    }

    let positions_per_sb =
        (cfg.batches_per_superbatch as u64).saturating_mul(cfg.batch_size as u64);
    let run_start = Instant::now();

    // backend の async H2D で `train_step` が返って直ぐ `loader.recycle` すると、
    // queue 済 H2D copy のソース (`batch.stm_indices` 等の pageable `Vec`) を worker
    // thread が reset / 再充填してしまう。これを防ぐため **1 step 遅延 recycle**:
    // 直前 batch を `prev_pending` に保持し、次 `train_step` が queue 済 H2D を消化
    // した時点で recycle する (次 step の event sync が直前 batch の full pipeline
    // 完了を保証する)。同期 backend では実害なしだが、async backend を含めて統一形。
    let mut prev_pending: Option<(Batch, Vec<i32>, Vec<Vec<u32>>)> = None;

    // `Router` の hard-EM router 学習 state。他 bucket mode では未使用
    // (`router_adam = None`)。`RouterKPAbs` の重み自体は process-global なので
    // 呼び出し前に `init_random` / `init_with_weights` / `load_full` 済であること
    // (`run()` 自身はランダム初期化しない)。
    let mut router_adam: Option<RouterAdamState> = if resolved_bucket_mode == BucketMode::Router {
        Some(RouterAdamState::zeros(cfg.num_buckets))
    } else {
        None
    };
    // `RouterTrainingConfig::lr` / `balance_weight` の実効値。1 superbatch 終える
    // 度に `*_gamma` を掛けて減衰させる (`--router-lr-gamma` /
    // `--router-balance-weight-gamma`)。`balance_weight` は
    // `--router-balance-weight-min` を下回らないよう毎回 clamp する。
    // `cfg.router` が `None` (router 以外) のときは未使用。
    let mut current_router_lr: f64 = cfg.router.map(|rc| rc.lr).unwrap_or(0.0);
    let mut current_router_balance_weight: f64 = cfg.router.map(|rc| rc.balance_weight).unwrap_or(0.0);
    // `RouterTrainingConfig::top_k` の実効値。1 superbatch 終える度に
    // `--top-k-reduction-interval` superbatch ごとに 1 ずつ減らす
    // (`--top-k-reduction-interval` が `0` なら不変、従来動作)。`1` を下回らない。
    let mut current_top_k: usize = cfg.router.map(|rc| rc.top_k).unwrap_or(1);
    // sb 内で直近に観測した router 学習の診断情報 (負荷分散 loss / bucket 使用率
    // など)。sb 末の `[router]` ログ用。`refresh_interval > 1` の場合、更新され
    // なかった batch では前回の値がそのまま残る (= その sb 最後の refresh 時点の
    // スナップショット)。
    let mut last_router_stats: Option<shogi_features::router_kpabs::RouterTrainStats> = None;

    // sb 内 batch 進捗 print の頻度 (env var で可変、`0` で disable)。stderr が
    // TTY なら `\r` で同 line を上書き、それ以外 (pipe / `tee` ファイル等) なら
    // `\n` で改行して log file が pure text に保たれるようにする。
    let progress_every = std::env::var("NNUE_TRAIN_BATCH_PROGRESS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(128);
    let progress_terminator = if io::stderr().is_terminal() {
        '\r'
    } else {
        '\n'
    };

    // FP16 clamp monitor の前回 sb 末累積値 (delta 計算用)。`--monitor-fp16-clamps`
    // 無しなら read 自体 skip するので未使用。
    let mut prev_fp16_clamp_count: u64 = 0;
    let mut prev_fp16_clamp_elems: u64 = 0;

    for sb in cfg.start_superbatch..=cfg.end_superbatch {
        let sb_start = Instant::now();
        let mut sb_loss: f64 = 0.0;
        let mut sb_positions: u64 = 0;
        let mut sb_printed_progress = false;

        for batch_idx in 0..cfg.batches_per_superbatch {
            let lr = lr_scheduler.lr(batch_idx, sb);
            let wdl = wdl_scheduler.blend(batch_idx, sb, cfg.end_superbatch);

            let (batch, buckets, router_indices) = loader.next_batch()?.ok_or_else(|| {
                io::Error::other(
                    "dataloader stopped supplying batches unexpectedly (workers exited without an error)",
                )
            })?;
            let n_pos = batch.n_positions;

            // router (hard-EM/backprop 共通): bucket_mode 自体
            // (`bucket_mode.bucket_board` 経由で dataloader が既に割り当てた
            // `buckets`) は毎 batch 常に最新の router 出力を使う。ここでは
            // さらに router 自身を **この batch の現在の eval net に対する
            // 誤差** で更新する (`--router-refresh-interval` 間隔)。N 通り
            // (N = `cfg.num_buckets`) の固定 bucket で forward-only
            // (`validate_step`、backward/optimizer step なし) を回して各
            // position・各 bucket の誤差 `errs_by_position` を求め、その先の
            // M step は `rc.mode` で分岐する (詳細は `router_kpabs` モジュール
            // doc):
            // - `RouterMode::HardEm`: `oracle_targets_from_errors` (`--top-k`
            //   依存) で教師分布を作り、router の (soft-label) N-class cross
            //   entropy を 1 step 分 backprop する (`RouterKPAbs::train_oracle_batch`)。
            // - `RouterMode::Backprop`: 教師分布を経由せず、router 自身の
            //   softmax 分布の下での期待損失を router の重みについて直接
            //   backprop する (`RouterKPAbs::train_backprop_batch`)。
            // いずれも CPU 側の計算のみで GPU kernel の変更は不要。
            if let (Some(rc), Some(adam)) = (cfg.router, router_adam.as_mut())
                && batch_idx % rc.refresh_interval.max(1) == 0
            {
                let pred_scale = oracle_pred_scale(cfg.loss);
                let target_scale = oracle_target_scale(cfg.loss);
                // `errs[k][i]` = bucket k に固定して forward したときの position
                // i の誤差。Top-K/soft-EM ターゲットを作るには N 個すべての
                // bucket の誤差が要る (hard-EM の argmin だけでは足りない)。
                let mut errs: Vec<Vec<f64>> = Vec::with_capacity(cfg.num_buckets);
                for k in 0..cfg.num_buckets {
                    let k_buckets = vec![k as i32; n_pos];
                    let out = backend.validate_step(&batch, &k_buckets, wdl, cfg.loss)?;
                    let mut err_k = Vec::with_capacity(n_pos);
                    for i in 0..n_pos {
                        let pred = sigmoid(out.net_output[i] * pred_scale);
                        let target =
                            wdl * batch.wdl[i] + (1.0 - wdl) * sigmoid(batch.score[i] * target_scale);
                        err_k.push(((pred - target) * (pred - target)) as f64);
                    }
                    errs.push(err_k);
                }
                let errs_by_position: Vec<Vec<f64>> = (0..n_pos)
                    .map(|i| (0..cfg.num_buckets).map(|k| errs[k][i]).collect())
                    .collect();
                let stats = match rc.mode {
                    RouterMode::HardEm => {
                        let oracle_targets: Vec<Vec<f64>> = errs_by_position
                            .iter()
                            .map(|errs_i| {
                                shogi_features::router_kpabs::oracle_targets_from_errors(
                                    errs_i,
                                    current_top_k,
                                )
                            })
                            .collect();
                        RouterKPAbs::train_oracle_batch(
                            &router_indices[..n_pos],
                            &oracle_targets,
                            adam,
                            current_router_lr,
                            rc.weight_decay,
                            current_router_balance_weight,
                        )
                    }
                    RouterMode::Backprop => RouterKPAbs::train_backprop_batch(
                        &router_indices[..n_pos],
                        &errs_by_position,
                        adam,
                        current_router_lr,
                        rc.weight_decay,
                        current_router_balance_weight,
                        current_top_k,
                    ),
                };
                last_router_stats = Some(stats);
            }

            let loss = backend.train_step(&batch, &buckets, lr, wdl, cfg.loss)?;
            // 直前 batch を recycle: その H2D は今 `train_step` 呼出内の event sync で
            // 完了済 (lag-1 pipeline)。very first batch は prev_pending=None で no-op。
            if let Some(prev) = prev_pending.take() {
                loader.recycle(prev);
            }
            prev_pending = Some((batch, buckets, router_indices));
            sb_loss += loss;
            sb_positions += n_pos as u64;

            // batch 進捗 print: `progress_every` batches ごとに sb 内 pos/s + % +
            // batch count を stderr に出す。TTY なら `\r` で上書き、pipe なら `\n`
            // で改行 (`tee` log が editor で binary 判定されないよう)。
            if progress_every > 0
                && (batch_idx + 1) % progress_every == 0
                && batch_idx + 1 < cfg.batches_per_superbatch
            {
                let done = batch_idx + 1;
                let pct = 100.0 * done as f64 / cfg.batches_per_superbatch as f64;
                let pps = sb_positions as f64 / sb_start.elapsed().as_secs_f64().max(1e-9);
                let mut stderr = io::stderr().lock();
                let written = write!(
                    stderr,
                    "{}[train] sb {}/{} [{:.1}% ({}/{} batches, {:.0} pos/s)]",
                    progress_terminator,
                    sb,
                    cfg.end_superbatch,
                    pct,
                    done,
                    cfg.batches_per_superbatch,
                    pps,
                )
                .is_ok();
                if written {
                    let _ = stderr.flush();
                    sb_printed_progress = true;
                }
            }
        }
        // progress line は terminator を **prefix** に置く format で書いている
        // ため、TTY (`\r` 上書き) でも pipe (`\n` 改行) でも最後の line は末尾
        // 改行を持たない。sb 完了 println が直後に続くと同一 line に追記されて
        // しまうので、progress を 1 回でも印字した sb は明示的に改行を入れて
        // line を terminate する。`sb_printed_progress` で 0 回 sb の余分な
        // 空行を抑制。
        if sb_printed_progress {
            eprintln!();
        }
        // backend が前 step の loss を遅延報告する pipeline 実装 (async loss readback
        // 等) の場合、sb 内最後 batch の loss が未報告のまま残る。`flush_pending_loss`
        // で drain して sb_loss に加算することで、1 sb の loss 集計が正確になる
        // (同期 readback の default 実装は `0.0` を返すので no-op)。
        sb_loss += backend.flush_pending_loss()?;

        let sb_secs = sb_start.elapsed().as_secs_f64().max(1e-9);
        let mean_loss = if sb_positions == 0 {
            f64::NAN
        } else {
            sb_loss / sb_positions as f64
        };
        let pos_per_sec = sb_positions as f64 / sb_secs;
        let remaining_positions = positions_per_sb.saturating_mul((cfg.end_superbatch - sb) as u64);
        let eta_secs = if pos_per_sec > 0.0 {
            remaining_positions as f64 / pos_per_sec
        } else {
            f64::NAN
        };
        let lr_now = lr_scheduler.lr(0, sb);
        let wdl_now = wdl_scheduler.blend(0, sb, cfg.end_superbatch);

        // held-out validation: superbatch 末に forward-only 検証を 1 回走らせる
        // (`--test-data` 指定時のみ)。training step と同じ loss kind と、当 superbatch
        // 代表の wdl_lambda (`wdl_now` = batch_idx 0 の blend、sb 末 report と同値) で
        // 測り、test_loss を同 superbatch の training loss と比較可能にする。superbatch
        // 内で wdl が変動する scheduler (`WarmupWDL` の warmup 区間など) では
        // batch_idx 0 の値で代表させる近似になる (sb 末 report と同じ扱い)。
        let validation = match &heldout {
            Some(set) => {
                let report = set.evaluate(backend, wdl_now, cfg.loss)?;
                if !report.mean_loss.is_finite() {
                    eprintln!(
                        "[train] warning: superbatch {sb} held-out validation loss is \
                         non-finite ({}) — possible divergence",
                        report.mean_loss
                    );
                }
                Some(report)
            }
            None => None,
        };
        let val_str = match &validation {
            Some(r) => format!(
                " | test_loss {:.6} | test_acc {:.4}",
                r.mean_loss, r.accuracy
            ),
            None => String::new(),
        };

        println!(
            "[train] superbatch {}/{} | loss {:.6} | {:.0} pos/s | lr {:.4e} | wdl {:.3} | sb {:.1}s | ETA {}{}",
            sb,
            cfg.end_superbatch,
            mean_loss,
            pos_per_sec,
            lr_now,
            wdl_now,
            sb_secs,
            format_hms(eta_secs),
            val_str,
        );

        if let Some(stats) = last_router_stats.as_ref() {
            // 負荷分散 loss (`N * Σ f_i * P_i`、min=1.0 で完全均等、max=N で
            // 1 bucket に完全集中) と、直近 refresh 時点での bucket 使用率
            // (router 自身の argmax の分布) を出す。1 bucket に偏り続けている
            // 場合は `--router-balance-weight` を上げる目安になる。
            let usage_str = stats
                .bucket_usage
                .iter()
                .map(|u| format!("{:.2}", u))
                .collect::<Vec<_>>()
                .join(",");
            let mode = cfg.router.map(|rc| rc.mode).unwrap_or_default();
            let mode_str = match mode {
                RouterMode::HardEm => format!("hard-em top_k={current_top_k}"),
                RouterMode::Backprop => format!("backprop top_k={current_top_k}"),
            };
            eprintln!(
                "[router] sb={} loss={:.4} balance_loss={:.4} (min=1.0,max={}) mode={} usage=[{}]",
                sb,
                stats.cross_entropy_loss,
                stats.balance_loss,
                stats.bucket_usage.len(),
                mode_str,
                usage_str,
            );
        }

        if cfg.monitor_fp16_clamps {
            // dft FP16 書き込みで `±65504` cap が当たった要素数 (累積) と処理要素数
            // (累積) を device から読み、当 sb の delta + ratio を出す。`--ft-fp16-out`
            // 無し / clamp 計装を持たない backend では `(0, 0)` で delta 0 / elems 0 /
            // ratio 0 になる。
            let (total_count, total_elems) = backend.read_fp16_clamp_count()?;
            let delta_count = total_count.saturating_sub(prev_fp16_clamp_count);
            let delta_elems = total_elems.saturating_sub(prev_fp16_clamp_elems);
            prev_fp16_clamp_count = total_count;
            prev_fp16_clamp_elems = total_elems;
            let ratio = if delta_elems > 0 {
                delta_count as f64 / delta_elems as f64
            } else {
                0.0
            };
            eprintln!(
                "[fp16-clamp] sb={} total={} delta={} elems={} ratio={:.3e}",
                sb, total_count, delta_count, delta_elems, ratio,
            );
        }

        if cfg.monitor_active_features
            && let Some(hist) = loader.active_histogram_snapshot()
            && let Some(stats) = ActiveHistStats::from_histogram(&hist)
        {
            // 累積 histogram に対する分布統計 (実 active feature 数)。overflow は
            // `push_decoded_counting` 側の hard-error で弾かれるため histogram には
            // 現れない。
            eprintln!(
                "[active-hist] sb={} positions={} mean={:.1} p50={} p90={} p99={} max={}",
                sb, stats.positions, stats.mean, stats.p50, stats.p90, stats.p99, stats.max,
            );
        }

        let saved = sb % cfg.save_rate == 0 || sb == cfg.end_superbatch;
        if saved {
            let path = cfg.output_dir.join(format!("{}-{}.bin", cfg.net_id, sb));
            backend.save_checkpoint(&path, cfg.fv_scale, cfg.output_format)?;
            println!("[train] checkpoint saved: {}", path.display());

            // resume 用 raw checkpoint: weight raw f32 + optimizer state + step + sb。
            // 実験ログがあれば run id を渡し `*.ckpt` に埋め込む (resume 時に
            // その run が lineage の親として参照される)。
            let raw_path = cfg.output_dir.join(format!("{}-{}.ckpt", cfg.net_id, sb));
            let run_id = experiment
                .as_deref()
                .map(ExperimentLogger::id)
                .unwrap_or("");
            backend.save_resume_checkpoint(&raw_path, sb, run_id, lr_scheduler.horizon())?;
            println!("[train] resume checkpoint saved: {}", raw_path.display());

            if let Some(keep) = cfg.keep_raw_checkpoints {
                prune_old_raw_checkpoints(&cfg.output_dir, &cfg.net_id, keep);
            }

            // router: 重み + Adam optimizer state を sidecar file に保存する
            // (main net の raw checkpoint とは別ファイル、`--router-resume` で
            // 読み戻せる)。
            if let Some(adam) = router_adam.as_ref() {
                let router_path = cfg
                    .output_dir
                    .join(format!("{}-{}.router.ckpt", cfg.net_id, sb));
                if let Err(e) = RouterKPAbs::save_full(&router_path, adam) {
                    eprintln!(
                        "[train] warning: failed to save router checkpoint {}: {e}",
                        router_path.display()
                    );
                } else {
                    println!("[train] router checkpoint saved: {}", router_path.display());
                }
            }
        }

        // superbatch の処理 (checkpoint 保存を含む) をすべて終えてから
        // experiment.json を更新する。これで `checkpoints` に載せたファイル名は
        // 書き込み時点で実在する。
        if let Some(log) = experiment.as_mut() {
            log.record_superbatch(
                sb,
                mean_loss,
                sb_positions,
                run_start.elapsed().as_secs_f64(),
                validation.map(|r| r.mean_loss),
                validation.map(|r| r.accuracy),
            );
            if let Some(stats) = last_router_stats.as_ref() {
                // `current_router_lr` / `current_router_balance_weight` は
                // まさにこの sb で使った (decay 前の) 値 — この下の decay 適用は
                // 次 sb 向けなので、記録は decay 前の現在値で行う。
                log.record_router(RouterHistoryEntry {
                    superbatch: sb,
                    ce_loss: stats.cross_entropy_loss,
                    balance_loss: stats.balance_loss,
                    usage: stats.bucket_usage.clone(),
                    lr: current_router_lr,
                    balance_weight: current_router_balance_weight,
                });
            }
            if saved {
                log.note_checkpoint(format!("{}-{}.bin", cfg.net_id, sb));
                log.note_checkpoint(format!("{}-{}.ckpt", cfg.net_id, sb));
            }
            write_experiment_log(log);
        }

        // router: 1 superbatch 終える度に lr / balance_weight を減衰させる
        // (`--router-lr-gamma` / `--router-balance-weight-gamma`)。
        // `balance_weight` は `--router-balance-weight-min` を下限に clamp する
        // (`balance_weight_gamma < 1.0` で完全に 0 まで落ちてしまうのを防ぐ)。
        if let Some(rc) = cfg.router {
            current_router_lr *= rc.lr_gamma;
            current_router_balance_weight =
                (current_router_balance_weight * rc.balance_weight_gamma).max(rc.balance_weight_min);
            // `--top-k-reduction-interval` superbatch 終える度に top_k を 1 減らす
            // (`0` なら無効化、従来動作のまま)。`start_superbatch` からの経過
            // superbatch 数で数えるので resume 後も interval の周期は維持される。
            // `hard-EM` / `backprop` 共通 (`backprop` で `top_k` が `top_k_min`
            // 未満、特に `1` まで落ちると勾配が恒等的に 0 になる点は `top_k_min`
            // の doc を参照)。
            if rc.top_k_reduction_interval > 0
                && (sb - cfg.start_superbatch + 1) % rc.top_k_reduction_interval == 0
            {
                current_top_k = current_top_k.saturating_sub(1).max(rc.top_k_min);
            }
        }
    }

    // 最後に残った batch を recycle: 直前 sb 末の `flush_pending_loss` 内 event sync で
    // 当該 batch の H2D は完了済、loader に返しても安全。
    if let Some(prev) = prev_pending.take() {
        loader.recycle(prev);
    }

    // 学習終了時に 1 回だけ full histogram を dump (non-zero bin のみ)。累積なので
    // run 全体の実 active feature 数分布になる。
    if cfg.monitor_active_features
        && let Some(hist) = loader.active_histogram_snapshot()
    {
        for (active, count) in hist.iter().enumerate() {
            if *count > 0 {
                eprintln!("[active-hist] count[{active}]={count}");
            }
        }
    }

    println!(
        "[train] done in {} ({} superbatches)",
        format_hms(run_start.elapsed().as_secs_f64()),
        cfg.end_superbatch + 1 - cfg.start_superbatch,
    );

    if let Some(log) = experiment.as_mut() {
        log.mark_finished(run_start.elapsed().as_secs_f64());
        write_experiment_log(log);
    }
    Ok(())
}

/// experiment.json を書き出し、失敗時は warning のみ出して training を止めない
/// (構造化ログは補助情報であり、書き込み失敗で学習を落とさない)。
fn write_experiment_log(log: &ExperimentLogger) {
    if let Err(e) = log.write() {
        eprintln!(
            "[train] warning: failed to write experiment log {}: {e}",
            log.path().display()
        );
    }
}

/// `{net_id}-{sb}.ckpt` 形式の raw checkpoint のうち、superbatch 番号 (`sb`) の
/// 大きい順に `keep` 個だけ残し、それより古いものを削除する
/// (`--keep-checkpoints`)。量子化 `.bin` には触らない (推論 artifact なので全保持)。
///
/// 削除失敗 (権限・他プロセス) は警告のみで `run` を止めない (training 続行優先)。
/// `keep == 0` は `TrainingConfig::validate` で reject 済の想定だが、万一渡されても
/// 全削除はしない (no-op で警告)。
fn prune_old_raw_checkpoints(output_dir: &Path, net_id: &str, keep: usize) {
    if keep == 0 {
        eprintln!(
            "[train] warning: keep_raw_checkpoints=0 ignored (would delete all raw checkpoints)"
        );
        return;
    }
    let prefix = format!("{net_id}-");
    let entries = match std::fs::read_dir(output_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "[train] warning: cannot read {} to prune raw checkpoints: {e}",
                output_dir.display()
            );
            return;
        }
    };
    // (superbatch 番号, パス) を収集。`{net_id}-<digits>.ckpt` だけ対象。
    let mut found: Vec<(usize, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(num_str) = rest.strip_suffix(".ckpt") else {
            continue;
        };
        if let Ok(sb) = num_str.parse::<usize>() {
            found.push((sb, path));
        }
    }
    if found.len() <= keep {
        return;
    }
    // superbatch 降順 → 先頭 `keep` 個を残し、残りを削除。
    found.sort_by_key(|(sb, _)| std::cmp::Reverse(*sb));
    for (sb, path) in found.into_iter().skip(keep) {
        match std::fs::remove_file(&path) {
            Ok(()) => println!(
                "[train] pruned old raw checkpoint: {} (sb {sb})",
                path.display()
            ),
            Err(e) => eprintln!(
                "[train] warning: failed to prune {} (sb {sb}): {e}",
                path.display()
            ),
        }
    }
}

/// 秒数を `1h23m45s` / `12m05s` / `42s` 形式に整形する (`??` if not finite)。
/// `router` hard-EM oracle sweep 用のスケール抽出。loss kernel が実際に
/// 使う変換 (`Sigmoid` の `scale`、`Wrm` の `in_scaling`/`target_scaling`) を
/// 流用して `net_output` / `score` を `[0, 1]` の勝率らしき値に写像し、9 通りの
/// bucket 間で誤差を比較可能にする。学習本体の loss とビット一致させる必要は
/// なく、**bucket の相対順位が概ね合っていれば十分**な router 教師信号として
/// 使うだけ (`Wrm` の完全な in_offset/target_offset シフトは簡略化のため省略)。
fn oracle_pred_scale(loss: LossKind) -> f32 {
    match loss {
        LossKind::Sigmoid { scale } => scale,
        LossKind::Wrm { in_scaling, .. } => 1.0 / in_scaling.max(1e-6),
    }
}

fn oracle_target_scale(loss: LossKind) -> f32 {
    match loss {
        LossKind::Sigmoid { scale } => scale,
        LossKind::Wrm { target_scaling, .. } => 1.0 / target_scaling.max(1e-6),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn format_hms(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "??".to_string();
    }
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{ConstantWDL, StepLR};

    fn sample_psv_path() -> PathBuf {
        // crates/nnue-train/Cargo.toml から相対で shogi-format/tests/data/sample.psv。
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/nnue-train has a parent dir")
            .join("shogi-format/tests/data/sample.psv")
    }

    /// loop driver の挙動 (step 回数 / checkpoint path / bucket 受け渡し) を検証する
    /// 最小 backend。GPU には触らず、loss は単調減少する dummy 値を返す。
    struct MockBackend {
        steps: usize,
        saves: Vec<PathBuf>,
        /// raw resume checkpoint の保存呼び出し (path, superbatch)。
        resume_saves: Vec<(PathBuf, usize)>,
        /// `save_resume_checkpoint` に渡された producer run id (呼び出し順)。
        resume_run_ids: Vec<String>,
        last_buckets: Vec<i32>,
        max_batch_positions: usize,
        seen_lr: Vec<f32>,
        /// `validate_step` の呼び出し回数 (held-out validation の検証用)。
        validate_steps: usize,
        /// `read_fp16_clamp_count` の呼び出し回数。`--monitor-fp16-clamps` が
        /// sb 末に呼んだ回数の検証用。
        clamp_reads: usize,
        /// `read_fp16_clamp_count` が返す `(count, elems)` の擬似累積。`train_step`
        /// 1 回ごとに `(7, 100)` ずつ増やし、sb 末 delta が log line に反映される
        /// ことを確認する。
        clamp_count_sim: u64,
        clamp_elems_sim: u64,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                steps: 0,
                saves: Vec::new(),
                resume_saves: Vec::new(),
                resume_run_ids: Vec::new(),
                last_buckets: Vec::new(),
                max_batch_positions: 0,
                seen_lr: Vec::new(),
                validate_steps: 0,
                clamp_reads: 0,
                clamp_count_sim: 0,
                clamp_elems_sim: 0,
            }
        }
    }

    impl TrainerBackend for MockBackend {
        fn train_step(
            &mut self,
            batch: &Batch,
            bucket_idx: &[i32],
            lr: f32,
            wdl_lambda: f32,
            loss: LossKind,
        ) -> io::Result<f64> {
            assert_eq!(
                bucket_idx.len(),
                batch.n_positions,
                "one bucket per position"
            );
            assert!(batch.n_positions <= batch.batch_size);
            assert!(lr > 0.0, "lr should be positive");
            assert!(
                loss.validate().is_ok(),
                "loss params should be valid: {loss}"
            );
            assert!(wdl_lambda.is_finite());
            assert!(
                bucket_idx.iter().all(|&b| (0..9).contains(&b)),
                "bucket in 0..9: {bucket_idx:?}"
            );
            self.steps += 1;
            self.last_buckets = bucket_idx.to_vec();
            self.max_batch_positions = self.max_batch_positions.max(batch.n_positions);
            self.seen_lr.push(lr);
            self.clamp_count_sim += 7;
            self.clamp_elems_sim += 100;
            // 単調減少する dummy loss (loss 推移の monotonic decreasing assertion 用)。
            Ok(1.0 / self.steps as f64)
        }

        fn validate_step(
            &mut self,
            batch: &Batch,
            bucket_idx: &[i32],
            wdl_lambda: f32,
            loss: LossKind,
        ) -> io::Result<ValidationStepOutput> {
            assert_eq!(
                bucket_idx.len(),
                batch.n_positions,
                "one bucket per position"
            );
            assert!(wdl_lambda.is_finite());
            assert!(
                loss.validate().is_ok(),
                "loss params should be valid: {loss}"
            );
            self.validate_steps += 1;
            // deterministic dummy: net_output に各 position の score を流用する
            // (sign-agreement 計算が回ることだけ確認できればよい)。
            let net_output: Vec<f32> = batch.score[..batch.n_positions].to_vec();
            Ok(ValidationStepOutput {
                sum_sq_err: 0.25 * batch.n_positions as f64,
                net_output,
            })
        }

        fn save_checkpoint(
            &mut self,
            path: &Path,
            _fv_scale: Option<i32>,
            _output_format: OutputFormat,
        ) -> io::Result<()> {
            self.saves.push(path.to_path_buf());
            Ok(())
        }

        fn save_resume_checkpoint(
            &mut self,
            path: &Path,
            superbatch: usize,
            run_id: &str,
            _lr_horizon: Option<usize>,
        ) -> io::Result<()> {
            self.resume_saves.push((path.to_path_buf(), superbatch));
            self.resume_run_ids.push(run_id.to_string());
            Ok(())
        }

        fn read_fp16_clamp_count(&mut self) -> io::Result<(u64, u64)> {
            self.clamp_reads += 1;
            Ok((self.clamp_count_sim, self.clamp_elems_sim))
        }
    }

    fn base_cfg() -> TrainingConfig {
        TrainingConfig {
            net_id: "test".to_string(),
            feature_set: shogi_features::FeatureSet::HalfKaHmMerged.spec(),
            output_dir: PathBuf::from("/tmp/nnue-train-trainer-test-unused"),
            start_superbatch: 1,
            end_superbatch: 3,
            batches_per_superbatch: 2,
            batch_size: 8,
            save_rate: 2,
            fv_scale: Some(nnue_format::layerstack_weights::FV_SCALE),
            output_format: OutputFormat::Tatara,
            keep_raw_checkpoints: None,
            loss: LossKind::Sigmoid { scale: 1.0 / 290.0 },
            score_drop_abs: None,
            score_clamp_abs: None,
            threads: 2,
            test_data: None,
            test_positions: 0,
            test_tail_positions: None,
            compute_bucket: true,
            num_buckets: 9,
            monitor_fp16_clamps: false,
            monitor_active_features: false,
            router: None,
        }
    }

    fn run_drives_superbatches_with_threads(threads: usize) {
        let progress = ShogiProgressKPAbs; // zero weights → p = sigmoid(0) = 0.5 → bucket 4
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            threads,
            ..base_cfg()
        };
        let mut backend = MockBackend::new();

        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            None,
        )
        .expect("run ok");

        // 3 superbatch × 2 batch = 6 train_step。
        assert_eq!(backend.steps, 6);
        assert_eq!(
            backend.max_batch_positions, cfg.batch_size,
            "every batch fully filled (file wraps)"
        );
        // save_rate=2 → sb 2 (2 % 2 == 0) と sb 3 (== end_superbatch) で save。
        assert_eq!(
            backend.saves,
            vec![
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-2.bin"),
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-3.bin"),
            ]
        );
        // raw resume checkpoint は同 superbatch で `{net_id}-{sb}.ckpt` に保存される。
        assert_eq!(
            backend.resume_saves,
            vec![
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-2.ckpt"),
                    2
                ),
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-3.ckpt"),
                    3
                ),
            ]
        );
        // 各 superbatch で lr が gamma 倍 (StepLR step=1, gamma=0.9)。batch 内は一定。
        // (lr は train_step 呼び出し順 = run の loop 順で決まり、dataloader の worker
        // 順序には依らない。)
        assert!((backend.seen_lr[0] - 1.0e-3).abs() < 1e-9);
        assert!((backend.seen_lr[2] - 1.0e-3 * 0.9).abs() < 1e-9); // 2nd superbatch, 1st batch
        // zero-weight progress → 全 position が bucket 4。
        assert!(!backend.last_buckets.is_empty());
        assert!(
            backend.last_buckets.iter().all(|&b| b == 4),
            "got {:?}",
            backend.last_buckets
        );
    }

    #[test]
    fn run_drives_superbatches_and_writes_checkpoints_single_worker() {
        // threads=1: 決定論的逐次 read 相当のパス。
        run_drives_superbatches_with_threads(1);
    }

    #[test]
    fn run_drives_superbatches_and_writes_checkpoints_multi_worker() {
        // threads>=2: 並列パース。順序は非決定的でも step 回数 / checkpoint / bucket /
        // lr schedule は不変。
        run_drives_superbatches_with_threads(4);
    }

    #[test]
    fn run_with_start_superbatch_offset_resumes_loop_and_lr_schedule() {
        // `start_superbatch != 1` (resume) で呼んだとき:
        //  - 正しい step 回数 (start..=end の superbatch 数 × batches/sb)
        //  - checkpoint / resume-checkpoint が start..=end の番号で命名される
        //  - lr schedule が offset を反映する (StepLR sb=3 = start * gamma^2)
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            start_superbatch: 3,
            end_superbatch: 5,
            save_rate: 2, // sb 4 (4 % 2 == 0) と sb 5 (== end) で save
            threads: 1,
            ..base_cfg()
        };
        let mut backend = MockBackend::new();
        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            None,
        )
        .expect("run ok");

        // 3 superbatch (3,4,5) × 2 batch = 6 step。
        assert_eq!(backend.steps, 6);
        // save_rate=2 → sb 4, sb 5。番号は start_superbatch offset を反映。
        assert_eq!(
            backend.saves,
            vec![
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-4.bin"),
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-5.bin"),
            ]
        );
        assert_eq!(
            backend.resume_saves,
            vec![
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-4.ckpt"),
                    4
                ),
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-5.ckpt"),
                    5
                ),
            ]
        );
        // lr schedule: sb 3 (1st batch) = start * gamma^((3-1)/1) = start * gamma^2。
        // StepLR は `start * gamma^((sb-1)/step)` (resume 時は sb を渡せば自動で正しい lr)。
        let expected_sb3 = 1.0e-3 * 0.9_f32 * 0.9_f32;
        assert!(
            (backend.seen_lr[0] - expected_sb3).abs() < 1e-9,
            "sb3 lr = {} expected {expected_sb3}",
            backend.seen_lr[0]
        );
        // sb 5 (5th step = 1st batch of sb 5) = start * gamma^4。
        let expected_sb5 = 1.0e-3 * 0.9_f32.powi(4);
        assert!(
            (backend.seen_lr[4] - expected_sb5).abs() < 1e-9,
            "sb5 lr = {} expected {expected_sb5}",
            backend.seen_lr[4]
        );
    }

    /// experiment.json テスト用の最小 `Params` (sigmoid loss、held-out 既定 None)。
    fn experiment_params() -> crate::experiment::Params {
        crate::experiment::Params {
            architecture: "LayerStack-1536-16-32-9bucket".to_string(),
            feature_set: "halfka-hm-merged".to_string(),
            ft_in: 73_305,
            ft_factorize: None,
            l0: 1536,
            l1: 16,
            l2: 32,
            num_buckets: Some(9),
            optimizer: "ranger".to_string(),
            bucket_mode: Some("progress8kpabs".to_string()),
            activation: None,
            progress_coeff: None,
            lr: 1.0e-3,
            lr_gamma: 0.9,
            lr_step: 1,
            lr_schedule: "start 0.001 gamma 0.9 drop every 1 superbatches".to_string(),
            batch_size: 8,
            batches_per_superbatch: 2,
            superbatches: 3,
            start_superbatch: 1,
            wdl: 0.0,
            start_wdl: None,
            end_wdl: None,
            scale: 290.0,
            weight_decay: 0.0,
            ft_weight_decay: None,
            dense_weight_decay: None,
            bias_weight_decay: None,
            ft_lr_mult: None,
            dense_lr_mult: None,
            bias_lr_mult: None,
            norm_loss_factor: None,
            qa: 127,
            qb: 64,
            loss_kind: "sigmoid".to_string(),
            wrm_in_scaling: None,
            wrm_in_offset: None,
            wrm_nnue2score: None,
            wrm_target_offset: None,
            wrm_target_scaling: None,
            wrm_pow_exp: None,
            wrm_qp_asymmetry: None,
            wrm_weight_boost_w1: None,
            wrm_weight_boost_w2: None,
            score_drop_abs: None,
            score_clamp_abs: None,
            init_from: None,
            init_preset: None,
            test_data: None,
            test_positions: None,
            test_tail_positions: None,
            tf32: false,
            ft_fp16: false,
            ft_fp16_out: false,
            fp16_opt_state: false,
            threads: 1,
        }
    }

    #[test]
    fn run_writes_experiment_json() {
        // `run` に ExperimentLogger を渡すと、run 完了時に status "completed" の
        // experiment.json が書かれ、history が superbatch 数、checkpoints が
        // 保存した .bin/.ckpt 名で埋まることを検証する。
        use crate::experiment::{DataInfo, ExperimentDoc, ExperimentLogger};

        let dir = std::env::temp_dir().join(format!(
            "nnue-train-exp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let json_path = dir.join("experiments").join("exp-test.json");

        let params = experiment_params();
        let data = DataInfo {
            name: "sample.psv".to_string(),
            positions: 1_000,
            total_positions: 0,
            dataset_passes: 0.0,
        };
        let doc = ExperimentDoc::new(
            "exp-test".to_string(),
            "exp".to_string(),
            1_747_000_000,
            None,
            "nnue-train --data sample.psv".to_string(),
            None,
            params,
            data,
            Some(nnue_format::layerstack_weights::FV_SCALE),
        );
        let mut logger = ExperimentLogger::new(json_path.clone(), doc);

        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = base_cfg(); // start 1, end 3, save_rate 2
        let mut backend = MockBackend::new();

        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            Some(&mut logger),
        )
        .expect("run ok");

        let raw = std::fs::read_to_string(&json_path).expect("experiment.json written");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(v["status"], "completed");
        // start 1 .. end 3 → history 3 点。
        assert_eq!(v["history"].as_array().expect("history array").len(), 3);
        // save_rate 2 → sb 2, sb 3 で checkpoint。各 .bin + .ckpt = 4 件。
        assert_eq!(
            v["checkpoints"]
                .as_array()
                .expect("checkpoints array")
                .len(),
            4
        );
        assert_eq!(v["results"]["interrupted"], false);
        // resume checkpoint には logger の run id が渡される (`*.ckpt` の producer
        // run id 埋め込み経路)。sb 2 / sb 3 の 2 回とも実験ログの id と一致する。
        assert_eq!(backend.resume_run_ids, vec!["exp-test", "exp-test"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_with_test_data_runs_held_out_validation_each_superbatch() {
        // `--test-data` 相当: cfg.test_data を渡すと各 superbatch 末に
        // validate_step が呼ばれる。training step 数は影響を受けない。
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            test_data: Some(sample_psv_path()),
            test_positions: 8, // batch_size 8 → 満タン検証 batch 1 個
            ..base_cfg()       // start 1, end 3, batches/sb 2
        };
        let mut backend = MockBackend::new();
        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            None,
        )
        .expect("run ok");
        // 3 superbatch、各 superbatch 末に検証 batch 1 個 = validate_step 3 回。
        assert_eq!(backend.validate_steps, 3);
        // training step (3 sb × 2 batch/sb) は held-out validation の有無に依らない。
        assert_eq!(backend.steps, 6);
    }

    #[test]
    fn run_with_test_data_records_validation_in_experiment_json() {
        // held-out validation の結果が run → record_superbatch → experiment.json
        // の history / results まで配線されていることを検証する。
        use crate::experiment::{DataInfo, ExperimentDoc, ExperimentLogger};

        let dir = std::env::temp_dir().join(format!(
            "nnue-train-exp-val-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let json_path = dir.join("experiments").join("exp-val.json");
        let doc = ExperimentDoc::new(
            "exp-val".to_string(),
            "exp".to_string(),
            1_747_000_000,
            None,
            "nnue-train --data sample.psv --test-data sample.psv".to_string(),
            None,
            experiment_params(),
            DataInfo {
                name: "sample.psv".to_string(),
                positions: 1_000,
                total_positions: 0,
                dataset_passes: 0.0,
            },
            Some(nnue_format::layerstack_weights::FV_SCALE),
        );
        let mut logger = ExperimentLogger::new(json_path.clone(), doc);

        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            test_data: Some(sample_psv_path()),
            test_positions: 8,
            ..base_cfg()
        };
        let mut backend = MockBackend::new();
        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            Some(&mut logger),
        )
        .expect("run ok");

        let raw = std::fs::read_to_string(&json_path).expect("experiment.json written");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        // history の各 superbatch に test_loss / test_accuracy が載る。
        let h0 = &v["history"][0];
        assert!(
            h0["test_loss"].is_number(),
            "history[0].test_loss missing: {h0}"
        );
        assert!(
            h0["test_accuracy"].is_number(),
            "history[0].test_accuracy missing: {h0}"
        );
        // best_test_loss は results に集約される。
        assert!(v["results"]["best_test_loss"].is_number());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_without_test_data_skips_validation() {
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 1.0,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = base_cfg(); // test_data: None
        let mut backend = MockBackend::new();
        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            None,
        )
        .expect("run ok");
        assert_eq!(
            backend.validate_steps, 0,
            "--test-data 未指定なら held-out validation は走らない"
        );
    }

    #[test]
    fn run_without_fp16_clamp_monitor_skips_clamp_read() {
        // `--monitor-fp16-clamps` 未指定 (= base_cfg().monitor_fp16_clamps == false) なら
        // backend の `read_fp16_clamp_count` を呼ばない (D2H copy / log line 共に skip)。
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 1.0,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = base_cfg(); // monitor_fp16_clamps: false
        let mut backend = MockBackend::new();
        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            None,
        )
        .expect("run ok");
        assert_eq!(
            backend.clamp_reads, 0,
            "monitor_fp16_clamps=false なら read_fp16_clamp_count は呼ばれない"
        );
    }

    #[test]
    fn run_with_fp16_clamp_monitor_reads_each_superbatch() {
        // `--monitor-fp16-clamps` 指定で各 sb 末に backend の `read_fp16_clamp_count`
        // を 1 回呼ぶ。base_cfg は start=1 / end=3 で 3 sb 走る。
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 1.0,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            monitor_fp16_clamps: true,
            ..base_cfg()
        };
        let mut backend = MockBackend::new();
        run(
            &mut backend,
            &sample_psv_path(),
            &progress,
            &lr,
            &wdl,
            &cfg,
            None,
        )
        .expect("run ok");
        let expected_sb_count = cfg.end_superbatch - cfg.start_superbatch + 1;
        assert_eq!(
            backend.clamp_reads, expected_sb_count,
            "monitor_fp16_clamps=true なら sb 数だけ read される"
        );
        // MockBackend は train_step ごとに (count, elems) を (+7, +100) ずつ
        // 累積する。run 完了時 backend.steps = sb 数 × batches/sb で、最終 D2H
        // read は正確に (7 * steps, 100 * steps) を返さなければならない。
        let expected_count = 7 * backend.steps as u64;
        let expected_elems = 100 * backend.steps as u64;
        let (final_count, final_elems) = backend
            .read_fp16_clamp_count()
            .expect("read_fp16_clamp_count ok");
        assert_eq!(final_count, expected_count);
        assert_eq!(final_elems, expected_elems);
    }

    #[test]
    fn config_validate_rejects_test_positions_zero_with_test_data() {
        // test_data 指定時に test_positions=0 は reject (満タン batch を作れない)。
        assert!(
            TrainingConfig {
                test_data: Some(sample_psv_path()),
                test_positions: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                test_data: Some(sample_psv_path()),
                test_positions: 8,
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn config_validate_rejects_test_data_and_tail_together() {
        // 同時指定は held-out source 二重で意味不明 → error。
        let err = TrainingConfig {
            test_data: Some(sample_psv_path()),
            test_positions: 8,
            test_tail_positions: Some(16),
            ..base_cfg()
        }
        .validate()
        .expect_err("must reject simultaneous test_data + test_tail_positions");
        assert!(err.to_string().contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn config_validate_rejects_zero_tail() {
        let err = TrainingConfig {
            test_tail_positions: Some(0),
            test_positions: 8,
            ..base_cfg()
        }
        .validate()
        .expect_err("test_tail_positions == 0 should error");
        assert!(err.to_string().contains(">= 1"), "got: {err}");
    }

    #[test]
    fn config_validate_rejects_zero_test_positions_with_tail() {
        let err = TrainingConfig {
            test_tail_positions: Some(8),
            test_positions: 0,
            ..base_cfg()
        }
        .validate()
        .expect_err("test_positions must be >= 1 when tail is set");
        assert!(err.to_string().contains("test_positions"), "got: {err}");
    }

    #[test]
    fn config_validate_accepts_tail_with_positive_test_positions() {
        assert!(
            TrainingConfig {
                test_tail_positions: Some(16),
                test_positions: 8,
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn keep_raw_checkpoints_prunes_oldest() {
        // `--keep-checkpoints N` 相当。end_superbatch=6, save_rate=1 で
        // 6 個の .ckpt が書かれるが keep=2 なら直近 2 個 (sb 5, 6) だけ残る。
        // (MockBackend は実 file を書かないので、テスト用に空 file を実 dir に置いて
        //  prune ロジックを exercise する。)
        let dir = std::env::temp_dir().join(format!(
            "nnue-train-trainer-prune-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");

        // 既存 .ckpt と .bin を散らかしておく (`.bin` は prune 対象外であることを確認)。
        for sb in 1..=4usize {
            std::fs::write(dir.join(format!("net-{sb}.ckpt")), b"x").unwrap();
            std::fs::write(dir.join(format!("net-{sb}.bin")), b"x").unwrap();
        }
        // 別 net_id の .ckpt は触られないこと。
        std::fs::write(dir.join("other-1.ckpt"), b"x").unwrap();
        // 数値でない名前は無視されること。
        std::fs::write(dir.join("net-foo.ckpt"), b"x").unwrap();

        prune_old_raw_checkpoints(&dir, "net", 2);

        // sb 3, 4 だけ残る (sb 1, 2 削除)。
        assert!(
            !dir.join("net-1.ckpt").exists(),
            "net-1.ckpt should be pruned"
        );
        assert!(
            !dir.join("net-2.ckpt").exists(),
            "net-2.ckpt should be pruned"
        );
        assert!(dir.join("net-3.ckpt").exists(), "net-3.ckpt should be kept");
        assert!(dir.join("net-4.ckpt").exists(), "net-4.ckpt should be kept");
        // .bin は全部残る。
        for sb in 1..=4usize {
            assert!(
                dir.join(format!("net-{sb}.bin")).exists(),
                "net-{sb}.bin kept"
            );
        }
        // 別 net_id / 非数値名は無傷。
        assert!(dir.join("other-1.ckpt").exists());
        assert!(dir.join("net-foo.ckpt").exists());

        // keep >= 個数 のときは何も消さない。
        prune_old_raw_checkpoints(&dir, "net", 10);
        assert!(dir.join("net-3.ckpt").exists());
        assert!(dir.join("net-4.ckpt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_raw_checkpoints_sorts_numerically_not_lexically() {
        // superbatch 番号は parse 済 `usize` で降順 sort される (string sort
        // への regression 検出)。
        // 9, 10, 11 を keep=2 で prune したとき、数値 sort なら最古の 9 が消え 10/11 が残る。
        // lexical (string) sort に regress すると "10" < "11" < "9" となり 11 を誤って消す。
        let dir = std::env::temp_dir().join(format!(
            "nnue-train-trainer-prune-numeric-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");

        for sb in [9usize, 10, 11] {
            std::fs::write(dir.join(format!("net-{sb}.ckpt")), b"x").unwrap();
        }

        prune_old_raw_checkpoints(&dir, "net", 2);

        assert!(
            !dir.join("net-9.ckpt").exists(),
            "net-9.ckpt should be pruned (smallest superbatch by numeric sort)"
        );
        assert!(
            dir.join("net-10.ckpt").exists(),
            "net-10.ckpt should be kept (lexical sort would wrongly prune it)"
        );
        assert!(
            dir.join("net-11.ckpt").exists(),
            "net-11.ckpt should be kept (newest)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_data_file_errors_instead_of_looping_forever() {
        // 空 file は即 EOF → epoch wrap が無限ループする危険があるが、dataloader 内の
        // `PsvEpochReader` の `MAX_BARREN_PASSES` ガードで error にして抜け、worker が
        // error slot 経由で run に伝える。
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 1.0,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            end_superbatch: 1,
            batches_per_superbatch: 1,
            threads: 1,
            ..base_cfg()
        };

        let tmp = std::env::temp_dir().join(format!(
            "nnue-train-trainer-empty-{}.psv",
            std::process::id()
        ));
        std::fs::write(&tmp, b"").expect("write empty psv");

        let mut backend = MockBackend::new();
        let result = run(&mut backend, &tmp, &progress, &lr, &wdl, &cfg, None);
        let _ = std::fs::remove_file(&tmp);

        let err = result.expect_err("empty data file should error, not hang");
        assert!(
            err.to_string().contains("no usable positions"),
            "got: {err}"
        );
        assert_eq!(backend.steps, 0, "no step should run on an empty data file");
    }

    #[test]
    fn config_validate_rejects_bad_ranges() {
        assert!(base_cfg().validate().is_ok());
        assert!(
            TrainingConfig {
                start_superbatch: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                start_superbatch: 5,
                end_superbatch: 4,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                batch_size: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                batches_per_superbatch: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                save_rate: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                keep_raw_checkpoints: Some(0),
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                keep_raw_checkpoints: Some(3),
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Sigmoid { scale: 0.0 },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Sigmoid { scale: f32::NAN },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: 340.0,
                    in_offset: 270.0,
                    target_offset: 270.0,
                    target_scaling: 380.0,
                    pow_exp: 2.0,
                    qp_asymmetry: 0.0,
                    weight_boost_w1: 0.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 0.0,
                    in_scaling: 340.0,
                    in_offset: 270.0,
                    target_offset: 270.0,
                    target_scaling: 380.0,
                    pow_exp: 2.0,
                    qp_asymmetry: 0.0,
                    weight_boost_w1: 0.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: -1.0,
                    in_offset: 270.0,
                    target_offset: 270.0,
                    target_scaling: 380.0,
                    pow_exp: 2.0,
                    qp_asymmetry: 0.0,
                    weight_boost_w1: 0.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        // target_scaling <= 0 は WRM target sigmoid を壊すので reject。
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: 340.0,
                    in_offset: 270.0,
                    target_offset: 270.0,
                    target_scaling: 0.0,
                    pow_exp: 2.0,
                    qp_asymmetry: 0.0,
                    weight_boost_w1: 0.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        // in_offset は任意の有限値だが非有限値は reject (target_offset と同じ扱い)。
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: 340.0,
                    in_offset: f32::NAN,
                    target_offset: 270.0,
                    target_scaling: 380.0,
                    pow_exp: 2.0,
                    qp_asymmetry: 0.0,
                    weight_boost_w1: 0.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        // 拡張パラメータ (pow_exp / qp_asymmetry / weight boost) を有効化した WRM は ok。
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: 340.0,
                    in_offset: 270.0,
                    target_offset: 270.0,
                    target_scaling: 380.0,
                    pow_exp: 2.5,
                    qp_asymmetry: 0.3,
                    weight_boost_w1: 1.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
        // pow_exp < 1 は grad が err→0 で発散するので reject。
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: 340.0,
                    in_offset: 270.0,
                    target_offset: 270.0,
                    target_scaling: 380.0,
                    pow_exp: 0.5,
                    qp_asymmetry: 0.0,
                    weight_boost_w1: 0.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        // 拡張パラメータの非有限値は reject。
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: 340.0,
                    in_offset: 270.0,
                    target_offset: 270.0,
                    target_scaling: 380.0,
                    pow_exp: 2.0,
                    qp_asymmetry: f32::NAN,
                    weight_boost_w1: 0.0,
                    weight_boost_w2: 0.5
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        // 拡張パラメータの負値は reject: qp_asymmetry<0 (asym<=0 で勾配反転)、weight
        // boost の w1<0 (de-emphasis) / w2<0 (weight base 0 で inf)。
        for (qp, w1, w2) in [
            (-2.0_f32, 0.0_f32, 0.5_f32),
            (0.0, -1.0, 0.5),
            (0.0, 0.0, -0.5),
        ] {
            assert!(
                TrainingConfig {
                    loss: LossKind::Wrm {
                        nnue2score: 600.0,
                        in_scaling: 340.0,
                        in_offset: 270.0,
                        target_offset: 270.0,
                        target_scaling: 380.0,
                        pow_exp: 2.0,
                        qp_asymmetry: qp,
                        weight_boost_w1: w1,
                        weight_boost_w2: w2
                    },
                    ..base_cfg()
                }
                .validate()
                .is_err(),
                "qp={qp} w1={w1} w2={w2} must be rejected"
            );
        }
        // score-drop-abs は >= 1。0 や負値は「全 position を drop」になるので reject。
        assert!(
            TrainingConfig {
                score_drop_abs: Some(0),
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                score_drop_abs: Some(-1),
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                score_drop_abs: Some(32000),
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
        // score-clamp-abs は >= 1 (上限 i16::MAX は型で保証)。
        for bad in [0, -1] {
            assert!(
                TrainingConfig {
                    score_clamp_abs: Some(bad),
                    ..base_cfg()
                }
                .validate()
                .is_err(),
                "score_clamp_abs {bad} must be rejected"
            );
        }
        assert!(
            TrainingConfig {
                score_clamp_abs: Some(4144),
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
        // clamp >= drop は「drop 生存者に clamp が一度も発火しない」silent no-op
        // 構成なので reject。clamp < drop は OK。
        assert!(
            TrainingConfig {
                score_drop_abs: Some(3000),
                score_clamp_abs: Some(4144),
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                score_drop_abs: Some(32000),
                score_clamp_abs: Some(4144),
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn format_hms_renders_expected_buckets() {
        assert_eq!(format_hms(0.0), "0s");
        assert_eq!(format_hms(42.0), "42s");
        assert_eq!(format_hms(125.0), "2m05s");
        assert_eq!(format_hms(3661.0), "1h01m01s");
        assert_eq!(format_hms(f32::NAN as f64), "??");
        assert_eq!(format_hms(-1.0), "??");
    }
}
