use std::io;
use std::path::{Component, Path, PathBuf};

use clap::{Args, Parser, Subcommand};
#[cfg(any(feature = "gpu", test))]
use nnue_format::ArchKind;

use crate::arch::*;

fn parse_positive_i32(value: &str) -> Result<i32, String> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| "fv_scale must be an integer greater than zero".to_string())?;
    if parsed <= 0 {
        return Err("fv_scale must be greater than zero".to_string());
    }
    Ok(parsed)
}

// ===========================================================================
// `--data` wildcard 展開
//
// `--data` は複数 PSV file を 1 本の training stream として扱うため、
// `*` / `?` を含む path を shell 風 glob として展開する (shell が展開済の
// 場合はここに来ず単一 path のまま通る; クオートして渡した場合や、shell の
// argv 上限を避けたい場合に有効)。追加の依存 crate は増やさず本 file 内で
// 完結させる。
// ===========================================================================

/// 単一 path segment (path 区切り文字を含まない) に対する shell 風 glob
/// pattern match。`*` は 0 文字以上の任意文字列、`?` は任意の 1 文字に
/// マッチする。古典的な動的計画法による wildcard match で、`pattern` /
/// `name` の長さの積に比例した時間で判定する (指数爆発しない)。
fn glob_match_segment(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let s: Vec<char> = name.chars().collect();
    let mut dp = vec![vec![false; s.len() + 1]; p.len() + 1];
    dp[0][0] = true;
    for i in 1..=p.len() {
        if p[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        }
    }
    for i in 1..=p.len() {
        for j in 1..=s.len() {
            dp[i][j] = match p[i - 1] {
                '*' => dp[i - 1][j] || dp[i][j - 1],
                '?' => dp[i - 1][j - 1],
                c => c == s[j - 1] && dp[i - 1][j - 1],
            };
        }
    }
    dp[p.len()][s.len()]
}

/// `pattern` の1 path component (`Component::Normal`) が `*` / `?` を含むか。
fn component_has_wildcard(comp: &Component<'_>) -> bool {
    matches!(comp, Component::Normal(seg) if seg.to_string_lossy().contains(['*', '?']))
}

/// `--data` の値を wildcard 展開する。`pattern` が `*` / `?` を含まないときは
/// 展開せず `[pattern]` をそのまま返す (存在確認は行わない: 従来どおり後続の
/// file open で error にする)。
///
/// `*` / `?` を含む場合は shell 風 glob として展開し、一致した path を
/// (安定した学習順序のため) 昇順ソートして返す。中間 path component の
/// wildcard (`data/*/train.psv` 等) にも対応するが、再帰的ディレクトリ探索
/// (`**`) は対応しない。1 件も一致しなければ error にする (誤って空データで
/// 学習が始まるのを防ぐ)。dotfile は shell 慣例と同様、pattern 側の
/// component が `.` から始まらない限りマッチしない。
pub(crate) fn expand_data_glob(pattern: &Path) -> io::Result<Vec<PathBuf>> {
    let components: Vec<Component<'_>> = pattern.components().collect();
    if !components.iter().any(component_has_wildcard) {
        return Ok(vec![pattern.to_path_buf()]);
    }

    let mut current: Vec<PathBuf> = vec![PathBuf::new()];
    for comp in &components {
        if component_has_wildcard(comp) {
            let Component::Normal(seg) = comp else {
                unreachable!("component_has_wildcard only matches Component::Normal");
            };
            let seg_str = seg.to_string_lossy();
            let mut next = Vec::new();
            for base in &current {
                let dir = if base.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    base.clone()
                };
                let entries = std::fs::read_dir(&dir).map_err(|e| {
                    io::Error::other(format!(
                        "--data '{}': cannot list directory {} while expanding wildcard: {e}",
                        pattern.display(),
                        dir.display(),
                    ))
                })?;
                for entry in entries {
                    let entry = entry?;
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with('.') && !seg_str.starts_with('.') {
                        continue; // shell 慣例: 明示しない限り dotfile は除外
                    }
                    if glob_match_segment(&seg_str, &name_str) {
                        let mut matched = base.clone();
                        matched.push(&name);
                        next.push(matched);
                    }
                }
            }
            current = next;
        } else {
            for base in current.iter_mut() {
                base.push(comp.as_os_str());
            }
        }
    }

    current.sort();
    if current.is_empty() {
        return Err(io::Error::other(format!(
            "--data '{}' matched no files",
            pattern.display()
        )));
    }
    Ok(current)
}

#[cfg(test)]
mod data_glob_tests {
    use super::*;

    #[test]
    fn glob_match_segment_star_and_question_mark() {
        assert!(glob_match_segment("*.psv", "train.psv"));
        assert!(glob_match_segment("*.psv", ".psv"));
        assert!(!glob_match_segment("*.psv", "train.hcpe"));
        assert!(glob_match_segment("shard-???.psv", "shard-001.psv"));
        assert!(!glob_match_segment("shard-???.psv", "shard-1.psv"));
        assert!(glob_match_segment("*", "anything.psv"));
    }

    #[test]
    fn expand_data_glob_passthrough_without_wildcard() {
        let path = PathBuf::from("some/exact/file.psv");
        let expanded = expand_data_glob(&path).unwrap();
        assert_eq!(expanded, vec![path]);
    }

    #[test]
    fn expand_data_glob_matches_and_sorts_files() {
        let dir = std::env::temp_dir().join(format!(
            "nnue-train-cli-glob-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        for name in ["shard-002.psv", "shard-000.psv", "shard-001.psv", "notes.txt"] {
            std::fs::write(dir.join(name), b"").expect("write fixture file");
        }

        let pattern = dir.join("shard-*.psv");
        let expanded = expand_data_glob(&pattern).expect("glob should match files");
        let expected = vec![
            dir.join("shard-000.psv"),
            dir.join("shard-001.psv"),
            dir.join("shard-002.psv"),
        ];
        assert_eq!(expanded, expected);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn expand_data_glob_errors_when_no_match() {
        let dir = std::env::temp_dir().join(format!(
            "nnue-train-cli-glob-empty-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let pattern = dir.join("*.psv");
        let err = expand_data_glob(&pattern).expect_err("empty match should error");
        assert!(err.to_string().contains("matched no files"), "got: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ===========================================================================
// CLI (clap)
// ===========================================================================

/// Shogi NNUE trainer.
///
/// Pick the NNUE architecture to train with the `layerstack` / `simple`
/// subcommands. Shared options are global arguments and may be placed before or
/// after the subcommand. Passing `--data <PSV>` runs the training loop;
/// omitting it runs a GPU smoke test that exercises the forward/backward path.
#[derive(Parser, Debug)]
#[command(name = "nnue-train", about = "Shogi NNUE trainer")]
pub(crate) struct Cli {
    /// Training data PSV file (`PackedSfenValue` x N, 40 bytes each). When omitted, runs a GPU
    /// smoke test. Accepts a `*` / `?` wildcard (e.g. `shards/*.psv`) to train on multiple files
    /// as a single concatenated stream, in sorted filename order; quote the pattern so the shell
    /// doesn't expand it first if you rely on this.
    #[arg(long, global = true)]
    pub(crate) data: Option<PathBuf>,

    /// PSV or HCPE file for held-out validation. Pass positions that are never used for
    /// a gradient update, separate from the training `--data`. When set, a
    /// forward-only validation pass runs at the end of each superbatch and
    /// reports test_loss (mean held-out loss) and test_accuracy (agreement
    /// between the output sign and the game result) in the training log and
    /// experiment.json. Used for early detection of divergence and overfitting.
    /// Mutually exclusive with `--test-tail-positions`.
    #[arg(long, global = true, conflicts_with = "test_tail_positions")]
    pub(crate) test_data: Option<PathBuf>,

    /// Reserve the last N positions of `--data` as a same-file held-out
    /// validation set: training reads positions in `[0, file_end - N * 40)`
    /// and validation reads `[file_end - N * 40, file_end)`. Useful when the
    /// training PSV is too large to split into separate files. Mutually
    /// exclusive with `--test-data`. The first `--test-positions` of the
    /// reserved tail are evaluated at the end of each superbatch.
    #[arg(long, global = true, conflicts_with = "test_data")]
    pub(crate) test_tail_positions: Option<u64>,

    /// Number of positions per held-out validation pass. Takes this many
    /// positions from the start of the held-out source (either `--test-data`
    /// or the tail reserved by `--test-tail-positions`) and rounds up to a
    /// whole `--batch-size` multiple to form full batches. Used only when
    /// one of those flags is set.
    #[arg(long, default_value_t = 10000, global = true)]
    pub(crate) test_positions: usize,

    /// Output directory for checkpoints (writes `{net_id}-{superbatch}.bin`).
    #[arg(long, default_value = "checkpoints", global = true)]
    pub(crate) output: PathBuf,

    /// Inference checkpoint format. `yaneuraou` is available only for a
    /// LayerStack trained with `--bucket-mode kingrank9` and writes an SFNN
    /// evaluation file directly.
    #[arg(long, value_enum, default_value_t = OutputFormatArg::Tatara, global = true)]
    pub(crate) output_format: OutputFormatArg,

    /// Network id (used in checkpoint file names).
    #[arg(long, default_value = "rshogi", global = true)]
    pub(crate) net_id: String,

    /// Input feature set. One of: halfkp, halfka-split, halfka-merged,
    /// halfka-hm-split, halfka-hm-merged. Determines the FT input dimension and
    /// the number of active features. The default halfka-hm-merged is
    /// king-symmetric merged HalfKA.
    #[arg(long, default_value = "halfka-hm-merged", global = true)]
    pub(crate) feature_set: String,

    /// `name` field in experiment.json (display name in the experiment-tracking
    /// UI). Defaults to net_id, or `{net_id} (resume @sb{start superbatch})`
    /// when `--resume` is used.
    #[arg(long, global = true)]
    pub(crate) experiment_name: Option<String>,

    /// Number of superbatches to train (runs 1..=superbatches). The default of
    /// 10 is for smoke testing; use a much larger value for real training.
    #[arg(long, default_value_t = 10, global = true)]
    pub(crate) superbatches: usize,

    /// Number of batches per superbatch.
    #[arg(long, default_value_t = 6104, global = true)]
    pub(crate) batches_per_superbatch: usize,

    /// Number of positions per batch. Affects both GPU throughput and training
    /// dynamics. The default of 16384 is for smoke testing.
    #[arg(long, default_value_t = 16384, global = true)]
    pub(crate) batch_size: usize,

    /// Initial learning rate.
    #[arg(long, default_value_t = 8.75e-4, global = true)]
    pub(crate) lr: f32,

    /// Learning-rate schedule shape. The schedule maps a superbatch index to a
    /// learning rate; --lr is the starting (or, for one-cycle, the peak) rate.
    ///
    /// - step (default): multiply --lr by --lr-gamma every --lr-step
    ///   superbatches. Bit-identical to the historical behaviour.
    /// - constant: hold --lr for the whole run (--lr-gamma / --lr-step ignored).
    /// - drop: hold --lr, then multiply by --lr-gamma once after --lr-step
    ///   superbatches.
    /// - linear / cosine / exponential: decay from --lr to --lr-final by
    ///   --lr-final-superbatch (defaults to --superbatches), then hold
    ///   --lr-final. exponential requires --lr-final > 0.
    /// - one-cycle: warm up from --lr/--lr-div-factor to the peak --lr over the
    ///   first --lr-warmup-pct of --superbatches, then cosine-anneal to
    ///   --lr/--lr-div-factor/--lr-final-div-factor.
    #[arg(long, value_enum, default_value_t = LrScheduleArg::Step, global = true)]
    pub(crate) lr_schedule: LrScheduleArg,

    /// LR gamma. For --lr-schedule step, multiplies the LR every --lr-step
    /// superbatches; for drop, the one-shot multiplier applied after --lr-step
    /// superbatches. Ignored by the other schedules.
    #[arg(long, default_value_t = 0.992, global = true)]
    pub(crate) lr_gamma: f32,

    /// LR step. For --lr-schedule step, the superbatch interval at which the LR
    /// is multiplied by --lr-gamma; for drop, the superbatch after which the LR
    /// drops once. Ignored by the other schedules.
    #[arg(long, default_value_t = 1, global = true)]
    pub(crate) lr_step: usize,

    /// Final learning rate for the linear / cosine / exponential decay
    /// schedules. The LR decays from --lr to this value by
    /// --lr-final-superbatch and then holds. exponential requires a value > 0
    /// (it interpolates the LR geometrically). Ignored by the other schedules.
    #[arg(long, default_value_t = 1e-5, global = true)]
    pub(crate) lr_final: f32,

    /// Superbatch by which the linear / cosine / exponential decay reaches
    /// --lr-final. When omitted, the horizon is taken from (in priority order):
    /// the saved horizon in a v5+ --resume checkpoint, else --superbatches.
    /// Passing this flag explicitly always wins, even over a resumed
    /// checkpoint's saved horizon. one-cycle uses the same precedence for its
    /// total horizon but has no explicit flag, so on resume its saved horizon
    /// wins over --superbatches. A checkpoint written before v5 (or by a
    /// schedule without a horizon) carries none, so resume falls back to
    /// --superbatches. Ignored by the other schedules.
    #[arg(long, global = true)]
    pub(crate) lr_final_superbatch: Option<usize>,

    /// Warm up the learning rate over the first N batches of the first
    /// superbatch, ramping from a small fraction of the scheduled LR up to it,
    /// on top of any --lr-schedule. Applies to every schedule except one-cycle
    /// (which carries its own warmup). When omitted, no batch-level warmup.
    #[arg(long, global = true)]
    pub(crate) lr_warmup_steps: Option<usize>,

    /// one-cycle only: fraction of --superbatches spent warming up from the
    /// initial LR to the peak --lr before annealing. Must be in [0.0, 1.0].
    #[arg(long, default_value_t = 0.2, global = true)]
    pub(crate) lr_warmup_pct: f32,

    /// one-cycle only: the initial LR is --lr divided by this factor (the peak
    /// is --lr). Must be >= 1 so the initial LR does not exceed the peak.
    #[arg(long, default_value_t = 25.0, global = true)]
    pub(crate) lr_div_factor: f32,

    /// one-cycle only: the final LR is the initial LR (--lr / --lr-div-factor)
    /// divided by this factor. Must be > 0.
    #[arg(long, default_value_t = 1e4, global = true)]
    pub(crate) lr_final_div_factor: f32,

    /// WDL blend lambda (constant). Mutually exclusive with the linear-taper
    /// pair `--start-wdl` / `--end-wdl`.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) wdl: f32,

    /// Start of a linear WDL lambda taper, used at the first superbatch. Requires
    /// `--end-wdl`; the lambda interpolates linearly from `--start-wdl` to
    /// `--end-wdl` across superbatches. Conflicts with `--wdl`.
    #[arg(long, global = true, conflicts_with = "wdl")]
    pub(crate) start_wdl: Option<f32>,

    /// End of a linear WDL lambda taper, reached at the final superbatch. Requires
    /// `--start-wdl`. Conflicts with `--wdl`.
    #[arg(long, global = true, conflicts_with = "wdl")]
    pub(crate) end_wdl: Option<f32>,

    /// Score scale for the sigmoid loss (`loss_scale = 1 / scale`). On the
    /// layerstack subcommand this is unused when `--win-rate-model` is set (WRM
    /// loss uses the `--wrm-*` scaling instead). The simple trainer always uses
    /// it to derive the exported `fv_scale`, so it stays in effect even under
    /// the WRM there.
    #[arg(long, default_value_t = 290.0, global = true)]
    pub(crate) scale: f32,

    /// Write a checkpoint every `save_rate` superbatches (and at the end).
    #[arg(long, default_value_t = 20, global = true)]
    pub(crate) save_rate: usize,

    /// Exclude positions with `|score| >= score_drop_abs` from the loss.
    #[arg(long, global = true)]
    pub(crate) score_drop_abs: Option<i32>,

    /// Saturate the teacher score of surviving positions to `[-N, N]` (applied
    /// after the `--score-drop-abs` filter when set, so e.g. mate stamps are
    /// dropped, not clamped; without that filter they are clamped into range).
    /// Useful to normalise teacher files whose encode variants clip at
    /// different ceilings. Must be in `[1, 32767]`.
    #[arg(long, global = true, value_parser = clap::value_parser!(i16).range(1..))]
    pub(crate) score_clamp_abs: Option<i16>,

    /// Inject weights from a quantised NNUE binary before training starts
    /// (pretrained start). The optimizer state (m/v/slow/step) is
    /// **reset** — use `--resume` for a true resume (`--init-from` and
    /// `--resume` are mutually exclusive).
    #[arg(long, global = true)]
    pub(crate) init_from: Option<PathBuf>,

    /// Load weights (via --init-from / --resume), evaluate held-out test_loss /
    /// test_accuracy once, and exit without training. Requires held-out data
    /// (--test-tail-positions or --test-data).
    #[arg(long, global = true)]
    pub(crate) eval_only: bool,

    /// Zero a subset of the loaded threat FT rows before eval/train, to measure
    /// that subset's eval contribution (threat net + --init-from only). One of:
    /// all | slider-attacker | step-attacker | bigslider-attacker | defense |
    /// attack | same-class | random:<seed>:<dims>.
    #[arg(long, global = true)]
    pub(crate) threat_ablate: Option<String>,

    /// Print a pair-class L2-norm breakdown of the loaded threat FT weights and
    /// exit (no eval; threat net + --init-from only).
    #[arg(long, global = true)]
    pub(crate) threat_norm_dump: bool,

    /// Resume training by restoring weights + optimizer state
    /// (m/v/slow/step) from a raw checkpoint (`{net_id}-{sb}.ckpt`) — a true
    /// resume. Mutually exclusive with `--init-from` (which injects weights only
    /// and resets the optimizer). When `--start-superbatch` is omitted, resumes
    /// from the superbatch recorded in the checkpoint + 1. A v5+ checkpoint also
    /// stores the LR-schedule horizon; on resume the saved horizon is restored
    /// so the LR curve is reproduced independently of --superbatches (see
    /// --lr-final-superbatch for the full precedence).
    #[arg(long, global = true)]
    pub(crate) resume: Option<PathBuf>,

    /// Superbatch number to start training from (1-indexed, inclusive). When
    /// omitted: with `--resume`, the checkpoint's superbatch + 1; otherwise 1.
    /// An error if outside `1 <= N <= --superbatches` (may be set explicitly to
    /// redo past superbatches on resume).
    #[arg(long, global = true)]
    pub(crate) start_superbatch: Option<usize>,

    /// Keep only the most recent N raw checkpoints (`*.ckpt`) to save disk
    /// space. When omitted, all are kept. Raw state is large (~1.8GB each) and
    /// piles up over long runs, so setting this is recommended. Quantised `.bin`
    /// files (~116MB) are always kept regardless of this setting (inference
    /// artifacts).
    #[arg(long, global = true)]
    pub(crate) keep_checkpoints: Option<usize>,

    /// Use the win-rate-model loss. When set, uses the `loss_wrm` kernel
    /// (applies WRM to both prediction and target); otherwise uses `loss_wdl`
    /// (plain sigmoid-MSE + `--scale`). The net_output scale becomes
    /// `out ≈ cp / --wrm-nnue2score`, matching the scale that quantisation
    /// (`QA=127/QB=64/FV_SCALE=28`) assumes.
    #[arg(long, global = true)]
    pub(crate) win_rate_model: bool,
    /// In-scaling for the WRM prediction side (default 340). Independent of the
    /// target-side scaling (`--wrm-target-scaling`). Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 340.0, global = true)]
    pub(crate) wrm_in_scaling: f32,
    /// Center offset of the WRM prediction win-rate sigmoid, subtracted from the
    /// scaled net score inside `sigmoid((net*nnue2score - offset)/in_scaling)`
    /// (default 270). Independent of the target-side offset
    /// (`--wrm-target-offset`). Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 270.0, global = true)]
    pub(crate) wrm_in_offset: f32,
    /// WRM nnue2score (`scorenet = net_output * --wrm-nnue2score`, default 600).
    /// Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 600.0, global = true)]
    pub(crate) wrm_nnue2score: f32,
    /// Center offset of the WRM target sigmoid (the score at which `target` is
    /// 0.5, default 270). Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 270.0, global = true)]
    pub(crate) wrm_target_offset: f32,
    /// Input scale of the WRM target sigmoid (inverse of steepness, default
    /// 380). The defaults 270/380 are tuned for the chess score distribution;
    /// retune them if your score distribution differs. Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 380.0, global = true)]
    pub(crate) wrm_target_scaling: f32,
    /// Exponent of the WRM error term `|qf - target|^pow_exp` (default 2.0, plain
    /// squared error). nnue-pytorch uses 2.5. Must be >= 1 (the gradient contains
    /// `|err|^(pow_exp-1)`). Used only when `--win-rate-model` is set; the default
    /// 2.0 keeps the loss kernel on its bit-identical squared-error path.
    #[arg(long, default_value_t = 2.0, global = true)]
    pub(crate) loss_pow_exp: f32,
    /// Asymmetric penalty for overprediction (default 0.0 = symmetric; must be
    /// non-negative). When set, positions where the prediction `qf` exceeds the
    /// target are weighted by `1 + qp_asymmetry`. Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) loss_qp_asymmetry: f32,
    /// Weight-boost parameter w1 (default 0.0, must be >= 0). Per-position loss
    /// weight is `1 + (2^w1 - 1) * ((pf-0.5)^2 * pf*(1-pf))^w2`, amplifying
    /// decisive positions; `w1 = 0` gives uniform weight 1 (no boost). The total
    /// loss is then normalised by the sum of weights. Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) loss_weight_boost_w1: f32,
    /// Weight-boost parameter w2 (default 0.5, the exponent in the weight
    /// formula, must be >= 0). Has no effect when `--loss-weight-boost-w1` is 0.
    /// Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 0.5, global = true)]
    pub(crate) loss_weight_boost_w2: f32,
    /// Optimizer: "ranger" (RAdam + lookahead, beta1=0.99), "radam" (rectified
    /// Adam without lookahead, beta1=0.9), or "adamw" (Adam without bias
    /// correction, decoupled weight decay, beta1=0.9). All three share
    /// beta2=0.999 and the per-layer weight clamp. When resuming from a raw
    /// checkpoint, pass the same optimizer as the original run (the checkpoint
    /// stores moment buffers but not the optimizer name).
    #[arg(long, default_value = "ranger", global = true)]
    pub(crate) optimizer: String,
    /// Weight decay coefficient for the optimizer (AdamW-style decoupled
    /// weight decay). The default 0.0 means no decay. A non-zero value slightly
    /// decays the weights of every weight group toward 0 on each step.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) weight_decay: f32,
    /// Weight decay override for the FT param group (the feature-transformer
    /// input weights, plus the PSQT shortcut weights when --psqt is set). When
    /// unset, this group uses the global --weight-decay. The optimizer splits
    /// all trainable parameters into three param groups — FT (input-side
    /// weights), dense (the L1/L1f/L2/L3 hidden-layer weights), and bias (every
    /// layer's bias) — each with its own weight decay and learning-rate
    /// multiplier. Leaving all six per-group flags unset is bit-identical to
    /// the single --weight-decay path.
    #[arg(long, global = true)]
    pub(crate) ft_weight_decay: Option<f32>,
    /// Weight decay override for the dense param group (the L1/L1f/L2/L3
    /// hidden-layer weights). Unset falls back to the global --weight-decay.
    #[arg(long, global = true)]
    pub(crate) dense_weight_decay: Option<f32>,
    /// Weight decay override for the bias param group (every layer's bias).
    /// Unset falls back to the global --weight-decay. Pass `0` to disable
    /// weight decay on biases (the common deep-learning default; opt-in here so
    /// the no-flag path stays bit-identical).
    #[arg(long, global = true)]
    pub(crate) bias_weight_decay: Option<f32>,
    /// Learning-rate multiplier for the FT param group: this group's per-step
    /// learning rate is the scheduled learning rate times this value. Unset
    /// means 1.0 (no scaling). Note: decoupled weight decay scales with the
    /// effective learning rate, so this multiplier also scales the group's
    /// effective weight decay.
    #[arg(long, global = true)]
    pub(crate) ft_lr_mult: Option<f32>,
    /// Learning-rate multiplier for the dense param group: this group's per-step
    /// learning rate is the scheduled learning rate times this value. Unset
    /// means 1.0.
    #[arg(long, global = true)]
    pub(crate) dense_lr_mult: Option<f32>,
    /// Learning-rate multiplier for the bias param group: this group's per-step
    /// learning rate is the scheduled learning rate times this value. Unset
    /// means 1.0.
    #[arg(long, global = true)]
    pub(crate) bias_lr_mult: Option<f32>,
    /// Enable norm loss (per-weight-group L2-norm regularisation, Georgiou et
    /// al. 2021). With the default `false`, the optimizer step is bit-identical
    /// to the baseline. When enabled, each step (just before the optimizer update)
    /// every targeted weight group is nudged so its L2 norm relaxes toward 1
    /// (the oblique manifold): the 2D layer weights per output neuron (FT /
    /// L1f / L1 / L2 / L3), the PSQT shortcut weights per output bucket (when
    /// --psqt is enabled), and 1D biases by their whole-tensor norm. Strength
    /// is set by --norm-loss-factor. An opt-in regularizer whose playing-strength
    /// effect must be confirmed by SPRT.
    #[arg(long, global = true)]
    pub(crate) norm_loss: bool,
    /// Norm loss strength (only used when --norm-loss is set). Each step every
    /// targeted weight w is multiplied by
    /// `1 - lr * 2 * factor * (1 - 1 / (||w_group||_2 + eps))`. The default 1e-4
    /// matches the Ranger21 reference.
    #[arg(long, default_value_t = 1e-4, global = true)]
    pub(crate) norm_loss_factor: f32,
    /// Number of dataloader prefetch workers. Each worker does PSV parsing +
    /// HalfKA_hm sparse extraction + progress8kpabs bucket computation in a
    /// single `decode()` call and supplies positions ahead of time. `1` gives
    /// deterministic sequential reads; `>= 2` parses in parallel (position order
    /// within an epoch is non-deterministic, which is fine for training).
    #[arg(long, default_value_t = 16, global = true)]
    pub(crate) threads: usize,

    /// Fast mode that runs the FT weight (`ft_w`) forward pass through an FP16
    /// mirror. With the default `false`, it is bit-identical to the FP32 path.
    /// `true` halves the weight DRAM bandwidth of `sparse_ft_forward`, but
    /// quantisation error may shift playing strength (an opt-in option for
    /// quick, fast training; default OFF until production quality is confirmed
    /// by SPRT).
    ///
    /// FT weights stay small in practice — small-scale initialization plus
    /// AdamW weight decay keep the master FP32 magnitude orders of magnitude
    /// below the FP16 finite range (`|x| <= 65504`), so the mirror conversion is
    /// not expected to overflow to ±inf under normal training. There is no hard
    /// clamp on FT, so `--ft-fp16` remains an opt-in precision trade-off.
    #[arg(long, global = true)]
    pub(crate) ft_fp16: bool,

    /// Fast mode that keeps the feature transformer (FT) optimizer state in
    /// FP16. With the default `false`, it is bit-identical to the FP32 path.
    ///
    /// The FT is the largest layer in this network, and its optimizer update is
    /// memory-bandwidth bound on the state read/write. Halving the precision of
    /// the state reduces the memory traffic of the optimizer step and raises
    /// training throughput. The state values are extremely small, so they are
    /// multiplied by a fixed factor to bring them into the FP16 usable range
    /// before being stored.
    ///
    /// An independent flag from `--ft-fp16` / `--ft-fp16-out`. Quantisation
    /// error may shift playing strength, so it is default OFF and production
    /// quality is not guaranteed until confirmed by SPRT (an opt-in option for
    /// sanity checks and quick, fast training).
    #[arg(long, global = true)]
    pub(crate) fp16_opt_state: bool,

    /// Shortcut to opt into all four risky speed flags at once. Effectively
    /// turns on `--ft-fp16` / `--fp16-opt-state` / (on the subcommand)
    /// `--ft-fp16-out` / `--tf32` together (OR-combined with the individual
    /// flags; works with both subcommands).
    ///
    /// Default OFF (all flags OFF gives a pure FP32, bit-identical path). When
    /// set, the effective values are expanded in the startup log
    /// (`[train] --all-optim → ft_fp16=true ft_fp16_out=true fp16_opt_state=true
    /// tf32=true`) to keep experiment.json reproducible. Default OFF because
    /// quantisation / TF32 error may shift playing strength.
    ///
    /// For fine-grained control (turning on only some), do not use this flag;
    /// list the four individual flags instead.
    #[arg(long, global = true)]
    pub(crate) all_optim: bool,

    /// Log the number of FP16 clamp events (`|x| > 65504` cap to `±65504`) in the
    /// FT activation backward kernels (the `--ft-fp16-out` write path) at the end
    /// of every superbatch. Used to gauge how often the loss-scaled gradient
    /// saturates the FP16 finite range — a high rate suggests the loss scale
    /// should be retuned, as systematic clamping caps gradient magnitudes and
    /// can shift playing strength.
    ///
    /// The clamp counter is always active in the FP16 path; this flag only gates
    /// the host-side D2H read and log line (`[fp16-clamp] sb=... clamps=...
    /// delta=... elems=... ratio=...`). With `--ft-fp16-out` off, the counter
    /// stays at zero because the clamp kernels are not launched.
    #[arg(long, global = true)]
    pub(crate) monitor_fp16_clamps: bool,

    /// Log a histogram of the real active-feature count per position (the value
    /// returned by the feature extractor before `-1` padding) at the end of every
    /// superbatch. Used to check how much of the `max_active` sparse capacity is
    /// actually used and whether any feature set (e.g. with threats appended)
    /// pushes counts toward the cap.
    ///
    /// Each superbatch prints a one-line summary over the cumulative histogram
    /// (`[active-hist] sb=... positions=... mean=... p50=... p90=... p99=...
    /// max=...`); at the end of training the full non-zero histogram is dumped
    /// once (`[active-hist] count[<active>]=<n>`). Off by default: when unset the
    /// dataloader allocates no histogram and runs no counting code on the hot
    /// path. Works for any feature set (threats on or off).
    #[arg(long, global = true)]
    pub(crate) monitor_active_features: bool,

    /// Override the feature-transformer (L0) weight initialiser. Applies to a
    /// fresh run; ignored when `--init-from` / `--resume` loads weights.
    ///
    /// Defaults differ per architecture: `layerstack` initialises the FT with
    /// `uniform:fanin` (half-width `sqrt(1 / fan_in)`), while `simple` uses
    /// `[-0.01, 0.01]` uniform. Every other weight defaults to `[-0.01, 0.01]`
    /// uniform in both architectures.
    ///
    /// Grammar: `zero`, `<uniform|normal>:abs:<value>`, or
    /// `<uniform|normal>:fanin[:<gain>[:<effective>]]` where the magnitude is
    /// `sqrt(gain / effective_or_fan_in)` (half-width for uniform, std for
    /// normal). Examples: `uniform:fanin`, `normal:fanin:2:32`
    /// (`sqrt(2/32) = 0.25`), `uniform:abs:0.01`. Applies to the weight only;
    /// the bias keeps the default.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_ft: Option<nnue_train::init::LayerInitOverride>,

    /// Override the L1 weight initialiser. Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l1: Option<nnue_train::init::LayerInitOverride>,

    /// Override the shared factorised L1f weight initialiser (layerstack only).
    /// Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l1f: Option<nnue_train::init::LayerInitOverride>,

    /// Override the L2 weight initialiser. Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l2: Option<nnue_train::init::LayerInitOverride>,

    /// Override the L3 (output) weight initialiser. Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l3: Option<nnue_train::init::LayerInitOverride>,

    /// FT factorizer (training-time virtual features). **Default ON.** Pass
    /// `--no-ft-factorize` to disable. Supported by both the `layerstack` and
    /// `simple` trainers.
    ///
    /// The FT weight table gains virtual piece-input rows for piece values independent
    /// of the king position. Each virtual row accumulates the gradients of
    /// every real row sharing its piece-input ordinal, so rarely visited king-square
    /// cells inherit a sensible shared prior instead of staying near their
    /// initial values. The virtual rows are folded into the real rows when the
    /// quantised `.bin` is saved, so
    /// the exported net is identical in shape to a non-factorized net and
    /// inference engines need no changes.
    ///
    /// The sparse input stream is unchanged (the active-feature count stays
    /// the base value); virtual rows are wired through two dense kernels per
    /// optimizer step (a forward weight fold and a backward gradient
    /// reduction), costing a single-digit percentage of training throughput
    /// and zero inference cost.
    ///
    /// This flag is accepted for explicitness/back-compat but is redundant
    /// with the default. It is auto-disabled (logged at startup) only by
    /// `--init-from` (a quantised `.bin` has no virtual rows to initialise
    /// from); resume across the resulting on/off is rejected (checkpoint
    /// dimensions differ). On the layerstack trainer it coexists with `--psqt`
    /// (the PSQT shortcut rows share the same fold), `--threat-profile`, and
    /// `--effect-bucket`; the `simple` trainer has no such modifiers, so it
    /// always shares one virtual row per piece input.
    #[arg(
        long = "ft-factorize",
        global = true,
        overrides_with = "no_ft_factorize"
    )]
    pub(crate) ft_factorize: bool,

    /// Disable the FT factorizer (it is ON by default; see `--ft-factorize`).
    /// Use this to train the non-factorized network. Only `--init-from`
    /// auto-disables it otherwise.
    #[arg(
        long = "no-ft-factorize",
        global = true,
        overrides_with = "ft_factorize"
    )]
    pub(crate) no_ft_factorize: bool,

    /// Subcommand selecting the NNUE architecture to train (`layerstack` / `simple`).
    #[command(subcommand)]
    pub(crate) arch: ArchCommand,
}

impl Cli {
    /// FT factorizer の実効 ON/OFF (default ON、`--no-ft-factorize` で OFF)。
    /// `--ft-factorize` は back-compat の明示 ON で、`overrides_with` により
    /// command-line 上で後勝ちする。`--init-from` との排他は呼び出し側
    /// (`run_training` / `run_simple_training`) が auto-suppress で解決するため、
    /// ここには含めない (この値は「ユーザーが factorizer を望むか」だけを表す)。
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn ft_factorize_enabled(&self) -> bool {
        !self.no_ft_factorize
    }
}

/// `--lr-schedule` の選択肢。lib 側 schedule 型への runtime selection。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum LrScheduleArg {
    #[default]
    Step,
    Constant,
    Drop,
    Linear,
    Cosine,
    Exponential,
    #[value(name = "one-cycle")]
    OneCycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum OutputFormatArg {
    #[default]
    Tatara,
    Yaneuraou,
}

impl From<OutputFormatArg> for nnue_train::trainer::OutputFormat {
    fn from(value: OutputFormatArg) -> Self {
        match value {
            OutputFormatArg::Tatara => Self::Tatara,
            OutputFormatArg::Yaneuraou => Self::Yaneuraou,
        }
    }
}

/// `--router-mode` の選択肢。`router` bucket mode でのみ意味を持つ (`bucket_mode
/// != router` では無視される)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum RouterModeArg {
    /// oracle ターゲット分布 (`--top-k` で hard-EM / soft-EM / Top-K Hard
    /// Routing を切替) との cross entropy を最小化する従来の EM 手続き。
    #[default]
    #[value(name = "hard-em")]
    HardEm,
    /// oracle ターゲット分布を経由しない、通常の誤差逆伝播。
    Backprop,
}

impl From<RouterModeArg> for shogi_features::router_kpabs::RouterMode {
    fn from(value: RouterModeArg) -> Self {
        match value {
            RouterModeArg::HardEm => Self::HardEm,
            RouterModeArg::Backprop => Self::Backprop,
        }
    }
}

/// `--ft-fp16-out` が `--ft-fp16` を要求する制約を **実効値** (`--all-optim` の含意込み)
/// で検証する。`true` を返したら制約違反 = error (FT activation FP16 が ON だが
/// FT weight FP16 が OFF)。
///
/// `--all-optim` は `--ft-fp16` / `--ft-fp16-out` の双方を ON 相当にするため、`--all-optim`
/// が指定されていれば制約は常に満たされる (両 flag が実効 ON)。よって制約違反は
/// 「`--ft-fp16-out` が raw 指定されていて、`--all-optim` も無く、`--ft-fp16` も raw 指定
/// されていない」ときのみ。これにより `--all-optim --ft-fp16-out` (冗長指定) を
/// false-positive reject しない。
#[cfg(any(feature = "gpu", test))]
pub(crate) fn ft_fp16_out_missing_ft_fp16(
    ft_fp16_out_raw: bool,
    ft_fp16_raw: bool,
    all_optim: bool,
) -> bool {
    ft_fp16_out_raw && !all_optim && !ft_fp16_raw
}

/// 学習対象の NNUE アーキを選ぶサブコマンド。アーキ固有の引数を持つ。
#[derive(Subcommand, Debug)]
pub(crate) enum ArchCommand {
    /// Bucketed LayerStack architecture (FT → L1 → L2; layer dimensions set by --ft-out / --l1 / --l2).
    #[command(name = "layerstack")]
    LayerStack(LayerstackArgs),
    /// Simple 4-layer dense architecture (no buckets / PSQT / skip).
    Simple(SimpleArgs),
    /// Run reproducible end-to-end training benchmarks from TOML configuration.
    #[command(name = "bench-pos")]
    BenchPos(BenchPosArgs),
    /// Run the fixed native CUDA throughput benchmark and write a JSON report.
    #[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
    #[command(name = "native-bench")]
    NativeBench(NativeBenchArgs),
}

impl ArchCommand {
    /// サブコマンドに対応する [`ArchKind`]。
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn kind(&self) -> ArchKind {
        match self {
            ArchCommand::LayerStack(_) => ArchKind::LayerStack,
            ArchCommand::Simple(_) => ArchKind::Simple,
            ArchCommand::BenchPos(_) => {
                unreachable!("bench-pos does not select a training architecture")
            }
            #[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
            ArchCommand::NativeBench(_) => {
                unreachable!("native-bench does not select a training architecture")
            }
        }
    }
}

#[derive(Args, Debug)]
pub(crate) struct BenchPosArgs {
    /// Tracked benchmark profile containing measurement parameters and cases.
    #[arg(long, default_value = "bench-pos.toml")]
    pub(crate) profile: PathBuf,

    /// Gitignored machine-local paths and hardware settings.
    #[arg(long, default_value = "bench-pos.local.toml")]
    pub(crate) local_config: PathBuf,

    /// Run only the named case. Repeat this option to select multiple cases.
    #[arg(long = "case", value_name = "ID")]
    pub(crate) cases: Vec<String>,

    /// Directory receiving JSON reports and per-run logs.
    #[arg(long, default_value = "target/benchmark-results/bench-pos")]
    pub(crate) output_dir: PathBuf,

    /// Permit benchmarking an uncommitted working tree (recorded in JSON).
    #[arg(long)]
    pub(crate) allow_dirty: bool,
}

#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
#[derive(Args, Debug)]
pub(crate) struct NativeBenchArgs {
    /// Fixed fixture profile. Changing fixture defaults requires a new profile version.
    #[arg(long, value_enum, default_value_t = NativeBenchProfileArg::V1)]
    pub(crate) profile: NativeBenchProfileArg,

    /// Architecture fixture(s) to measure.
    #[arg(long, value_enum, default_value_t = NativeBenchArchitectureArg::All)]
    pub(crate) architecture: NativeBenchArchitectureArg,

    /// Precision configuration(s) to measure.
    #[arg(long, value_enum, default_value_t = NativeBenchPrecisionArg::All)]
    pub(crate) precision: NativeBenchPrecisionArg,

    /// `native-only` runs CUDA C++ alone; `compare` alternates cuda-oxide and CUDA C++.
    #[arg(long, value_enum, default_value_t = NativeBenchModeArg::NativeOnly)]
    pub(crate) mode: NativeBenchModeArg,

    /// Warm-up steps excluded from timing.
    #[arg(long, default_value_t = 3)]
    pub(crate) warmup_steps: usize,

    /// Timed training steps per run.
    #[arg(long, default_value_t = 100)]
    pub(crate) steps: usize,

    /// Independent runs per backend and precision.
    #[arg(long, default_value_t = 3)]
    pub(crate) runs: usize,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    pub(crate) device: usize,

    /// Directory receiving a timestamped JSON report.
    #[arg(long, default_value = "target/benchmark-results/native-cuda")]
    pub(crate) output_dir: PathBuf,

    /// Permit benchmarking an uncommitted working tree (recorded as dirty in JSON).
    #[arg(long)]
    pub(crate) allow_dirty: bool,
}

#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum NativeBenchProfileArg {
    #[default]
    V1,
}

#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum NativeBenchArchitectureArg {
    Layerstack,
    Simple,
    #[default]
    All,
}

#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum NativeBenchPrecisionArg {
    Fp32,
    #[value(name = "all-optim")]
    AllOptim,
    #[default]
    All,
}

#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum NativeBenchModeArg {
    #[default]
    #[value(name = "native-only")]
    NativeOnly,
    Compare,
}

/// LayerStack アーキ固有の引数。
#[derive(Args, Debug)]
pub(crate) struct LayerstackArgs {
    /// Evaluation scale written to the LayerStack architecture string. When
    /// omitted, sigmoid loss derives it from `--scale`; WRM loss leaves the
    /// token out so the consuming engine can supply it through its FV_SCALE
    /// option.
    #[arg(long, allow_hyphen_values = true, value_parser = parse_positive_i32)]
    pub(crate) fv_scale: Option<i32>,

    /// progress8kpabs coefficient file (`progress.bin`; f64 LE x 125388 = 81
    /// king squares x 1548 KP-abs piece inputs). When omitted in progress8kpabs
    /// mode, every position falls in bucket 4 (zero weights → `sigmoid(0) =
    /// 0.5`). Do not specify this option in kingrank9 mode.
    #[arg(long)]
    pub(crate) progress_coeff: Option<PathBuf>,

    /// Bucket assignment: `progress8kpabs` uses the KP-absolute progress model;
    /// `kingrank9` uses YaneuraOu KingRank9 and requires exactly 9 buckets;
    /// `router` trains a `progress8kpabs`-shaped linear KP-absolute model
    /// (random init, no hidden layer) with `--num-buckets` outputs jointly
    /// with the eval net, treating the outputs as a softmax multi-class
    /// bucket-selection network (MoE-style router; equivalent to N
    /// progress8kpabs-sized linear models sharing the same sparse input,
    /// N = `--num-buckets`, same [2, 9] range as progress8kpabs).
    #[arg(long, default_value = "progress8kpabs")]
    pub(crate) bucket_mode: String,

    /// `router` only: how the router's own weights are trained. `hard-em`
    /// (default) fits the router to an oracle target distribution built from
    /// per-bucket errors. `backprop` skips the oracle target and instead
    /// backpropagates the expected loss under the router's own softmax
    /// distribution directly into its weights, the way a gating network in a
    /// standard (LLM-style) Mixture-of-Experts is trained. `--top-k`,
    /// `--top-k-reduction-interval`, and `--top-k-min` apply to both modes
    /// (see `--top-k` for how the meaning differs by mode). Ignored for
    /// other bucket modes.
    #[arg(long, value_enum, default_value_t = RouterModeArg::HardEm)]
    pub(crate) router_mode: RouterModeArg,

    /// `router` only: Adam learning rate for the router's own training step
    /// (independent of the main `--lr` schedule). On `--resume` /
    /// `--router-resume`, the effective starting value has
    /// `--router-lr-gamma` decay fast-forwarded by the number of superbatches
    /// already completed (`start_superbatch - 1`), so the curve is the same
    /// as an uninterrupted run at any given superbatch. Ignored for other
    /// bucket modes.
    #[arg(long, default_value_t = 0.01)]
    pub(crate) router_lr: f32,

    /// `router` only: multiplicative decay applied to `--router-lr`
    /// once per superbatch (`lr *= router_lr_gamma`). `1.0` disables decay
    /// (constant lr, matching the earlier behavior). Ignored for other
    /// bucket modes.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) router_lr_gamma: f32,

    /// `router` only: L2 weight decay applied to the router's weight
    /// table (`RouterKPAbsWeights::w`; there is no bias term) during its Adam
    /// step. Ignored for other bucket modes.
    #[arg(long, default_value_t = 1.0e-5)]
    pub(crate) router_weight_decay: f32,

    /// `router` only: coefficient `λ` for the Switch-Transformer-style
    /// load-balancing auxiliary loss (`L_balance = N * Σ_i f_i * P_i`, added to
    /// the router's cross-entropy loss) that discourages the router from
    /// collapsing onto a small subset of the `--num-buckets` buckets. `0.0`
    /// disables it (cross-entropy only, matching the earlier behavior). On
    /// `--resume` / `--router-resume`, the effective starting value has
    /// `--router-balance-weight-gamma` decay (and the
    /// `--router-balance-weight-min` clamp) fast-forwarded by the number of
    /// superbatches already completed, matching an uninterrupted run.
    /// Ignored for other bucket modes.
    #[arg(long, default_value_t = 0.01)]
    pub(crate) router_balance_weight: f32,

    /// `router` only: multiplicative decay applied to
    /// `--router-balance-weight` once per superbatch
    /// (`balance_weight = (balance_weight * gamma).max(router_balance_weight_min)`).
    /// `1.0` disables decay. Typical use: start strong to avoid early expert
    /// collapse, then taper off so cross-entropy (oracle accuracy) dominates
    /// once the router has stabilized. Ignored for other bucket modes.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) router_balance_weight_gamma: f32,

    /// `router` only: floor for `--router-balance-weight` after
    /// `--router-balance-weight-gamma` decay; the balance term never drops
    /// below this value. Ignored for other bucket modes.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) router_balance_weight_min: f32,

    /// `router` only: run the hard-EM oracle sweep (`--num-buckets` extra
    /// forward passes per refreshed batch, one per candidate bucket) and
    /// update the router every N batches. `1` refreshes every batch (closest
    /// to training "jointly" with the eval net, but slowest); larger N trades
    /// router freshness for throughput. Ignored for other bucket modes.
    #[arg(long, default_value_t = 1)]
    pub(crate) router_refresh_interval: usize,

    /// `router` only: of the `--num-buckets` E-step candidate buckets (ranked
    /// by error, smallest first), only the `top_k` best are used by the M
    /// step; the meaning of "used" depends on `--router-mode`:
    /// - `hard-em` (default `top_k=1`): the selected buckets are weighted by
    ///   `softmax(-error)` to form the router's soft training target (`1`
    ///   reduces to the previous one-hot hard-EM oracle; `--num-buckets`
    ///   gives the classic Jacobs & Jordan 1991 soft-EM responsibility over
    ///   every bucket).
    /// - `backprop`: softmax is restricted to the selected buckets (mirroring
    ///   real top-k MoE routing, e.g. Switch Transformer's top-1 or Mixtral's
    ///   top-2); only those buckets' logits receive gradient this step.
    ///   `top_k=1` collapses the softmax to a single point and the gradient
    ///   is identically zero, so training stalls — use `top_k >= 2` (see
    ///   `--top-k-min` if annealing `--top-k` down over the run).
    ///
    /// Inference in YaneuraOu always picks a single bucket via argmax
    /// regardless of this setting — Top-K/soft routing only slows down
    /// search with no benefit there, so it stays a training-only technique.
    /// Must be in `[1, --num-buckets]`. On `--resume` / `--router-resume`,
    /// the effective starting value has `--top-k-reduction-interval`
    /// annealing fast-forwarded by the number of superbatches already
    /// completed, matching an uninterrupted run. Ignored for other bucket
    /// modes.
    #[arg(long, default_value_t = 1)]
    pub(crate) top_k: usize,

    /// `router` only: superbatch interval at which `--top-k` is annealed
    /// down by 1 (`1 superbatch ごとに top_k -= 1` when `1`, `every N
    /// superbatches` when `N`). Applied at the end of each superbatch,
    /// alongside `--router-lr-gamma` / `--router-balance-weight-gamma`
    /// decay; never lowers `top_k` below `--top-k-min`. `0` (default)
    /// disables the anneal, keeping `--top-k` constant for the whole run
    /// (previous behavior). Ignored for other bucket modes.
    #[arg(long, default_value_t = 0)]
    pub(crate) top_k_reduction_interval: usize,

    /// `router` only: floor for `--top-k` when annealed by
    /// `--top-k-reduction-interval`. `1` (default) matches the previous
    /// behavior (anneal all the way down to hard-EM one-hot / a single
    /// selected bucket). `--router-mode backprop` needs `top_k >= 2` to get
    /// a nonzero gradient (see `--top-k`), so set this to `2` or higher when
    /// annealing in that mode. Must be in `[1, --top-k]`. Ignored for other
    /// bucket modes.
    #[arg(long, default_value_t = 1)]
    pub(crate) top_k_min: usize,

    /// `router` only: resume the router's weights + Adam optimizer state
    /// from a `{net_id}-{sb}.router.ckpt` sidecar file written by a previous
    /// `router` run (independent of `--resume`, which only restores the
    /// main eval net). Omit to start the router from a fresh random
    /// initialization even when `--resume` is also given.
    #[arg(long)]
    pub(crate) router_resume: Option<PathBuf>,

    /// Output dimension of the FT (feature transformer) per perspective. Must be
    /// a positive multiple of 128. The default value keeps the network
    /// bit-identical to the standard layout and resume-compatible with existing
    /// checkpoints.
    #[arg(long, default_value_t = DEFAULT_FT_OUT)]
    pub(crate) ft_out: usize,

    /// Output dimension of the L1 (per-bucket dense) layer. Specify a value in
    /// [2, 256]. The default keeps the network bit-identical to the standard
    /// layout and resume-compatible with existing checkpoints. Every value runs
    /// on the same per-bucket tiled matmul kernels — the output dimension is
    /// processed in 16-wide tiles — so non-default widths are not penalized.
    #[arg(long, default_value_t = DEFAULT_L1_OUT)]
    pub(crate) l1: usize,

    /// Output dimension of the L2 (per-bucket dense) layer. Specify a value in
    /// [2, 256]; the upper bound is the fixed shared-memory accumulator capacity
    /// of the per-bucket bias-gradient kernel. The default keeps the network
    /// bit-identical to the standard layout and resume-compatible with existing
    /// checkpoints. The L2 / L3 kernels take the output dimension as a runtime
    /// argument, so non-default widths are not penalized.
    #[arg(long, default_value_t = DEFAULT_L2_OUT)]
    pub(crate) l2: usize,

    /// LayerStack output bucket count. In progress8kpabs mode, each position is
    /// routed to `min(N-1, floor(p * N))` and N must be in `[2, 9]`. In
    /// kingrank9 mode this value must be 9. The upper bound is the fixed 9-register accumulator
    /// in the per-bucket weight backward kernels. The default 9 keeps the
    /// binning and weight-buffer shape identical to the standard layout and
    /// resume-compatible with existing checkpoints. The historical 8-bucket
    /// progress emission used `floor(p * 8)` on a 9-slot layout, leaving slot 8
    /// unused; the unified design here means setting `--num-buckets 9` (the
    /// default) actually emits index 8 — existing 9-bucket distributed nets
    /// have an untrained slot 8 and may see a short-term eval shift on the
    /// `p in [8/9, 1]` tail until continued training catches up.
    #[arg(long, default_value_t = DEFAULT_NUM_BUCKETS)]
    pub(crate) num_buckets: usize,

    /// Opt-in flag to use Ampere+ Tensor Cores in TF32 mode. `true` calls cuBLAS
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)`, rounding the
    /// FP32 Sgemm inputs to 10-bit-mantissa TF32 and running TC mma → FP32
    /// accumulate (~3 significant decimal digits of mantissa; exponent range
    /// same as FP32). With the default `false`, it runs `CUBLAS_DEFAULT_MATH` (a
    /// pure FP32 path, no Tensor Cores).
    ///
    /// Dropping 13 mantissa bits affects the numerics of the `fwd_L1f` /
    /// `bwd_L1f` Sgemm, so it is conservatively default OFF for quality.
    #[arg(long)]
    pub(crate) tf32: bool,

    /// Also keep the FT activation (the `ft_*_out` forward output and the
    /// `dft_*_out` backward gradient) in FP16. Requires `--ft-fp16` (an
    /// extension stacked on top of the weight FP16 path).
    ///
    /// `ft_*_out` is the output of `sparse_ft_forward`; making it FP16 halves
    /// the bandwidth of the subsequent read + inverse-index gather (`phD`, the
    /// heaviest DRAM read in a step). dft is a tiny value proportional to
    /// `1/batch` from batch normalization, so when made FP16 it is lifted into
    /// the normal range by loss scaling (a factor proportional to batch) before
    /// being stored.
    ///
    /// Split into a separate flag from weight FP16 (`--ft-fp16`) so that SPRT
    /// can isolate the strength impact in two steps:
    /// FP32 → `--ft-fp16` → `--ft-fp16 --ft-fp16-out`. Quantisation error may
    /// shift playing strength, so it is default OFF and production quality is
    /// not guaranteed until confirmed by SPRT.
    #[arg(long)]
    pub(crate) ft_fp16_out: bool,

    /// Enable the PSQT (Piece-Square Table) shortcut layer.
    ///
    /// PSQT is a per-feature × per-bucket scalar prior layered in parallel with
    /// the dense `FT -> L1 -> L2 -> L3` path: `net_output +=
    /// 0.5 * (Σ stm_active psqt_w[f, bucket] - Σ nstm_active psqt_w[f, bucket])`.
    /// Stockfish SFNNv10 style — the dense path then only has to learn
    /// non-material structure on top of the material prior carried by PSQT.
    ///
    /// Default OFF for bit-identical compatibility with non-PSQT checkpoints.
    /// When enabled the saved `.bin` carries an extra `PSQT=<num-buckets>,`
    /// token in the arch string plus an i32 PSQT block (scale `QA * QB = 8128`).
    #[arg(long)]
    pub(crate) psqt: bool,

    /// PSQT shortcut weight initialiser: `zeroed` (default) or `material`.
    ///
    /// - `zeroed`: every PSQT weight starts at 0; the dense path absorbs
    ///   material information first and PSQT only picks up the residual
    ///   correction. Known to leave a long plateau early in training.
    /// - `material`: PSQT weights are pre-loaded with centipawn piece values
    ///   divided by `--wrm-nnue2score` (or `--scale` when WRM is off) so the
    ///   shortcut already encodes piece material from step 0. The dense path
    ///   then specialises in non-material structure (positional/tactical
    ///   patterns).
    ///
    /// Requires `--psqt`. Material init additionally requires the loss to know
    /// the centipawn → logit scaling: either use `--win-rate-model` with
    /// `--wrm-in-scaling` (and `--wrm-nnue2score`) set, or use the sigmoid path
    /// where `--scale` provides the conversion factor.
    #[arg(long, value_enum, default_value_t = PsqtInit::Zeroed, requires = "psqt")]
    pub(crate) psqt_init: PsqtInit,

    /// effect bucket FT factorizer sharing mode. `pool-buckets` shares one virtual row
    /// for each piece across effect buckets; `per-bucket` keeps effect buckets
    /// separate.
    #[arg(long = "ft-factorize-effect-bucket-share", value_enum, default_value_t = EffectBucketFactorizeShare::PoolBuckets)]
    pub(crate) ft_factorize_effect_bucket_share: EffectBucketFactorizeShare,

    /// Threat sparse feature profile. One of: off (default), full, same-class,
    /// same-class-major-pawn, step-attacker, full-symdedup, cross-side. When not
    /// `off`, threat edge features (one piece attacking another) are concatenated
    /// after the base feature transformer inputs, growing the FT input dimension
    /// and the active-feature count. `off` is bit-identical to the base feature
    /// set. `full-symdedup` shares `full`'s input dimension but drops each
    /// symmetric-redundant edge (one side of a mutually-implied attack pair),
    /// lowering the active-feature count without changing the index space.
    ///
    /// Threat coexists with the FT factorizer (the fold/reduce/coalesce paths
    /// stay within the base rows and never touch the threat block); it is still
    /// mutually exclusive with `--psqt` (base-only PSQT is not yet validated).
    /// The dimension increase makes higher profiles (full / same-class)
    /// GPU-memory heavy, and the factorizer fold buffer adds roughly one extra
    /// FT-weight matrix — pick a smaller profile or lower `--ft-out` if training
    /// OOMs.
    #[arg(long = "threat-profile", default_value = "off")]
    pub(crate) threat_profile: String,

    /// effect bucket feature config. One of: off (default),
    /// 2x2-kingfixed, 2x2-kingbucketed, 3x3-kingfixed,
    /// 3x3-kingbucketed. effect bucket rewrites every base feature row as
    /// `base_index * NB + bucket`, so it is mutually exclusive with
    /// `--threat-profile` and `--psqt`.
    #[arg(long = "effect-bucket", default_value = "off")]
    pub(crate) effect_bucket_config: String,
}

/// PSQT shortcut の初期化方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum PsqtInit {
    /// Start every PSQT weight at zero (PSQT is initially inert).
    Zeroed,
    /// Pre-load PSQT with centipawn piece values / out_scaling (Material prior).
    Material,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum EffectBucketFactorizeShare {
    PoolBuckets,
    PerBucket,
}

/// Simple 4 層アーキ固有の引数。
#[derive(Args, Debug)]
pub(crate) struct SimpleArgs {
    /// Layer-dimension preset (`<l1>x2-<l2>-<l3>`). l1 is the accumulator (FT
    /// output) dimension; l2 / l3 are the hidden-layer dimensions. Each can be
    /// overridden individually with `--l1` / `--l2` / `--l3`.
    #[arg(long, default_value = "256x2-32-32")]
    pub(crate) arch: String,

    /// Accumulator (FT output) dimension. Defaults to the `--arch` preset value.
    #[arg(long)]
    pub(crate) l1: Option<usize>,

    /// Dimension of hidden layer 1. Defaults to the `--arch` preset value.
    #[arg(long)]
    pub(crate) l2: Option<usize>,

    /// Dimension of hidden layer 2. Defaults to the `--arch` preset value.
    #[arg(long)]
    pub(crate) l3: Option<usize>,

    /// FT post-activation function ("crelu" / "screlu" / "pairwise").
    /// "pairwise" multiplies the corresponding indices of the first and second
    /// halves, halving the L1 input dimension (the L1 / L2 dense layers use
    /// CReLU activation).
    #[arg(long, default_value = "crelu")]
    pub(crate) activation: String,

    /// Also keep the FT activation (the `ft_*_out` forward output and the
    /// `dft_*_out` backward gradient) in FP16. Requires the global `--ft-fp16`
    /// (supports crelu / screlu / pairwise).
    ///
    /// `ft_*_out` is the output of `sparse_ft_forward`; making it FP16 halves
    /// the bandwidth of the subsequent read + the `sparse_ft_backward` read. dft
    /// is a tiny value proportional to `1/batch` from batch normalization, so
    /// when made FP16 it is lifted into the normal range by loss scaling
    /// (proportional to batch) before being stored.
    ///
    /// An opt-in option: quantisation error may shift playing strength, so it is
    /// default OFF and production quality is not guaranteed until confirmed by
    /// SPRT.
    #[arg(long)]
    pub(crate) ft_fp16_out: bool,

    /// Opt-in flag to use Ampere+ Tensor Cores in TF32 mode. `true` calls cuBLAS
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)`, rounding the
    /// FP32 inputs of the L1/L2/L3 dense Sgemm to 10-bit-mantissa TF32 and
    /// running TC mma → FP32 accumulate (~3 significant decimal digits of
    /// mantissa; exponent range same as FP32). With the default `false`, it runs
    /// `CUBLAS_DEFAULT_MATH` (a pure FP32 path, no Tensor Cores).
    ///
    /// Dropping 13 mantissa bits affects the numerics of the dense Sgemm, so it
    /// is conservatively default OFF for quality. Same policy as LayerStack
    /// `--tf32` (an opt-in flag with a playing-strength risk).
    #[arg(long)]
    pub(crate) tf32: bool,
}
