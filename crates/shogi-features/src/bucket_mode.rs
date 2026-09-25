//! `--bucket-mode` の複合バケット DSL。
//!
//! YaneuraOu (現行, `architectures/nnue_arch_gen.py` V1.03) の SFNN layer-stack
//! トークン列と **同じ文字列** ・**同じ合成規則** を tatara の学習側でも解釈できるように
//! する。`hand4/16/64/64z/256/1024` ・ `k3k3/k9k9/k9k9z/k13k13z/k21k21/k29k29` ・
//! `progress2/3/4/8/16/32` を `_` 区切りで自由に複合でき (各カテゴリ最大1個)、
//! `routerkpabs<N>` / `routerft<R>ft<R>` (どちらか一方のみ、最大1個) を追加できる。
//! さらに末尾に `wsb` (WithSharedBucket) を置くと、常に選ばれる共有バケットを
//! 1個追加できる。
//!
//! 合成順序 (バケットindexの桁の重み) は **hand → king → progress → router** の順に
//! `idx = idx * category_buckets + category_index` を繰り返したもの。router 系は
//! カテゴリの中で常に最後 (= 最下位桁) に合成される。これは YaneuraOu
//! `evaluate_nnue.cpp` の `stack_index_for_nnue()` と完全に同じ規則 (このファイルは
//! そのRust移植)。`wsb` は上記の合成には加わらない別枠で、hand/king/progress/router
//! の合成バケット数 (`prefix_buckets() * router.bucket_count()`) に対して常に
//! index = その総数 (=最後の1個) を追加する。この共有バケットの選択規則も
//! `evaluate_nnue.cpp` の `NNUE_SFNN_USE_SHARED_BUCKET` 分岐と同じ (常に有効)。
//!
//! 保存形式は YaneuraOu の慣習 (このバケットindexの並び) を正として、tatara旧形式や
//! yaneuraou-privateとは非互換。旧形式からの変換は `net_convert_bucket_layout`
//! (bins/net_convert_bucket_layout) を使う。

use shogi_format::{Color, ShogiBoard, Square};

pub use crate::kingrank9::kingrank9_bucket_board as king3_by_king3_bucket;

/// router系サブモード。`routerkpabs` と `routerft{R}ft{R}` は排他 (最大1個)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterSubMode {
    /// バケット選択専用の学習可能な `RouterKPAbs` (progress8kpabs と同じ特徴、N-way argmax)。
    /// バケット数はそのまま `n`。
    Kpabs { n: u32 },
    /// FeatureTransformerに同居する router (`--router-arch ft-by-ft` 相当)。
    /// STM側/NSTM側それぞれ `r` 通りのargmaxを取り、`r*r` 通りに合成する。
    FtByFt { r: u32 },
}

impl RouterSubMode {
    /// このサブモードが持つ総バケット数 (kpabsはN、ft-by-ftはR*R)。
    pub fn bucket_count(self) -> u32 {
        match self {
            RouterSubMode::Kpabs { n } => n,
            RouterSubMode::FtByFt { r } => r * r,
        }
    }

    /// 生成器 (`nnue_arch_gen.py`) の `NNUE_SFNN_ROUTER_N` に相当する値
    /// (kpabsはN本人、ft-by-ftはR)。
    pub fn router_n_macro(self) -> u32 {
        match self {
            RouterSubMode::Kpabs { n } => n,
            RouterSubMode::FtByFt { r } => r,
        }
    }

    /// アーキテクチャ名トークン (`routerkpabs9`, `routerft8ft8` 等)。
    pub fn token(self) -> String {
        match self {
            RouterSubMode::Kpabs { n } => format!("routerkpabs{n}"),
            RouterSubMode::FtByFt { r } => format!("routerft{r}ft{r}"),
        }
    }
}

/// hand バケットの種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandSubMode {
    Hand4,
    Hand16,
    Hand64,
    Hand64Z,
    Hand256,
    Hand1024,
}

impl HandSubMode {
    pub fn bucket_count(self) -> u32 {
        match self {
            HandSubMode::Hand4 => 4,
            HandSubMode::Hand16 => 16,
            HandSubMode::Hand64 | HandSubMode::Hand64Z => 64,
            HandSubMode::Hand256 => 256,
            HandSubMode::Hand1024 => 1024,
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            HandSubMode::Hand4 => "hand4",
            HandSubMode::Hand16 => "hand16",
            HandSubMode::Hand64 => "hand64",
            HandSubMode::Hand64Z => "hand64z",
            HandSubMode::Hand256 => "hand256",
            HandSubMode::Hand1024 => "hand1024",
        }
    }
}

/// king バケットの種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KingSubMode {
    K3K3,
    K9K9,
    K9K9Z,
    K13K13Z,
    K21K21,
    K29K29,
}

impl KingSubMode {
    pub fn bucket_count(self) -> u32 {
        match self {
            KingSubMode::K3K3 => 9,
            KingSubMode::K9K9 | KingSubMode::K9K9Z => 81,
            KingSubMode::K13K13Z => 13 * 13,
            KingSubMode::K21K21 => 21 * 21,
            KingSubMode::K29K29 => 29 * 29,
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            KingSubMode::K3K3 => "k3k3",
            KingSubMode::K9K9 => "k9k9",
            KingSubMode::K9K9Z => "k9k9z",
            KingSubMode::K13K13Z => "k13k13z",
            KingSubMode::K21K21 => "k21k21",
            KingSubMode::K29K29 => "k29k29",
        }
    }
}

/// パース済みの `--bucket-mode` (複合可能な hand/king/progress/router)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BucketMode {
    pub hand: Option<HandSubMode>,
    pub king: Option<KingSubMode>,
    /// progress2/3/4/8/16/32 の N。指定なしは None (= 従来のprogress8kpabs系とは別、
    /// 単に progress バケットを使わないことを表す)。
    pub progress: Option<u32>,
    pub router: Option<RouterSubMode>,
    /// `wsb` (WithSharedBucket)。true のとき、hand/king/progress/router の合成
    /// バケットに加えて「常に選ばれる共有バケット」を1個追加する
    /// (index は常に `prefix_buckets() * router.bucket_count()`、すなわち
    /// 合成バケット数そのもの = 最後の1個)。
    pub shared_bucket: bool,
}

impl BucketMode {
    pub const NONE: BucketMode =
        BucketMode { hand: None, king: None, progress: None, router: None, shared_bucket: false };

    /// router以外 (hand/king/progress) の合成バケット数。router併用時、
    /// 合成後の総バケットindexから `prefix = idx / router.bucket_count()` で
    /// 復元できる (router は必ず最後=最下位桁に合成されるため)。
    pub fn prefix_buckets(&self) -> u32 {
        let mut n = 1u32;
        if let Some(h) = self.hand {
            n *= h.bucket_count();
        }
        if let Some(k) = self.king {
            n *= k.bucket_count();
        }
        if let Some(p) = self.progress {
            n *= p;
        }
        n
    }

    /// router 自身のバケット数 (kpabsならN、ft-by-ftならR*R)。router無しならNone。
    pub fn router_bucket_count(&self) -> Option<u32> {
        self.router.map(RouterSubMode::bucket_count)
    }

    /// hand/king/progress/router の合成バケット数 (`wsb` の共有バケットを含まない)。
    /// `wsb` 有効時、共有バケットの index はこの値そのもの (= 合成バケットの直後)。
    pub fn selectable_buckets(&self) -> u32 {
        let mut n = self.prefix_buckets();
        if let Some(r) = self.router {
            n *= r.bucket_count();
        }
        n
    }

    /// 総バケット数 (hand * king * progress * router、`wsb` 有効時はさらに+1)。
    pub fn total_buckets(&self) -> u32 {
        self.selectable_buckets() + u32::from(self.shared_bucket)
    }

    /// `wsb` 有効時の共有バケットのindex (`selectable_buckets()` と同値)。
    /// `wsb` 無効時に呼ぶのは呼び出し側のバグなので `None` を返す。
    pub fn shared_bucket_index(&self) -> Option<u32> {
        self.shared_bucket.then(|| self.selectable_buckets())
    }

    /// YaneuraOu生成器の `NNUE_SFNN_*` マクロと同じ形の正準トークン列
    /// (`hand64z_k9k9_progress4_routerkpabs5` のように、hand→king→progress→routerの順)。
    /// `wsb` は常に末尾に付与する。空 (バケット無し) のときは `"NONE"` (`wsb` 単独の
    /// ときは `"WSB"`)。
    pub fn canonical_token(&self) -> String {
        let mut parts = Vec::new();
        if let Some(h) = self.hand {
            parts.push(h.token().to_string());
        }
        if let Some(k) = self.king {
            parts.push(k.token().to_string());
        }
        if let Some(p) = self.progress {
            parts.push(format!("progress{p}"));
        }
        if let Some(r) = self.router {
            parts.push(r.token());
        }
        if self.shared_bucket {
            parts.push("wsb".to_string());
        }
        if parts.is_empty() {
            "NONE".to_string()
        } else {
            parts.join("_")
        }
    }

    /// `--bucket-mode` 文字列 (`_` 区切りトークン列、順不同、大文字小文字不問) をパースする。
    /// 各カテゴリ (hand/king/progress/router) は最大1個、router系
    /// (routerkpabs/routerft{R}ft{R}) は互いに排他。`wsb` はどのトークンとも複合でき、
    /// **文字列上、必ず最後のトークン**でなければならない (YaneuraOu
    /// `nnue_arch_gen.py` の `wsb` 検証と同じ規則)。
    ///
    /// 空文字列 / `"none"` はバケット無し (`BucketMode::NONE`、常に bucket 0 の1バケット)
    /// を表す。
    pub fn parse(spec: &str) -> Result<BucketMode, String> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("none") {
            return Ok(BucketMode::NONE);
        }

        let mut mode = BucketMode::NONE;
        let raw_tokens: Vec<&str> = spec.split('_').filter(|t| !t.is_empty()).collect();
        let last_index = raw_tokens.len().checked_sub(1);
        for (i, raw_token) in raw_tokens.iter().enumerate() {
            let token = normalize_token(raw_token);
            if token == "WSB" {
                if Some(i) != last_index {
                    return Err(format!(
                        "wsb (WithSharedBucket) must be the last token in bucket-mode {spec:?}"
                    ));
                }
                mode.shared_bucket = true;
                continue;
            }
            if let Some(hand) = parse_hand_token(&token) {
                if mode.hand.is_some() {
                    return Err(format!("duplicate hand bucket in bucket-mode {spec:?}"));
                }
                mode.hand = Some(hand);
                continue;
            }
            if let Some(king) = parse_king_token(&token) {
                if mode.king.is_some() {
                    return Err(format!("duplicate king bucket in bucket-mode {spec:?}"));
                }
                mode.king = Some(king);
                continue;
            }
            if let Some(n) = parse_progress_token(&token) {
                if mode.progress.is_some() {
                    return Err(format!("duplicate progress bucket in bucket-mode {spec:?}"));
                }
                mode.progress = Some(n);
                continue;
            }
            if let Some(router) = parse_router_token(&token)? {
                if mode.router.is_some() {
                    return Err(format!(
                        "router bucket (routerkpabs / routerft<R>ft<R>) may appear at most once in bucket-mode {spec:?}"
                    ));
                }
                mode.router = Some(router);
                continue;
            }
            return Err(format!(
                "unknown bucket-mode token {raw_token:?} in {spec:?}; expected hand4/16/64/64z/256/1024, k3k3/k9k9/k9k9z/k13k13z/k21k21/k29k29, progress2/3/4/8/16/32, routerkpabs<N>, routerft<R>ft<R>, or wsb"
            ));
        }
        Ok(mode)
    }
}

fn normalize_token(token: &str) -> String {
    let upper = token.to_ascii_uppercase();
    upper
        .replace("KING3_BY_KING3", "K3K3")
        .replace("KING9_BY_KING9", "K9K9")
        .replace("KING9Z_BY_KING9Z", "K9K9Z")
        .replace("KING9ZONE_BY_KING9ZONE", "K9K9Z")
        .replace("KING13Z_BY_KING13Z", "K13K13Z")
        .replace("KING13ZONE_BY_KING13ZONE", "K13K13Z")
        .replace("KING21_BY_KING21", "K21K21")
        .replace("KING29_BY_KING29", "K29K29")
}

fn parse_hand_token(token: &str) -> Option<HandSubMode> {
    match token {
        "HAND4" => Some(HandSubMode::Hand4),
        "HAND16" => Some(HandSubMode::Hand16),
        "HAND64" => Some(HandSubMode::Hand64),
        "HAND64Z" => Some(HandSubMode::Hand64Z),
        "HAND256" => Some(HandSubMode::Hand256),
        "HAND1024" => Some(HandSubMode::Hand1024),
        _ => None,
    }
}

fn parse_king_token(token: &str) -> Option<KingSubMode> {
    match token {
        "K3K3" => Some(KingSubMode::K3K3),
        "K9K9" => Some(KingSubMode::K9K9),
        "K9K9Z" => Some(KingSubMode::K9K9Z),
        "K13K13Z" => Some(KingSubMode::K13K13Z),
        "K21K21" => Some(KingSubMode::K21K21),
        "K29K29" => Some(KingSubMode::K29K29),
        _ => None,
    }
}

fn parse_progress_token(token: &str) -> Option<u32> {
    let raw = token.strip_prefix("PROGRESS")?;
    let n: u32 = raw.parse().ok()?;
    matches!(n, 2 | 3 | 4 | 8 | 16 | 32).then_some(n)
}

fn parse_router_token(token: &str) -> Result<Option<RouterSubMode>, String> {
    if let Some(raw) = token.strip_prefix("ROUTERKPABS") {
        let n: u32 = raw
            .parse()
            .map_err(|_| format!("routerkpabs<N> requires an integer N, got {token:?}"))?;
        if n < 1 {
            return Err(format!("routerkpabs<N> requires N >= 1, got {token:?}"));
        }
        return Ok(Some(RouterSubMode::Kpabs { n }));
    }
    if let Some(rest) = token.strip_prefix("ROUTERFT") {
        // "8FT8" のように "FT<R2>" が続く形を要求する。
        let ft_pos = rest
            .find("FT")
            .ok_or_else(|| format!("routerft<R>ft<R> must be like routerft8ft8, got {token:?}"))?;
        let (r1_str, tail) = rest.split_at(ft_pos);
        let r2_str = &tail[2..]; // "FT" の後ろ
        let r1: u32 = r1_str
            .parse()
            .map_err(|_| format!("routerft<R>ft<R> requires an integer R, got {token:?}"))?;
        let r2: u32 = r2_str
            .parse()
            .map_err(|_| format!("routerft<R>ft<R> requires an integer R, got {token:?}"))?;
        if r1 != r2 {
            return Err(format!("routerft<R>ft<R> requires both R to match, got {token:?}"));
        }
        if r1 < 1 {
            return Err(format!("routerft<R>ft<R> requires R >= 1, got {token:?}"));
        }
        return Ok(Some(RouterSubMode::FtByFt { r: r1 }));
    }
    Ok(None)
}

// ============================================================
//  hand / king バケット index の計算 (YaneuraOu evaluate_nnue.cpp を移植)
// ============================================================
//
// 手番側視点への正規化・180度回転などは全て YaneuraOu `evaluate_nnue.cpp` の
// `king*_bucket` / `hand*_bucket` 系関数とビット完全に同じ規則にしてある。
// (手番側の玉は反転しない。非手番側の玉だけ180度回転して手番側視点に正規化する。
//  ただし k3k3 だけは歴史的経緯で「非手番側視点なら手番側を反転」という実装になって
//  いるが、これは 9 通りの分割としては対称なので同値になる。)

fn king9_single(sq: Square) -> u32 {
    sq.rank().clamp(0, 8) as u32
}

/// k9k9 バケット (0..=80)。
pub fn king9_by_king9_bucket(board: &ShogiBoard) -> u32 {
    let (f_king, e_king) = friend_enemy_king(board);
    king9_single(f_king) * 9 + king9_single(e_king)
}

fn file3_bucket(file: u8) -> u32 {
    (file.clamp(0, 8) / 3) as u32
}

fn king9_zone_single(sq: Square) -> u32 {
    let rank = sq.rank().clamp(0, 8);
    let file = sq.file();
    if rank < 3 {
        0
    } else if rank < 6 {
        1
    } else if rank == 6 {
        2
    } else {
        3 + (rank as u32 - 7) * 3 + file3_bucket(file)
    }
}

/// k9k9z バケット (0..=80)。
pub fn king9_zone_by_king9_zone_bucket(board: &ShogiBoard) -> u32 {
    let (f_king, e_king) = friend_enemy_king(board);
    king9_zone_single(f_king) * 9 + king9_zone_single(e_king)
}

fn king13_zone_single(sq: Square) -> u32 {
    let rank = sq.rank().clamp(0, 8);
    let file = sq.file();
    if rank < 7 {
        rank as u32
    } else {
        7 + (rank as u32 - 7) * 3 + file3_bucket(file)
    }
}

/// k13k13z バケット (0..=168)。
pub fn king13_zone_by_king13_zone_bucket(board: &ShogiBoard) -> u32 {
    let (f_king, e_king) = friend_enemy_king(board);
    king13_zone_single(f_king) * 13 + king13_zone_single(e_king)
}

fn king21_single(sq: Square) -> u32 {
    let rank = sq.rank().clamp(0, 8);
    let file = sq.file().clamp(0, 8) as u32;
    if rank < 3 {
        0
    } else if rank < 6 {
        1
    } else if rank == 6 {
        2
    } else {
        3 + (rank as u32 - 7) * 9 + file
    }
}

/// k21k21 バケット (0..=440)。
pub fn king21_by_king21_bucket(board: &ShogiBoard) -> u32 {
    let (f_king, e_king) = friend_enemy_king(board);
    king21_single(f_king) * 21 + king21_single(e_king)
}

fn king29_single(sq: Square) -> u32 {
    let rank = sq.rank().clamp(0, 8);
    let file = sq.file().clamp(0, 8) as u32;
    if rank < 3 {
        0
    } else if rank < 6 {
        1
    } else {
        2 + (rank as u32 - 6) * 9 + file
    }
}

/// k29k29 バケット (0..=840)。
pub fn king29_by_king29_bucket(board: &ShogiBoard) -> u32 {
    let (f_king, e_king) = friend_enemy_king(board);
    king29_single(f_king) * 29 + king29_single(e_king)
}

/// 手番側玉 (そのまま) / 非手番側玉 (180度回転して手番側視点に正規化) の座標を返す。
/// YaneuraOu の `stm == BLACK ? f_king : Inv(f_king)` / `stm == BLACK ? Inv(e_king) : e_king`
/// と等価 (手番側視点＝先手視点になるように揃える)。
fn friend_enemy_king(board: &ShogiBoard) -> (Square, Square) {
    let (f_king, e_king) = match board.side_to_move {
        Color::Black => (board.black_king_sq, board.white_king_sq),
        Color::White => (board.white_king_sq, board.black_king_sq),
    };
    match board.side_to_move {
        Color::Black => (f_king, e_king.inverse()),
        Color::White => (f_king.inverse(), e_king),
    }
}

fn hand_of(board: &ShogiBoard, color: Color) -> shogi_format::Hand {
    match color {
        Color::Black => board.black_hand,
        Color::White => board.white_hand,
    }
}

fn hand4_single(hand: shogi_format::Hand) -> u32 {
    u32::from(hand.bishop() > 0)
}

fn hand16_single(hand: shogi_format::Hand) -> u32 {
    let mut bucket = 0u32;
    if hand.pawn() > 0 {
        bucket |= 1;
    }
    if hand.bishop() > 0 {
        bucket |= 2;
    }
    bucket
}

fn hand64_single(hand: shogi_format::Hand) -> u32 {
    let mut bucket = 0u32;
    if hand.pawn() + hand.lance() + hand.knight() > 0 {
        bucket |= 1;
    }
    if hand.gold() + hand.silver() + hand.rook() > 0 {
        bucket |= 2;
    }
    if hand.bishop() > 0 {
        bucket |= 4;
    }
    bucket
}

fn hand64z_single(hand: shogi_format::Hand) -> u32 {
    let score = i32::from(hand.pawn())
        + i32::from(hand.lance() + hand.knight()) * 2
        + i32::from(hand.silver() + hand.gold()) * 3
        + i32::from(hand.bishop() + hand.rook()) * 5;
    (((score + 3) / 4).clamp(0, 7)) as u32
}

fn hand256_single(hand: shogi_format::Hand) -> u32 {
    let mut bucket = 0u32;
    if hand.pawn() + hand.lance() + hand.knight() > 0 {
        bucket |= 1;
    }
    if hand.silver() + hand.gold() > 0 {
        bucket |= 2;
    }
    if hand.bishop() > 0 {
        bucket |= 4;
    }
    if hand.rook() > 0 {
        bucket |= 8;
    }
    bucket
}

fn hand1024_single(hand: shogi_format::Hand) -> u32 {
    let mut bucket = 0u32;
    if hand.pawn() > 0 {
        bucket |= 1;
    }
    if hand.lance() + hand.knight() > 0 {
        bucket |= 2;
    }
    if hand.silver() + hand.gold() > 0 {
        bucket |= 4;
    }
    if hand.bishop() > 0 {
        bucket |= 8;
    }
    if hand.rook() > 0 {
        bucket |= 16;
    }
    bucket
}

pub fn hand4_bucket(board: &ShogiBoard) -> u32 {
    let stm = board.side_to_move;
    hand4_single(hand_of(board, stm)) * 2 + hand4_single(hand_of(board, stm.opponent()))
}
pub fn hand16_bucket(board: &ShogiBoard) -> u32 {
    let stm = board.side_to_move;
    hand16_single(hand_of(board, stm)) * 4 + hand16_single(hand_of(board, stm.opponent()))
}
pub fn hand64_bucket(board: &ShogiBoard) -> u32 {
    let stm = board.side_to_move;
    hand64_single(hand_of(board, stm)) * 8 + hand64_single(hand_of(board, stm.opponent()))
}
pub fn hand64z_bucket(board: &ShogiBoard) -> u32 {
    let stm = board.side_to_move;
    hand64z_single(hand_of(board, stm)) * 8 + hand64z_single(hand_of(board, stm.opponent()))
}
pub fn hand256_bucket(board: &ShogiBoard) -> u32 {
    let stm = board.side_to_move;
    hand256_single(hand_of(board, stm)) * 16 + hand256_single(hand_of(board, stm.opponent()))
}
pub fn hand1024_bucket(board: &ShogiBoard) -> u32 {
    let stm = board.side_to_move;
    hand1024_single(hand_of(board, stm)) * 32 + hand1024_single(hand_of(board, stm.opponent()))
}

/// `HandSubMode` に応じたバケットindexを返す (0.. `sub.bucket_count()-1`)。
pub fn hand_bucket(sub: HandSubMode, board: &ShogiBoard) -> u32 {
    match sub {
        HandSubMode::Hand4 => hand4_bucket(board),
        HandSubMode::Hand16 => hand16_bucket(board),
        HandSubMode::Hand64 => hand64_bucket(board),
        HandSubMode::Hand64Z => hand64z_bucket(board),
        HandSubMode::Hand256 => hand256_bucket(board),
        HandSubMode::Hand1024 => hand1024_bucket(board),
    }
}

/// `KingSubMode` に応じたバケットindexを返す (0.. `sub.bucket_count()-1`)。
pub fn king_bucket(sub: KingSubMode, board: &ShogiBoard) -> u32 {
    match sub {
        KingSubMode::K3K3 => u32::from(king3_by_king3_bucket(board)),
        KingSubMode::K9K9 => king9_by_king9_bucket(board),
        KingSubMode::K9K9Z => king9_zone_by_king9_zone_bucket(board),
        KingSubMode::K13K13Z => king13_zone_by_king13_zone_bucket(board),
        KingSubMode::K21K21 => king21_by_king21_bucket(board),
        KingSubMode::K29K29 => king29_by_king29_bucket(board),
    }
}

/// `mode` の router以外 (hand/king/progress) の合成バケットindex
/// (`0..mode.prefix_buckets()`)。router自身の選択が決まる前に、まずこれを
/// 求めてから `combine_bucket_index` に渡す (dataloaderのpush時と、router
/// oracle sweepでの「同じprefixを保ったままrouterの候補だけ振る」用途の両方
/// で使う)。
pub fn prefix_index(mode: &BucketMode, board: &ShogiBoard, progress_bucket: Option<u32>) -> u32 {
    let mut idx = 0u32;
    if let Some(h) = mode.hand {
        idx = hand_bucket(h, board);
    }
    if let Some(k) = mode.king {
        idx = idx * k.bucket_count() + king_bucket(k, board);
    }
    if let Some(p) = mode.progress {
        idx = idx * p + progress_bucket.unwrap_or(0).min(p - 1);
    }
    idx
}

/// `BucketMode` 全体のバケットindexを、hand/king/progressの合成値 (router抜き) と
/// router自身のバケットindexから合成する。router は必ず最後 (最下位桁)。
///
/// 戻り値は常に `0..mode.selectable_buckets()` の範囲 (`wsb` の共有バケット index
/// `mode.shared_bucket_index()` は含まない)。`wsb` 有効時、共有バケットは局面に
/// 依らず常に選ばれる別枠のバケットなので、呼び出し側 (dataloader / trainer) が
/// この関数の戻り値と `shared_bucket_index()` の両方を必要に応じて使う。
///
/// `progress_bucket` / `router_bucket` は呼び出し側 (progress8kpabs / RouterKPAbs /
/// FT-by-FTのargmax) が計算した値を渡す — このcrateはFT重みや学習済みrouterの
/// 重みを持たないため、progress・router自体のバケットindex計算はここでは行わない。
pub fn combine_bucket_index(
    mode: &BucketMode,
    board: &ShogiBoard,
    progress_bucket: Option<u32>,
    router_bucket: Option<u32>,
) -> u32 {
    let mut idx = prefix_index(mode, board, progress_bucket);
    if let Some(r) = mode.router {
        let rb = router_bucket.unwrap_or(0).min(r.bucket_count() - 1);
        idx = idx * r.bucket_count() + rb;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_is_none() {
        assert_eq!(BucketMode::parse("").unwrap(), BucketMode::NONE);
        assert_eq!(BucketMode::parse("none").unwrap(), BucketMode::NONE);
        assert_eq!(BucketMode::NONE.total_buckets(), 1);
    }

    #[test]
    fn parse_composes_hand_king_progress() {
        let mode = BucketMode::parse("hand64z_k9k9_progress4").unwrap();
        assert_eq!(mode.hand, Some(HandSubMode::Hand64Z));
        assert_eq!(mode.king, Some(KingSubMode::K9K9));
        assert_eq!(mode.progress, Some(4));
        assert_eq!(mode.router, None);
        assert_eq!(mode.total_buckets(), 64 * 81 * 4);
        assert_eq!(mode.canonical_token(), "hand64z_k9k9_progress4");
    }

    #[test]
    fn parse_router_kpabs() {
        let mode = BucketMode::parse("k3k3_routerkpabs9").unwrap();
        assert_eq!(mode.router, Some(RouterSubMode::Kpabs { n: 9 }));
        assert_eq!(mode.total_buckets(), 9 * 9);
    }

    #[test]
    fn parse_router_ftft() {
        let mode = BucketMode::parse("routerft8ft8").unwrap();
        assert_eq!(mode.router, Some(RouterSubMode::FtByFt { r: 8 }));
        assert_eq!(mode.total_buckets(), 64);
    }

    #[test]
    fn router_modes_are_exclusive() {
        assert!(BucketMode::parse("routerkpabs4_routerft2ft2").is_err());
    }

    #[test]
    fn duplicate_category_rejected() {
        assert!(BucketMode::parse("hand4_hand16").is_err());
        assert!(BucketMode::parse("k3k3_k9k9").is_err());
        assert!(BucketMode::parse("progress4_progress8").is_err());
    }

    #[test]
    fn unknown_token_rejected() {
        assert!(BucketMode::parse("bogus123").is_err());
    }

    #[test]
    fn router_always_combines_last() {
        // hand4 * king(k3k3=9) * router(kpabs 5) の合成順で、router が最下位桁になることを確認。
        let mode = BucketMode::parse("hand4_k3k3_routerkpabs5").unwrap();
        assert_eq!(mode.total_buckets(), 4 * 9 * 5);
        // combine_bucket_index の合成式そのものを直接検証 (盤面はダミーで hand=0/king=0 側)。
        let board = ShogiBoard {
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        let idx_r0 = combine_bucket_index(&mode, &board, None, Some(0));
        let idx_r1 = combine_bucket_index(&mode, &board, None, Some(1));
        assert_eq!(idx_r1 - idx_r0, 1, "router must be the least-significant digit");
    }

    #[test]
    fn wsb_adds_one_shared_bucket() {
        let mode = BucketMode::parse("hand4_k3k3_progress8_wsb").unwrap();
        assert!(mode.shared_bucket);
        let selectable = 4 * 9 * 8;
        assert_eq!(mode.selectable_buckets(), selectable);
        assert_eq!(mode.total_buckets(), selectable + 1);
        assert_eq!(mode.shared_bucket_index(), Some(selectable));
        assert_eq!(mode.canonical_token(), "hand4_k3k3_progress8_wsb");
    }

    #[test]
    fn wsb_alone_is_none_plus_shared() {
        let mode = BucketMode::parse("wsb").unwrap();
        assert_eq!(mode.selectable_buckets(), 1);
        assert_eq!(mode.total_buckets(), 2);
        assert_eq!(mode.shared_bucket_index(), Some(1));
        assert_eq!(mode.canonical_token(), "wsb");
    }

    #[test]
    fn wsb_combines_with_router() {
        let mode = BucketMode::parse("k3k3_routerkpabs9_wsb").unwrap();
        assert_eq!(mode.selectable_buckets(), 9 * 9);
        assert_eq!(mode.total_buckets(), 9 * 9 + 1);
        // combine_bucket_index は wsb の有無に関わらず selectable の範囲のみを返す。
        let board = ShogiBoard {
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        let idx = combine_bucket_index(&mode, &board, None, Some(3));
        assert!(idx < mode.selectable_buckets());
    }

    #[test]
    fn wsb_must_be_last_token() {
        assert!(BucketMode::parse("wsb_k3k3").is_err());
        assert!(BucketMode::parse("k3k3_wsb_progress4").is_err());
        assert!(BucketMode::parse("k3k3_progress4_wsb").is_ok());
    }

    #[test]
    fn no_shared_bucket_index_without_wsb() {
        let mode = BucketMode::parse("k3k3").unwrap();
        assert_eq!(mode.shared_bucket_index(), None);
        assert_eq!(mode.total_buckets(), mode.selectable_buckets());
    }
}
