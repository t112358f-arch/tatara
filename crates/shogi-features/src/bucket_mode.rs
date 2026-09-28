//! `--bucket-mode` の複合バケット DSL。
//!
//! YaneuraOu (現行, `architectures/nnue_arch_gen.py` V1.04) の SFNN layer-stack
//! トークン列と **同じ文字列** ・**同じ合成規則** を tatara の学習側でも解釈できるように
//! する。`hand4/16/64/64z/256/1024` ・ `k3k3/k9k9/k9k9z/k13k13z/k21k21/k29k29` ・
//! `progress2/3/4/8/16/32` を `_` 区切りで自由に複合でき (各カテゴリ最大1個)、
//! `routerkpabs<N>` / `routerft<R>ft<R>` (どちらか一方のみ、最大1個) を追加できる。
//!
//! # 合成順序 (`order` トークン)
//!
//! バケットindexの桁の重みを決める合成順序は、デフォルトでは
//! **hand → king → progress → router** の順 (`idx = idx * category_buckets +
//! category_index` を先頭カテゴリから順に繰り返す。実際に使われていない
//! カテゴリはスキップされる)。この順序は `order<letters>` トークン
//! (`H`=hand, `K`=king, `P`=progress, `R`=router の並び、例: `orderPRKH`) で
//! 明示的に変更できる。`order` トークンは、実際に使われているカテゴリ
//! ちょうど全部の順列でなければならず (使われていないカテゴリの文字が
//! 混じっているとエラー)、バケット名の中に最大1個。省略時はデフォルト順序
//! (`orderHKPR` 相当)。`order` トークンはバケットindexの合成順序だけを
//! 変えるものであり、`wsb` の付く位置 (次項) とは独立。
//!
//! # `wsb` (WithSharedBucket) — 直前のバケット単位での共有バケット
//!
//! バケット名の中のどこかに `wsb` を置く (最大1個、文字列上の位置は自由) と、
//! **`wsb` の直前のトークンに対応するカテゴリ以降 (=合成順序上、そのカテゴリの
//! 桁からより下位の桁まで全部) を束ねたブロック** に、そのブロックの選択肢
//! (＝そのブロックに属するカテゴリの合成バケット数の積) に加えてもう1個、
//! 「常に選ばれる共有バケット」を追加する。`wsb` より外側 (合成順序上、より
//! 上位の桁) のカテゴリはそのまま外側の乗数として残る。`wsb` がバケット名の
//! 先頭トークン (＝直前のトークンが無い) の場合は、合成順序の一番外側から
//! 全部をこのブロックとみなす — これは旧仕様 (`wsb` は常にバケット名の末尾、
//! 常に全体で1個だけの共有バケット) と同じ挙動になる。
//!
//! 具体例 (`progress8` と `routerkpabs16` の複合、デフォルト合成順序
//! progress→router):
//! - `progress8_wsb_routerkpabs16`: `wsb` の直前は `progress8`。合成順序上
//!   `progress` は一番外側の桁なので、ブロック = progress×router 全部。
//!   常に有効な共有バケット1個 + 選択バケット `8*16` 個、合計
//!   `1 + 8*16 = 129` バケット。
//! - `progress8_routerkpabs16_wsb`: `wsb` の直前は `routerkpabs16`。ブロック
//!   = router だけ (progress はブロックの外側)。progress の値ごとに
//!   共有バケット1個 + 選択バケット16個を持つので、合計
//!   `8 * (1 + 16) = 136` バケット。
//!
//! (`wsb` 単体、または `order` トークンだけ挟んで `wsb` を先頭に置いた場合は
//! 旧仕様と同じ「グローバルに1個だけの共有バケット」になる。)
//!
//! `wsb` は、その直前のトークンとして (合成順序に関わらず) hand/king/progress/
//! router のいずれのカテゴリトークンの直後にも置ける。`order` トークンの直後に
//! `wsb` を置くことはできない (どのカテゴリを指すか曖昧なため; エラーになる)。
//!
//! 保存形式は YaneuraOu の慣習 (このバケットindexの並び) を正として、tatara旧形式や
//! yaneuraou-privateとは非互換。旧形式からの変換は `net_convert_bucket_layout`
//! (bins/net_convert_bucket_layout) を使う。
//!
//! # 破壊的変更に関する注記
//!
//! 本モジュールの `wsb` の意味論は、旧版 (`wsb` は常にバケット名の末尾のみ許可、
//! 常に「グローバルに1個だけの共有バケット」を追加する仕様) から変更されている。
//! 旧版で `..._wsb` として学習した net の挙動を再現したい場合は、`wsb` を
//! バケット名の**先頭**に置く (`wsb_...`) こと。バケット名の末尾に置く
//! `..._wsb` は、直前のカテゴリ単位の共有バケット (新仕様) に意味が変わる。

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

/// バケットカテゴリの種類 (`order` トークンや `wsb` の付け根の指定に使う)。
/// 実際のサブモード (`HandSubMode` 等) とは別に、「合成順序上のどのカテゴリか」
/// だけを表す軽量な識別子。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Hand,
    King,
    Progress,
    Router,
}

impl Category {
    fn letter(self) -> char {
        match self {
            Category::Hand => 'H',
            Category::King => 'K',
            Category::Progress => 'P',
            Category::Router => 'R',
        }
    }

    fn from_letter(c: char) -> Option<Category> {
        match c.to_ascii_uppercase() {
            'H' => Some(Category::Hand),
            'K' => Some(Category::King),
            'P' => Some(Category::Progress),
            'R' => Some(Category::Router),
            _ => None,
        }
    }
}

/// `order` トークン省略時のデフォルト合成順序 (hand → king → progress → router、
/// 従来の固定順序と同じ)。
pub const DEFAULT_ORDER: [Category; 4] =
    [Category::Hand, Category::King, Category::Progress, Category::Router];

/// パース済みの `--bucket-mode` (複合可能な hand/king/progress/router)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketMode {
    pub hand: Option<HandSubMode>,
    pub king: Option<KingSubMode>,
    /// progress2/3/4/8/16/32 の N。指定なしは None (= 従来のprogress8kpabs系とは別、
    /// 単に progress バケットを使わないことを表す)。
    pub progress: Option<u32>,
    pub router: Option<RouterSubMode>,
    /// バケットindexの合成順序 (桁の重み、最上位桁から順)。`order` トークンで
    /// 明示しない限り `DEFAULT_ORDER`。実際に存在しないカテゴリのエントリは
    /// 無視される。
    pub order: [Category; 4],
    /// `wsb` (WithSharedBucket) の有無。
    pub shared_bucket: bool,
    /// `wsb` がトークン文字列上、直前に置かれていたカテゴリ。`shared_bucket`
    /// が `false` のときは常に `None`。`shared_bucket` が `true` で、かつ
    /// `wsb` が文字列の先頭 (直前トークンなし) だった場合も `None` になり、
    /// この場合は合成順序の最上位桁からブロックが始まる (＝全体で1個だけの
    /// グローバル共有バケット、旧仕様と同じ挙動)。
    pub shared_bucket_after: Option<Category>,
}

impl BucketMode {
    pub const NONE: BucketMode = BucketMode {
        hand: None,
        king: None,
        progress: None,
        router: None,
        order: DEFAULT_ORDER,
        shared_bucket: false,
        shared_bucket_after: None,
    };

    fn category_present(&self, cat: Category) -> bool {
        match cat {
            Category::Hand => self.hand.is_some(),
            Category::King => self.king.is_some(),
            Category::Progress => self.progress.is_some(),
            Category::Router => self.router.is_some(),
        }
    }

    fn category_bucket_count(&self, cat: Category) -> u32 {
        match cat {
            Category::Hand => self.hand.map(HandSubMode::bucket_count).unwrap_or(1),
            Category::King => self.king.map(KingSubMode::bucket_count).unwrap_or(1),
            Category::Progress => self.progress.unwrap_or(1),
            Category::Router => self.router.map(RouterSubMode::bucket_count).unwrap_or(1),
        }
    }

    /// 実際に使われているカテゴリだけを、合成順序 (`order`) 通りに並べたリスト
    /// (最上位桁から順)。
    pub fn present_order(&self) -> Vec<Category> {
        self.order.iter().copied().filter(|&c| self.category_present(c)).collect()
    }

    /// `wsb` のブロック境界 (`present_order()` 中のindex、このindex以降が
    /// 共有バケットのブロックに入る)。`shared_bucket` が `false` のときに
    /// 呼ぶのは呼び出し側のバグなので 0 を返す (使われない前提)。
    fn cut_position(&self, present: &[Category]) -> usize {
        match self.shared_bucket_after {
            None => 0,
            Some(cat) => present.iter().position(|&c| c == cat).unwrap_or(0),
        }
    }

    /// `wsb` のブロックより外側 (合成順序上、より上位の桁) のカテゴリの
    /// 合成バケット数の積。`wsb` 無効時は常に1。
    pub fn outer_buckets(&self) -> u32 {
        if !self.shared_bucket {
            return 1;
        }
        let present = self.present_order();
        let p = self.cut_position(&present);
        present[..p].iter().map(|&c| self.category_bucket_count(c)).product()
    }

    /// `wsb` のブロック (直前のカテゴリ以降、最下位桁まで) の合成バケット数の積
    /// (共有バケット自体の+1は含まない)。`wsb` 無効時は全カテゴリの積
    /// (＝ `selectable_buckets()` と同じ)。
    pub fn inner_buckets(&self) -> u32 {
        let present = self.present_order();
        if !self.shared_bucket {
            return present.iter().map(|&c| self.category_bucket_count(c)).product();
        }
        let p = self.cut_position(&present);
        present[p..].iter().map(|&c| self.category_bucket_count(c)).product()
    }

    /// hand/king/progress/router の合成バケット数 (`wsb` の共有バケットを
    /// 含まない、常に `outer_buckets() * inner_buckets()` に等しい)。
    pub fn selectable_buckets(&self) -> u32 {
        self.present_order().iter().map(|&c| self.category_bucket_count(c)).product()
    }

    /// 総バケット数 (＝実際の重み配列のサイズ)。`wsb` 有効時は
    /// `outer_buckets() * (inner_buckets() + 1)` (`outer_buckets()` 個のブロック
    /// それぞれに共有バケットが1個ずつ挿入される)。`wsb` 無効時は
    /// `selectable_buckets()` と同じ。
    pub fn total_buckets(&self) -> u32 {
        if self.shared_bucket {
            self.outer_buckets() * (self.inner_buckets() + 1)
        } else {
            self.selectable_buckets()
        }
    }

    /// 合成順序 (`present_order()`) に従って各カテゴリのバケット値を
    /// `idx = idx * count + sub` として合成した、隙間の無い密なindex
    /// (`0..selectable_buckets()`)。`wsb` による+1の挿入は反映しない
    /// (`remap_dense_index` / `shared_bucket_index_for` が行う)。
    fn dense_selectable_index(
        &self,
        board: &ShogiBoard,
        progress_bucket: Option<u32>,
        router_bucket: Option<u32>,
    ) -> u32 {
        let mut idx = 0u32;
        for cat in self.present_order() {
            let (count, sub) = match cat {
                Category::Hand => {
                    let h = self.hand.expect("present_order only yields present categories");
                    (h.bucket_count(), hand_bucket(h, board))
                }
                Category::King => {
                    let k = self.king.expect("present_order only yields present categories");
                    (k.bucket_count(), king_bucket(k, board))
                }
                Category::Progress => {
                    let p = self.progress.expect("present_order only yields present categories");
                    (p, progress_bucket.unwrap_or(0).min(p - 1))
                }
                Category::Router => {
                    let r = self.router.expect("present_order only yields present categories");
                    let rb = router_bucket.unwrap_or(0).min(r.bucket_count() - 1);
                    (r.bucket_count(), rb)
                }
            };
            idx = idx * count + sub;
        }
        idx
    }

    /// `dense_selectable_index` の密indexを、実際の重み配列index
    /// (`0..total_buckets()`、`wsb` 有効時は各外側ブロックに共有バケット1個が
    /// 挿入された形) に変換する。
    fn remap_dense_index(&self, dense_idx: u32) -> u32 {
        if !self.shared_bucket {
            return dense_idx;
        }
        let inner = self.inner_buckets();
        let outer_idx = dense_idx / inner;
        let inner_idx = dense_idx % inner;
        outer_idx * (inner + 1) + inner_idx
    }

    /// `dense_selectable_index` の密indexが属する外側ブロックの共有バケットの
    /// 実際の重み配列index。`wsb` 無効時は `None`。
    fn shared_bucket_index_for_dense(&self, dense_idx: u32) -> Option<u32> {
        if !self.shared_bucket {
            return None;
        }
        let inner = self.inner_buckets();
        let outer_idx = dense_idx / inner;
        Some(outer_idx * (inner + 1) + inner)
    }

    /// 局面に対応する、選択バケット側の実際の重み配列index
    /// (`0..total_buckets()`)。
    pub fn combine_bucket_index(
        &self,
        board: &ShogiBoard,
        progress_bucket: Option<u32>,
        router_bucket: Option<u32>,
    ) -> u32 {
        self.remap_dense_index(self.dense_selectable_index(board, progress_bucket, router_bucket))
    }

    /// 局面に対応する、共有バケット側の実際の重み配列index (同じ外側ブロック内
    /// の共有バケットindex)。`wsb` 無効時は `None`。`outer_buckets() == 1`
    /// (`wsb` が先頭トークン、または `wsb` 単体) のときは局面によらず常に同じ
    /// 1つのindexを返す (旧仕様のグローバル共有バケットと同じ)。
    pub fn shared_bucket_index_for_board(
        &self,
        board: &ShogiBoard,
        progress_bucket: Option<u32>,
        router_bucket: Option<u32>,
    ) -> Option<u32> {
        self.shared_bucket_index_for_dense(self.dense_selectable_index(
            board,
            progress_bucket,
            router_bucket,
        ))
    }

    /// `wsb` 有効時に、局面によらず常に同じ1個の共有バケットindexだけを持つか
    /// (＝ `outer_buckets() == 1`)。GPU学習側の高速経路 (全行が同一bucketである
    /// 前提の cuBLAS 直接呼び出し) が使えるかどうかの判定に使う。
    pub fn has_single_global_shared_bucket(&self) -> bool {
        self.shared_bucket && self.outer_buckets() == 1
    }

    /// `has_single_global_shared_bucket()` が `true` のときの、その唯一の共有
    /// バケットの実際の重み配列index (常に `total_buckets() - 1`)。それ以外
    /// (wsb無効、または `outer_buckets() > 1` で局面ごとに共有バケットindexが
    /// 変わる場合) は `None`。GPU学習側の「バッチ全行が同じ共有バケット」を
    /// 前提にした定数broadcast経路 (`GpuWorkspace` construction) が使う。
    pub fn global_shared_bucket_index(&self) -> Option<u32> {
        if self.has_single_global_shared_bucket() {
            Some(self.inner_buckets())
        } else {
            None
        }
    }

    /// YaneuraOu生成器の `NNUE_SFNN_*` マクロと同じ形の正準トークン列
    /// (`hand64z_k9k9_progress4_routerkpabs5` のように、hand→king→progress→routerの順、
    /// `order` が非デフォルトならそれも含む)。`wsb` は元々置かれていた直前カテゴリの
    /// 直後に挿入する (直前カテゴリが無かった場合は先頭)。空 (バケット無し) の
    /// ときは `"NONE"` (`wsb` 単独のときは `"WSB"`)。
    pub fn canonical_token(&self) -> String {
        let mut parts = Vec::new();
        if self.order != DEFAULT_ORDER && !self.selectable_buckets_is_trivial() {
            parts.push(order_token(&self.present_order()));
        }
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
            match self.shared_bucket_after {
                None => parts.insert(0, "wsb".to_string()),
                Some(cat) => {
                    let token = match cat {
                        Category::Hand => self.hand.map(|h| h.token().to_string()),
                        Category::King => self.king.map(|k| k.token().to_string()),
                        Category::Progress => self.progress.map(|p| format!("progress{p}")),
                        Category::Router => self.router.map(RouterSubMode::token),
                    };
                    match token.and_then(|t| parts.iter().position(|p| *p == t)) {
                        Some(pos) => parts.insert(pos + 1, "wsb".to_string()),
                        None => parts.push("wsb".to_string()),
                    }
                }
            }
        }
        if parts.is_empty() {
            "NONE".to_string()
        } else {
            parts.join("_")
        }
    }

    /// `order` トークンを出す意味が無いケース (使われているカテゴリが0か1個)
    /// を検出する — 順序を並べ替えても何も変わらないので、canonical化のとき
    /// ノイズになる `order` トークンを省く。
    fn selectable_buckets_is_trivial(&self) -> bool {
        self.present_order().len() <= 1
    }

    /// `--bucket-mode` 文字列 (`_` 区切りトークン列、順不同 [`order`/`wsb` の
    /// 相対位置を除く]、大文字小文字不問) をパースする。各カテゴリ
    /// (hand/king/progress/router) は最大1個、router系
    /// (routerkpabs/routerft{R}ft{R}) は互いに排他。`order<letters>` は最大1個で、
    /// 実際に使われているカテゴリちょうど全部の順列でなければならない。`wsb` は
    /// 最大1個、どのカテゴリトークンの直後にも置ける (文字列上の位置は自由。
    /// ただし `order` トークンの直後には置けない)。
    ///
    /// 空文字列 / `"none"` はバケット無し (`BucketMode::NONE`、常に bucket 0 の1バケット)
    /// を表す。
    pub fn parse(spec: &str) -> Result<BucketMode, String> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("none") {
            return Ok(BucketMode::NONE);
        }

        let mut mode = BucketMode::NONE;
        let mut explicit_order: Option<Vec<Category>> = None;
        let mut wsb_after_token: Option<String> = None;
        let mut wsb_seen = false;
        let raw_tokens: Vec<&str> = spec.split('_').filter(|t| !t.is_empty()).collect();
        for (i, raw_token) in raw_tokens.iter().enumerate() {
            let token = normalize_token(raw_token);
            if token == "WSB" {
                if wsb_seen {
                    return Err(format!("wsb (WithSharedBucket) may appear at most once in bucket-mode {spec:?}"));
                }
                wsb_seen = true;
                mode.shared_bucket = true;
                wsb_after_token = i.checked_sub(1).map(|prev| normalize_token(raw_tokens[prev]));
                continue;
            }
            if let Some(order) = parse_order_token(&token)? {
                if explicit_order.is_some() {
                    return Err(format!("order<letters> may appear at most once in bucket-mode {spec:?}"));
                }
                explicit_order = Some(order);
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
                "unknown bucket-mode token {raw_token:?} in {spec:?}; expected hand4/16/64/64z/256/1024, k3k3/k9k9/k9k9z/k13k13z/k21k21/k29k29, progress2/3/4/8/16/32, routerkpabs<N>, routerft<R>ft<R>, order<letters>, or wsb"
            ));
        }

        // order トークンの検証: 実際に使われているカテゴリちょうど全部の順列であること
        // (順列である事は parse_order_token 側で重複禁止により保証済みなので、ここでは
        // 集合として一致するかだけ見ればよい)。
        if let Some(order) = explicit_order {
            let mut present: Vec<Category> = [Category::Hand, Category::King, Category::Progress, Category::Router]
                .into_iter()
                .filter(|&c| mode.category_present(c))
                .collect();
            present.sort_by_key(|c| c.letter());
            let mut given_sorted = order.clone();
            given_sorted.sort_by_key(|c| c.letter());
            if given_sorted != present {
                return Err(format!(
                    "order<letters> must be a permutation of exactly the categories present in bucket-mode {spec:?}"
                ));
            }
            // `order` フィールドは常に4要素の配列として持つ。使われていない
            // カテゴリは (present_order() でどのみち除外されるので) デフォルト順で
            // 末尾に埋めておくだけでよい。
            let mut order4 = order;
            for cat in DEFAULT_ORDER {
                if !order4.contains(&cat) {
                    order4.push(cat);
                }
            }
            mode.order = [order4[0], order4[1], order4[2], order4[3]];
        }

        // wsb の直前トークンをカテゴリに解決する。
        if mode.shared_bucket {
            mode.shared_bucket_after = match wsb_after_token {
                None => None,
                Some(prev) if parse_order_token(&prev).ok().flatten().is_some() => {
                    return Err(format!(
                        "wsb (WithSharedBucket) cannot immediately follow an order<letters> token in bucket-mode {spec:?}"
                    ));
                }
                Some(prev) => Some(category_of_token(&prev).ok_or_else(|| {
                    format!("wsb (WithSharedBucket) must immediately follow a hand/king/progress/router token in bucket-mode {spec:?}")
                })?),
            };
        }

        Ok(mode)
    }
}

/// `order<letters>` トークンの正準表記 (`order` + 実際に使われているカテゴリの
/// 文字を合成順序通りに並べたもの)。
fn order_token(present_order: &[Category]) -> String {
    let mut s = String::from("order");
    for c in present_order {
        s.push(c.letter());
    }
    s
}

/// `order<letters>` トークンの中身をパースする。`letters` は使われている
/// カテゴリの個数ぶんだけ (1〜4文字) 指定すればよい (例: progress/router しか
/// 使わないなら `orderRP` の2文字でよい)。実際にその通り「使われている
/// カテゴリちょうど全部」になっているかどうかは、他のトークンを全部見終わった
/// 後で `BucketMode::parse` 側が検証する。
fn parse_order_token(token: &str) -> Result<Option<Vec<Category>>, String> {
    let Some(rest) = token.strip_prefix("ORDER") else {
        return Ok(None);
    };
    let len = rest.chars().count();
    if len == 0 || len > 4 {
        return Err(format!(
            "order<letters> must list 1 to 4 letters (a permutation of a subset of H/K/P/R), got \"order{rest}\""
        ));
    }
    let mut cats = Vec::with_capacity(len);
    let mut seen = std::collections::HashSet::new();
    for c in rest.chars() {
        let cat = Category::from_letter(c).ok_or_else(|| {
            format!("order<letters> letters must be H/K/P/R, got \"order{rest}\"")
        })?;
        if !seen.insert(c.to_ascii_uppercase()) {
            return Err(format!("order<letters> must not repeat a letter, got \"order{rest}\""));
        }
        cats.push(cat);
    }
    Ok(Some(cats))
}

/// トークン文字列 (正規化済み、大文字) がどのカテゴリに属するかを判定する
/// (`wsb` の直前トークンをカテゴリへ解決するのに使う)。
fn category_of_token(token: &str) -> Option<Category> {
    if parse_hand_token(token).is_some() {
        return Some(Category::Hand);
    }
    if parse_king_token(token).is_some() {
        return Some(Category::King);
    }
    if parse_progress_token(token).is_some() {
        return Some(Category::Progress);
    }
    if parse_router_token(token).ok().flatten().is_some() {
        return Some(Category::Router);
    }
    None
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

/// 旧APIとの互換のため残しているフリー関数版。`mode.combine_bucket_index(...)`
/// に委譲する (返り値は実際の重み配列index `0..mode.total_buckets()`)。
pub fn combine_bucket_index(
    mode: &BucketMode,
    board: &ShogiBoard,
    progress_bucket: Option<u32>,
    router_bucket: Option<u32>,
) -> u32 {
    mode.combine_bucket_index(board, progress_bucket, router_bucket)
}

/// 旧APIとの互換のため残しているフリー関数版。
/// `mode.shared_bucket_index_for_board(...)` に委譲する。
pub fn shared_bucket_index_for_board(
    mode: &BucketMode,
    board: &ShogiBoard,
    progress_bucket: Option<u32>,
    router_bucket: Option<u32>,
) -> Option<u32> {
    mode.shared_bucket_index_for_board(board, progress_bucket, router_bucket)
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
    fn router_always_last_by_default_order() {
        // デフォルト合成順序 (hand→king→progress→router) では router が最下位桁になる。
        let mode = BucketMode::parse("hand4_k3k3_routerkpabs5").unwrap();
        assert_eq!(mode.total_buckets(), 4 * 9 * 5);
        let board = ShogiBoard {
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        let idx_r0 = combine_bucket_index(&mode, &board, None, Some(0));
        let idx_r1 = combine_bucket_index(&mode, &board, None, Some(1));
        assert_eq!(idx_r1 - idx_r0, 1, "router must be the least-significant digit by default");
    }

    fn dummy_board() -> ShogiBoard {
        ShogiBoard {
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        }
    }

    // ------------------------------------------------------------------
    // wsb: 直前のバケット単位での共有バケット (新仕様)
    // ------------------------------------------------------------------

    #[test]
    fn wsb_before_router_is_globally_shared() {
        // progress8_wsb_routerkpabs16: wsb の直前は progress。デフォルト合成順序
        // progress→router では progress が最上位桁 (=全体の先頭) なので、
        // ブロック = progress×router 全部。共有バケットは1個だけ (グローバル)。
        let mode = BucketMode::parse("progress8_wsb_routerkpabs16").unwrap();
        assert!(mode.shared_bucket);
        assert_eq!(mode.shared_bucket_after, Some(Category::Progress));
        assert_eq!(mode.selectable_buckets(), 8 * 16);
        assert_eq!(mode.outer_buckets(), 1);
        assert_eq!(mode.inner_buckets(), 8 * 16);
        assert_eq!(mode.total_buckets(), 1 + 8 * 16);
        assert!(mode.has_single_global_shared_bucket());

        // 局面によらず共有バケットindexは常に同じ (selectable_buckets() の値)。
        let b1 = dummy_board();
        let mut b2 = dummy_board();
        b2.black_king_sq = Square::new(0, 8);
        assert_eq!(
            mode.shared_bucket_index_for_board(&b1, Some(0), Some(0)),
            Some(8 * 16)
        );
        assert_eq!(
            mode.shared_bucket_index_for_board(&b2, Some(7), Some(15)),
            Some(8 * 16)
        );
    }

    #[test]
    fn wsb_after_router_is_shared_per_progress() {
        // progress8_routerkpabs16_wsb: wsb の直前は router。ブロック = router
        // だけなので、progress の値ごとに共有バケット1個+選択16個を持つ。
        let mode = BucketMode::parse("progress8_routerkpabs16_wsb").unwrap();
        assert!(mode.shared_bucket);
        assert_eq!(mode.shared_bucket_after, Some(Category::Router));
        assert_eq!(mode.selectable_buckets(), 8 * 16);
        assert_eq!(mode.outer_buckets(), 8);
        assert_eq!(mode.inner_buckets(), 16);
        assert_eq!(mode.total_buckets(), 8 * (1 + 16));
        assert!(!mode.has_single_global_shared_bucket());

        // progress の値が違えば共有バケットindexも違う (progressブロックごとに1個)。
        let board = dummy_board();
        let shared_p0 = mode.shared_bucket_index_for_board(&board, Some(0), Some(3)).unwrap();
        let shared_p1 = mode.shared_bucket_index_for_board(&board, Some(1), Some(9)).unwrap();
        assert_eq!(shared_p0, 0 * 17 + 16);
        assert_eq!(shared_p1, 1 * 17 + 16);
        // 同じ progress なら router の値が違っても共有バケットindexは同じ。
        let shared_p0_again = mode.shared_bucket_index_for_board(&board, Some(0), Some(12)).unwrap();
        assert_eq!(shared_p0, shared_p0_again);

        // 選択側のindexは、progressブロック内で router の値がそのまま (0..15) 使われる。
        assert_eq!(mode.combine_bucket_index(&board, Some(0), Some(3)), 0 * 17 + 3);
        assert_eq!(mode.combine_bucket_index(&board, Some(1), Some(9)), 1 * 17 + 9);
    }

    #[test]
    fn wsb_alone_is_globally_shared() {
        let mode = BucketMode::parse("wsb").unwrap();
        assert_eq!(mode.shared_bucket_after, None);
        assert_eq!(mode.selectable_buckets(), 1);
        assert_eq!(mode.total_buckets(), 2);
        assert!(mode.has_single_global_shared_bucket());
        assert_eq!(mode.canonical_token(), "wsb");
    }

    #[test]
    fn wsb_first_token_is_globally_shared_like_old_behavior() {
        // 先頭に wsb を置くと、旧仕様 (常にバケット名の末尾、常にグローバル
        // 1個だけの共有バケット) と同じ挙動になる。
        let mode = BucketMode::parse("wsb_hand4_k3k3_progress8").unwrap();
        assert_eq!(mode.shared_bucket_after, None);
        let selectable = 4 * 9 * 8;
        assert_eq!(mode.selectable_buckets(), selectable);
        assert_eq!(mode.total_buckets(), selectable + 1);
        assert!(mode.has_single_global_shared_bucket());
    }

    #[test]
    fn wsb_may_appear_anywhere_but_only_once() {
        assert!(BucketMode::parse("wsb_k3k3").is_ok());
        assert!(BucketMode::parse("k3k3_wsb_progress4").is_ok());
        assert!(BucketMode::parse("k3k3_progress4_wsb").is_ok());
        assert!(BucketMode::parse("k3k3_wsb_progress4_wsb").is_err());
    }

    #[test]
    fn wsb_cannot_follow_order_token() {
        assert!(BucketMode::parse("orderPRKH_wsb_hand4_k3k3_progress8_routerkpabs4").is_err());
    }

    #[test]
    fn no_shared_bucket_index_without_wsb() {
        let mode = BucketMode::parse("k3k3").unwrap();
        let board = dummy_board();
        assert_eq!(mode.shared_bucket_index_for_board(&board, None, None), None);
        assert_eq!(mode.total_buckets(), mode.selectable_buckets());
    }

    // ------------------------------------------------------------------
    // order: 合成順序を任意に変更する
    // ------------------------------------------------------------------

    #[test]
    fn order_token_changes_digit_significance() {
        // デフォルト (progress→router) では router が最下位桁。
        let default_order = BucketMode::parse("progress4_routerkpabs5").unwrap();
        assert_eq!(default_order.order, DEFAULT_ORDER);
        let board = dummy_board();
        let a = default_order.combine_bucket_index(&board, Some(0), Some(0));
        let b = default_order.combine_bucket_index(&board, Some(0), Some(1));
        assert_eq!(b - a, 1, "router should be least-significant by default");

        // orderRP (router→progress) で桁の重みを逆転させると、今度は progress
        // が最下位桁になる。
        let swapped = BucketMode::parse("progress4_routerkpabs5_orderRP").unwrap();
        assert_eq!(swapped.order[..2], [Category::Router, Category::Progress]);
        let a2 = swapped.combine_bucket_index(&board, Some(0), Some(0));
        let b2 = swapped.combine_bucket_index(&board, Some(1), Some(0));
        assert_eq!(b2 - a2, 1, "progress should be least-significant after orderRP");
        assert_eq!(swapped.total_buckets(), 4 * 5);
    }

    #[test]
    fn order_token_must_be_permutation_of_present_categories() {
        // hand/king が無いのに order トークンに H/K を含めるのはエラー。
        assert!(BucketMode::parse("progress4_routerkpabs5_orderHKPR").is_err());
        // 使われているカテゴリを全部含まないのもエラー。
        assert!(BucketMode::parse("hand4_k3k3_progress4_orderPK").is_err());
        // 重複した文字もエラー。
        assert!(BucketMode::parse("progress4_routerkpabs5_orderPPPP").is_err());
    }

    #[test]
    fn order_combines_with_wsb() {
        // orderRP + wsb: router が最上位、progress が最下位。wsb の直前は
        // progress (router→progressの順で書いたとき)、合成順序上 progress は
        // 最下位桁 (かつ唯一そこから下のカテゴリ) なので、router の値ごとに
        // 共有バケット1個+選択4個 (progressのbucket数) を持つ。
        let mode = BucketMode::parse("routerkpabs5_progress4_wsb_orderRP").unwrap();
        assert_eq!(mode.order[..2], [Category::Router, Category::Progress]);
        assert_eq!(mode.shared_bucket_after, Some(Category::Progress));
        assert_eq!(mode.outer_buckets(), 5); // router
        assert_eq!(mode.inner_buckets(), 4); // progress
        assert_eq!(mode.total_buckets(), 5 * (4 + 1));
    }

    #[test]
    fn total_buckets_matches_example_from_spec() {
        // ユーザーの例そのもの: progress8_WSB_router16 / progress8_router16_WSB
        // (router は routerkpabs16 相当)。
        let a = BucketMode::parse("progress8_wsb_routerkpabs16").unwrap();
        assert_eq!(a.total_buckets(), 1 + 8 * 16);
        let b = BucketMode::parse("progress8_routerkpabs16_wsb").unwrap();
        assert_eq!(b.total_buckets(), 8 * (1 + 16));
    }
}
