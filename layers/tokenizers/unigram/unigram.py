# -*- coding: utf-8 -*-

# ***************************************************
# * File        : unigram.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 从零实现的 Unigram（SentencePiece 风格）分词器：
# *               种子词表（全部字符 + 高频子串）-> EM 迭代重估 piece 概率
# *               -> 按 log-likelihood 损失剪枝到目标词表大小。
# *               纯 Python，无第三方依赖，仅用于教学。
# * Link        : https://arxiv.org/abs/1804.10959
# * Requirement : 无第三方依赖
# ***************************************************

__all__ = ["UnigramTokenizer"]

# python libraries
from pathlib import Path
import json
import math
from collections import Counter

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class UnigramTokenizer:
    """
    Unigram language model tokenizer (SentencePiece style), from scratch.

    训练流程：
    1. 种子词表：语料中全部单字符 + 基于 n-gram 统计的高频子串；
    2. EM 迭代：E 步用 Viterbi 最优切分累计各 piece 的期望计数，
       M 步按计数重估 piece 概率（hard-EM，教学简化）；
    3. 剪枝：按「删除该 piece 造成的 log-likelihood 损失」从小到大剪枝，
       直到达到目标词表大小；单字符与 unk 永不剪枝。

    编码用 Viterbi 最优切分（log 概率），复杂度 O(L * max_piece_len)。
    """

    # TokenizerProtocol 属性：Unigram 仅 unk 特殊 token，无 eos/pad
    eos_token_id = None
    pad_token_id = None

    def __init__(self, vocab_size: int = 8000, unk_token: str = "<unk>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        # piece 列表，id 即下标；id 0 固定为 unk
        self.pieces = [unk_token]
        # piece -> id
        self.vocab = {unk_token: 0}
        # piece -> log 概率（与 self.pieces 同序）
        self.log_probs = [float("-inf")]
        # 训练语料中最长 piece 的字符长度，Viterbi 的搜索窗口
        self.max_piece_len = 1

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------
    @property
    def unk_id(self) -> int:
        return 0

    @property
    def n_vocab(self) -> int:
        return len(self.pieces)

    def token_to_id(self, token: str):
        """token -> id；未登录 token 返回 unk id。"""
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        """id -> token；越界 id 抛 ValueError。"""
        if not 0 <= token_id < len(self.pieces):
            raise ValueError(f"Token ID {token_id} not found in vocab.")
        return self.pieces[token_id]

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train(
        self,
        corpus,
        vocab_size: int = None,
        num_em_iterations: int = 3,
        seed_max_piece_len: int = 10,
        min_substr_count: int = 2,
    ):
        """
        在语料上训练 Unigram 词表。

        Args:
            corpus (str or List[str]): 训练语料，字符串或句子列表。
            vocab_size (int): 目标词表大小；None 时用构造时的 vocab_size。
            num_em_iterations (int): 每轮剪枝前的 EM 迭代次数。
            seed_max_piece_len (int): 种子子串的最大字符长度。
            min_substr_count (int): 子串进入种子词表的最低频次。
        """
        target_size = vocab_size if vocab_size is not None else self.vocab_size
        sentences = self._normalize_corpus(corpus)

        # 1) 种子词表：全部单字符 + 高频子串
        char_set = sorted({ch for s in sentences for ch in s})
        required_size = len(char_set) + 1  # +1 for unk
        if target_size < required_size:
            # 词表至少装得下 unk 与全部基础字符，否则无法保证可切分
            target_size = required_size
        self.vocab_size = target_size

        seed_counts = self._count_substrings(
            sentences,
            max_len=seed_max_piece_len,
            min_count=min_substr_count,
        )
        # 控制种子规模，避免朴素实现在大语料上过慢
        seed_budget = max(target_size * 10, 2000)
        seed_pieces = char_set + [
            piece for piece, _ in seed_counts.most_common(seed_budget)
            if len(piece) > 1
        ]

        self.pieces = [self.unk_token] + seed_pieces
        self.vocab = {piece: idx for idx, piece in enumerate(self.pieces)}
        total = sum(seed_counts.values()) or 1
        self.log_probs = [
            math.log((seed_counts.get(piece, 1) / total))
            for piece in seed_pieces
        ]
        self.log_probs.insert(0, float("-inf"))  # unk 不参与切分
        self.max_piece_len = max(len(p) for p in self.pieces)

        # 2) EM 迭代
        expected_counts = None
        for _ in range(num_em_iterations):
            expected_counts = self._e_step(sentences)
            self._m_step(expected_counts)

        # 3) 剪枝到目标大小：每轮删掉损失最低的一批，再跑一次 EM 稳定概率
        protected = {self.unk_token} | set(char_set)
        while len(self.pieces) > target_size:
            losses = self._prune_losses(expected_counts)
            n_remove = max(1, (len(self.pieces) - target_size) // 4)
            removable = [
                (loss, piece) for piece, loss in losses.items()
                if piece not in protected
            ]
            if not removable:
                break
            removable.sort(key=lambda x: x[0])
            drop = {piece for _, piece in removable[:n_remove]}
            self._drop_pieces(drop)
            expected_counts = self._e_step(sentences)
            self._m_step(expected_counts)

        return self

    @staticmethod
    def _normalize_corpus(corpus):
        if isinstance(corpus, str):
            return [line for line in corpus.splitlines() if line.strip()]
        return [s for s in corpus if s and s.strip()]

    @staticmethod
    def _count_substrings(sentences, max_len, min_count):
        """朴素 n-gram 统计：所有长度 1..max_len 的子串频次。"""
        counts = Counter()
        for s in sentences:
            n = len(s)
            for i in range(n):
                upper = min(i + max_len, n)
                for j in range(i + 1, upper + 1):
                    counts[s[i:j]] += 1
        return Counter({
            piece: c for piece, c in counts.items() if c >= min_count
        })

    def _e_step(self, sentences):
        """E 步：Viterbi 最优切分每条句子，累计各 piece 的期望计数。"""
        counts = Counter()
        for s in sentences:
            for piece in self._viterbi_segment(s):
                counts[piece] += 1
        return counts

    def _m_step(self, expected_counts):
        """M 步：按期望计数重估 piece 概率（零计数给极小概率以便剪枝）。"""
        total = sum(expected_counts.values()) or 1
        floor = 1e-12
        self.log_probs = [float("-inf")] + [
            math.log(max(expected_counts.get(piece, 0) / total, floor))
            for piece in self.pieces[1:]
        ]

    def _prune_losses(self, expected_counts):
        """
        每个 piece 的删除损失近似：
        loss(p) = expected_count(p) * (logprob(p) - 不用 p 时 p 的最优切分得分)。
        损失越小，删了对整体 log-likelihood 影响越小，优先剪枝。
        """
        losses = {}
        for idx, piece in enumerate(self.pieces):
            if idx == self.unk_id:
                continue
            count = expected_counts.get(piece, 0)
            alt_score = self._viterbi_score(piece, forbid_id=idx)
            losses[piece] = count * (self.log_probs[idx] - alt_score)
        return losses

    def _drop_pieces(self, drop):
        keep = [
            (piece, logp)
            for piece, logp in zip(self.pieces, self.log_probs)
            if piece not in drop
        ]
        self.pieces = [piece for piece, _ in keep]
        self.log_probs = [logp for _, logp in keep]
        self.vocab = {piece: idx for idx, piece in enumerate(self.pieces)}
        self.max_piece_len = max(len(p) for p in self.pieces[1:]) \
            if len(self.pieces) > 1 else 1

    # ------------------------------------------------------------------
    # Viterbi 切分：O(L * max_piece_len)
    # ------------------------------------------------------------------
    def _viterbi_score(self, text, forbid_id=None):
        """text 的最优切分 log 得分；forbid_id 用于模拟删除某个 piece。"""
        n = len(text)
        neg_inf = float("-inf")
        dp = [neg_inf] * (n + 1)
        dp[0] = 0.0
        max_len = self.max_piece_len
        for i in range(n):
            if dp[i] == neg_inf:
                continue
            upper = min(i + max_len, n)
            for j in range(i + 1, upper + 1):
                idx = self.vocab.get(text[i:j])
                if idx is None or idx == self.unk_id or idx == forbid_id:
                    continue
                score = dp[i] + self.log_probs[idx]
                if score > dp[j]:
                    dp[j] = score
        return dp[n]

    def _viterbi_segment(self, text):
        """text 的最优切分 piece 序列；未登录字符回退为 unk_token。"""
        n = len(text)
        neg_inf = float("-inf")
        dp = [neg_inf] * (n + 1)
        dp[0] = 0.0
        back = [None] * (n + 1)  # back[j] = (起点 i, piece)
        max_len = self.max_piece_len
        for i in range(n):
            if dp[i] == neg_inf:
                continue
            upper = min(i + max_len, n)
            for j in range(i + 1, upper + 1):
                piece = text[i:j]
                idx = self.vocab.get(piece)
                if idx is None or idx == self.unk_id:
                    continue
                score = dp[i] + self.log_probs[idx]
                if score > dp[j]:
                    dp[j] = score
                    back[j] = (i, piece)
            # 单字符未登录时回退 unk，保证任何文本都可切分
            if back[i + 1] is None and self.vocab.get(text[i]) is None:
                dp[i + 1] = dp[i]
                back[i + 1] = (i, self.unk_token)
        if back[n] is None:
            return [self.unk_token] * n
        pieces = []
        j = n
        while j > 0:
            i, piece = back[j]
            pieces.append(piece)
            j = i
        return pieces[::-1]

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def encode(self, text: str):
        """
        文本 -> token IDs（Viterbi 最优切分，未登录字符用 unk id 兜底）。
        """
        if len(self.pieces) <= 1:
            raise ValueError("Tokenizer is not trained or loaded yet.")
        return [self.token_to_id(p) for p in self._viterbi_segment(text)]

    def decode(self, ids) -> str:
        """token IDs -> 文本。"""
        return "".join(self.id_to_token(i) for i in ids)

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------
    def save(self, path):
        """词表与概率存为 JSON。"""
        data = {
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "pieces": self.pieces,
            "log_probs": self.log_probs,
            "max_piece_len": self.max_piece_len,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """从 JSON 加载词表。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            unk_token=data["unk_token"],
        )
        tokenizer.pieces = list(data["pieces"])
        tokenizer.log_probs = [float(x) for x in data["log_probs"]]
        tokenizer.vocab = {piece: idx for idx, piece in enumerate(tokenizer.pieces)}
        tokenizer.max_piece_len = data.get(
            "max_piece_len",
            max(len(p) for p in tokenizer.pieces[1:]) if len(tokenizer.pieces) > 1 else 1,
        )
        return tokenizer




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barks at the fox",
        "a quick brown dog runs fast",
        "the lazy dog sleeps all day",
        "foxes are quick and clever animals",
        "the brown fox is quick",
        "dogs and foxes are animals",
        "the cat sleeps on the mat",
        "a lazy cat sleeps all day",
        "the quick dog jumps over the lazy cat",
    ]

    tokenizer = UnigramTokenizer(vocab_size=100)
    tokenizer.train(corpus)
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")

    text = "the quick brown fox"
    ids = tokenizer.encode(text)
    logger.info(f"ids: {ids}")
    tokens = [tokenizer.id_to_token(i) for i in ids]
    logger.info(f"tokens: {tokens}")
    decoded = tokenizer.decode(ids)
    logger.info(f"decoded: {decoded!r}")

    # 未见过的字符走 unk，不抛异常
    unseen_ids = tokenizer.encode("the quick brown fox!!!")
    logger.info(f"unseen ids: {unseen_ids}")

    # save / load 往返
    save_path = Path("./layers/tokenizers/models/unigram_demo.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    loaded = UnigramTokenizer.load(save_path)
    logger.info(f"reload match: {loaded.encode(text) == ids}")

if __name__ == "__main__":
    main()
