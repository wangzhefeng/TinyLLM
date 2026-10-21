"""排行榜测试：Bradley-Terry 拟合与 Elo 更新（reasoning-from-scratch chF/03）。"""
import math
import unittest


class TestBradleyTerry(unittest.TestCase):
    """Bradley-Terry：Adam 拟合 -logsigmoid NLL + 400/log(10) 缩放到 Elo 尺度。"""

    def _fit(self, votes, **kwargs):
        from post_training.reasoning.evaluation.judgement_based_eval.bradley_terry_leaderboard import (
            fit_bradley_terry,
        )
        return dict(fit_bradley_terry(votes, **kwargs))

    def test_transitive_dominance_ordering(self):
        """A 全胜 B、B 全胜 C -> 排序 A > B > C。"""
        votes = [
            ("A", "B"), ("A", "B"), ("A", "C"),
            ("B", "C"), ("B", "C"),
        ]
        ratings = self._fit(votes, iterations=400)
        self.assertGreater(ratings["A"], ratings["B"])
        self.assertGreater(ratings["B"], ratings["C"])

    def test_scores_centered_around_1000(self):
        """书原版把分数中心平移到 1000：均值应接近 1000。"""
        votes = [("A", "B"), ("A", "C"), ("B", "C"), ("A", "B")]
        ratings = self._fit(votes, iterations=400)
        mean = sum(ratings.values()) / len(ratings)
        self.assertAlmostEqual(mean, 1000.0, delta=1.0)

    def test_equal_opponents_tie(self):
        """互有胜负时两者强度应接近（对称对局 -> 近似平分）。"""
        votes = [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")]
        ratings = self._fit(votes, iterations=600)
        self.assertAlmostEqual(ratings["A"], ratings["B"], delta=25.0)

    def test_returns_sorted_descending_list(self):
        from post_training.reasoning.evaluation.judgement_based_eval.bradley_terry_leaderboard import (
            fit_bradley_terry,
        )
        votes = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
        leaderboard = fit_bradley_terry(votes, iterations=300)
        self.assertIsInstance(leaderboard, list)
        scores = [s for _, s in leaderboard]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_deterministic_with_seed(self):
        votes = [("A", "B"), ("B", "C"), ("A", "C"), ("B", "C")]
        r1 = self._fit(votes, seed=7)
        r2 = self._fit(votes, seed=7)
        self.assertEqual(r1, r2)


class TestEloLeaderboard(unittest.TestCase):
    """Elo：llm_leaderboards.elo_ratings 的行为 + elo_leaderboard 教学辅助。"""

    def test_elo_ratings_winner_gains_loser_loses(self):
        from post_training.reasoning.evaluation.judgement_based_eval.llm_leaderboards import (
            elo_ratings,
        )
        votes = [("M1", "M2")]
        ratings = elo_ratings(votes, k_factor=32, initial_rating=1000)
        self.assertGreater(ratings["M1"], 1000)
        self.assertLess(ratings["M2"], 1000)
        # 单场对局总分守恒（零和转移）
        self.assertAlmostEqual(ratings["M1"] + ratings["M2"], 2000.0, places=6)

    def test_elo_dominant_model_ranks_first(self):
        from post_training.reasoning.evaluation.judgement_based_eval.llm_leaderboards import (
            elo_ratings,
        )
        votes = [
            ("GPT", "Claude"), ("GPT", "Llama"),
            ("Claude", "Llama"), ("GPT", "Claude"),
        ]
        ratings = elo_ratings(votes, k_factor=32, initial_rating=1000)
        self.assertGreater(ratings["GPT"], ratings["Claude"])
        self.assertGreater(ratings["Claude"], ratings["Llama"])

    def test_expected_score_symmetric_complement(self):
        from post_training.reasoning.evaluation.judgement_based_eval.elo_leaderboard import (
            expected_score,
        )
        self.assertEqual(expected_score(1000, 1000), 0.5)
        self.assertAlmostEqual(
            expected_score(1400, 1000) + expected_score(1000, 1400),
            1.0, places=9,
        )
        # 400 分差对应约 10 倍优势 -> 期望胜率约 0.909
        self.assertAlmostEqual(expected_score(1400, 1000), 10 / 11, places=9)

    def test_update_elo_equal_ratings(self):
        from post_training.reasoning.evaluation.judgement_based_eval.elo_leaderboard import (
            update_elo,
        )
        new_w, new_l = update_elo(1000, 1000, k_factor=32)
        self.assertAlmostEqual(new_w, 1016.0, places=6)   # +K*(1-0.5)
        self.assertAlmostEqual(new_l, 984.0, places=6)
        self.assertAlmostEqual(new_w + new_l, 2000.0, places=6)

    def test_update_elo_strong_beats_weak_small_change(self):
        """强者胜弱者时更新幅度应小于均势对局（Elo 的核心性质）。"""
        from post_training.reasoning.evaluation.judgement_based_eval.elo_leaderboard import (
            update_elo,
        )
        new_w, _ = update_elo(1600, 1000, k_factor=32)
        self.assertLess(new_w - 1600, 16.0)

    def test_reexport_identity(self):
        """elo_leaderboard.elo_ratings 必须与 llm_leaderboards 的实现同源。"""
        from post_training.reasoning.evaluation.judgement_based_eval import (
            elo_leaderboard,
            llm_leaderboards,
        )
        self.assertIs(elo_leaderboard.elo_ratings, llm_leaderboards.elo_ratings)


if __name__ == "__main__":
    unittest.main()
