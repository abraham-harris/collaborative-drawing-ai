"""Tests for Greedy mode, the ink limit, turn reports, and batched SketchRNN sampling.

No network or AuDrA checkpoint needed: models and scorers are faked.
"""

import random
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import ai_model
from drawing_models import MODELS, DrawingContext, greedy_model, strokes_from_actions
from drawing_models.sketch_rnn import SketchRNN
from drawing_models.strokes import Stroke, ink_length, path_length, trim_to_length


def line_action(x0, y0, x1, y1):
    return {"tool": "brush", "color": "#000000", "size": 4, "opacity": 1, "filled": False,
            "points": [{"x": x0, "y": y0}, {"x": x1, "y": y1}]}


def blank():
    return Image.new("RGBA", (800, 600), "white")


def tiny_sketch_rnn(seed=0, units=8):
    """A SketchRNN with random weights: draws nonsense, but exercises the sampler."""
    r = np.random.default_rng(seed)
    info = {"name": "test", "scale_factor": 50.0, "max_seq_len": 40}
    return SketchRNN(info, r.normal(0, 0.5, (units, 123)).astype(np.float32),
                     r.normal(0, 0.5, 123).astype(np.float32),
                     r.normal(0, 0.5, (5 + units, 4 * units)).astype(np.float32),
                     np.zeros(4 * units, np.float32))


class InkLimitTests(unittest.TestCase):
    def test_trim_cuts_exactly_at_budget(self):
        lines = [[(0, 0), (200, 0)], [(0, 10), (200, 10)], [(0, 20), (100, 20)]]
        kept = trim_to_length(lines, 300)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[1][-1], (100, 10))
        self.assertAlmostEqual(sum(path_length(p) for p in kept), 300)

    def test_greedy_returns_one_drawing_using_exactly_the_human_ink(self):
        def long_drawing(ctx):
            return [Stroke(points=[(10, 10), (790, 10)]), Stroke(points=[(10, 50), (790, 50)])], "drew"

        report = {}
        actions = [line_action(100, 300, 220, 300)]  # 120 px of human ink
        with patch.dict(MODELS, {"greedy": long_drawing}):
            results = ai_model.generate_drawing(blank(), actions, mode="greedy", scorer=None, report=report)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(report["humanInk"], 120)
        self.assertAlmostEqual(report["inkBudget"], 120)
        self.assertAlmostEqual(ink_length(strokes_from_actions(results[0][2])), 120, places=3)

    def test_no_human_ink_means_no_ai_ink_without_error(self):
        called = []
        with patch.dict(MODELS, {"greedy": lambda ctx: called.append(1) or ([], "")}):
            results = ai_model.generate_drawing(blank(), [], mode="greedy", scorer=None)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][2], [])
        self.assertIn("no ink", results[0][1])
        self.assertEqual(called, [])

    def test_greedy_reads_the_whole_drawing_but_spends_only_this_turns_ink(self):
        seen = []

        def record(ctx):
            seen.append((len(ctx.strokes), ctx.ink_budget, len(ctx.subjects)))
            return [Stroke(points=[(0, 0), (500, 0)])], "drew"

        earlier = [line_action(0, 0, 500, 0), line_action(0, 10, 500, 10)]
        current = [line_action(0, 20, 100, 20)]
        with patch.dict(MODELS, {"greedy": record}):
            ai_model.generate_drawing(blank(), current, context_actions=earlier + current, mode="greedy",
                                      scorer=None, rng=random.Random(5))
        self.assertEqual(seen, [(3, 100, ai_model.GREEDY_SUBJECTS)])

    def test_greedy_keeps_subjects_hidden(self):
        report = {}
        with patch.dict(MODELS, {"greedy": lambda ctx: (ctx.report.update(chosenSubject="cat")
                                                        or [Stroke(points=[(0, 0), (50, 0)])], "drew")}):
            ai_model.generate_drawing(blank(), [line_action(0, 0, 100, 0)], mode="greedy", scorer=None,
                                      report=report)
        self.assertNotIn("chosenSubject", report)


class GreedySelectionTests(unittest.TestCase):
    def make_ctx(self, scorer):
        strokes = strokes_from_actions([line_action(300, 300, 500, 300)])
        return DrawingContext.build(blank(), strokes, "cat", rng=random.Random(0), ink_budget=200,
                                    iterations=4, subjects=["cat"], scorer=scorer)

    def fake_candidates(self, subject, n, temps, ctx, human, budget, nrng):
        return [{"subject": subject, "strategy": "complete", "extends": None, "temperature": t,
                 "lines": [[(100, 100 + 50 * i), (300, 100 + 50 * i)]], "ink": 200.0}
                for i, t in enumerate(temps)]

    def test_keeps_highest_scoring_candidate_and_reports_average(self):
        ctx = self.make_ctx(lambda images: [0.2, 0.7, 0.4, 0.5][: len(images)])
        with patch.object(greedy_model, "_generate_candidates", self.fake_candidates):
            strokes, _ = greedy_model.generate(ctx)
        g = ctx.report["greedy"]
        self.assertTrue(all("subject" not in c for c in g["candidates"]))
        self.assertEqual(g["chosenIndex"], 1)
        self.assertAlmostEqual(g["chosenScore"], 0.7)
        self.assertAlmostEqual(g["averageScore"], 0.45)
        self.assertEqual(len(g["candidates"]), 4)
        self.assertEqual(strokes[0].points, [(100, 150), (300, 150)])

    def test_scoring_failure_still_draws(self):
        def broken(images):
            raise RuntimeError("no checkpoint")

        ctx = self.make_ctx(broken)
        with patch.object(greedy_model, "_generate_candidates", self.fake_candidates):
            strokes, message = greedy_model.generate(ctx)
        self.assertTrue(strokes)
        self.assertIn("no checkpoint", ctx.report["greedy"]["scoringError"])
        self.assertIn("unscored", message)


class TurnReportTests(unittest.TestCase):
    def test_scores_before_and_each_option_in_one_batch(self):
        batches = []

        def scorer(images):
            batches.append(len(images))
            return [0.1 * (i + 1) for i in range(len(images))]

        report = {}
        options = [(blank(), [line_action(0, 0, 100, 0)]), (blank(), [line_action(0, 0, 40, 0)])]
        ai_model.score_options(blank(), options, report, scorer=scorer)
        self.assertEqual(batches, [5])  # before + 2 drawings + 2 AI-only
        self.assertAlmostEqual(report["before"], 0.1)
        self.assertAlmostEqual(report["options"][0]["after"], 0.2)
        self.assertAlmostEqual(report["options"][1]["aiOnly"], 0.5)
        self.assertAlmostEqual(report["options"][0]["ink"], 100)
        self.assertIn("scoreDrawing", report["timings"])

    def test_missing_scorer_is_reported_but_ink_still_counted(self):
        report = {}
        ai_model.score_options(blank(), [(blank(), [line_action(0, 0, 60, 0)])], report, scorer=None)
        self.assertIn("unavailable", report["scoringError"])
        self.assertAlmostEqual(report["options"][0]["ink"], 60)
        self.assertNotIn("before", report)

    def test_scoring_failure_never_breaks_a_turn(self):
        def broken(images):
            raise RuntimeError("boom")

        report = {}
        ai_model.score_options(blank(), [(blank(), [])], report, scorer=broken)
        self.assertEqual(report["scoringError"], "creativity scoring failed")


class BatchSamplingTests(unittest.TestCase):
    def test_batch_returns_one_sketch_per_temperature(self):
        model = tiny_sketch_rnn()
        sketches = model.generate_batch(model.prime(None), [0.2, 0.5, 0.9], np.random.default_rng(0))
        self.assertEqual(len(sketches), 3)

    def test_stop_length_caps_ink_to_within_one_step(self):
        model = tiny_sketch_rnn(1)
        limit = 120.0
        for sk in model.generate_batch(model.prime(None), [0.5] * 6, np.random.default_rng(1), stop_length=limit):
            drawn = sum(path_length(s) for s in sk.strokes)
            longest_step = max((path_length(s[i:i + 2]) for s in sk.strokes for i in range(len(s) - 1)), default=0)
            self.assertLessEqual(drawn, limit + longest_step + 1e-6)

    def test_stop_at_lift_returns_at_most_one_stroke(self):
        model = tiny_sketch_rnn(2)
        cursor = model.prime([[(0, 0), (10, 0), (20, 5)]], hold_last=True)
        for sk in model.generate_batch(cursor, [0.6] * 5, np.random.default_rng(2), stop_at_lift=True):
            self.assertLessEqual(len(sk.strokes), 1)


if __name__ == "__main__":
    unittest.main()
