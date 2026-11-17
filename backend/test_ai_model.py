"""Tests for the two-choice, limited-ink turn logic."""

import random
import unittest
from unittest.mock import patch

from PIL import Image

from ai_model import INK_LIMIT, generate_drawing, trim_strokes
from drawing_models import MODELS
from drawing_models.strokes import Stroke


class AITurnTests(unittest.TestCase):
    def test_trim_strokes_keeps_only_first_budget_of_path(self):
        strokes = [
            Stroke(points=[(0, 0), (200, 0)]),
            Stroke(points=[(0, 10), (200, 10)]),
            Stroke(points=[(0, 20), (100, 20)]),
        ]
        limited = trim_strokes(strokes)
        self.assertEqual(len(limited), 2)
        self.assertEqual(limited[1].points[-1], (100, 10))
        self.assertEqual(sum(stroke.length for stroke in limited), INK_LIMIT)

    def test_candidates_use_distinct_subjects_and_selected_mode(self):
        seen = []

        def draw(ctx):
            seen.append(ctx.subject)
            return [Stroke(points=[(0, 0), (400, 0)])], "generated"

        image = Image.new("RGB", (800, 600), "white")
        with patch.dict(MODELS, {"continue": draw}):
            candidates = generate_drawing(image, [], mode="continue", rng=random.Random(1))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(set(seen)), 2)
        self.assertEqual([candidate[3] for candidate in candidates], seen)
        self.assertTrue(all(candidate[2][0]["points"][-1]["x"] == 300 for candidate in candidates))

    def test_empty_subject_is_retried(self):
        attempts = []

        def draw(ctx):
            attempts.append(ctx.subject)
            if len(attempts) == 1:
                return [], ""
            return [Stroke(points=[(0, 0), (20, 0)])], "generated"

        image = Image.new("RGB", (800, 600), "white")
        with patch.dict(MODELS, {"complete": draw}):
            candidates = generate_drawing(image, [], rng=random.Random(2))
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(candidates), 2)
        self.assertNotIn(attempts[0], [candidate[3] for candidate in candidates])

    def test_accepted_strokes_are_used_as_model_context(self):
        counts = []

        def draw(ctx):
            counts.append(len(ctx.strokes))
            return [Stroke(points=[(0, 0), (20, 0)])], "generated"

        image = Image.new("RGB", (800, 600), "white")
        prior = {"tool": "brush", "points": [{"x": 0, "y": 0}, {"x": 20, "y": 0}]}
        current = {"tool": "brush", "points": [{"x": 0, "y": 10}, {"x": 20, "y": 10}]}
        with patch.dict(MODELS, {"complete": draw}):
            generate_drawing(image, [current], context_actions=[prior, current],
                             rng=random.Random(3))
        self.assertEqual(counts, [2, 2])

    def test_failed_subjects_are_not_exposed_in_error(self):
        image = Image.new("RGB", (800, 600), "white")
        with patch.dict(MODELS, {"complete": lambda ctx: ([], "")}):
            with self.assertRaisesRegex(RuntimeError, "couldn't make two choices") as raised:
                generate_drawing(image, [], rng=random.Random(4))
        self.assertEqual(str(raised.exception), "The AI couldn't make two choices this turn")


if __name__ == "__main__":
    unittest.main()
