"""Tests for drawing recognition and how it picks the AI's subjects (no network needed)."""

import random
import unittest
from collections import Counter

import numpy as np

import ai_model
from drawing_models import recognizer, strokes_from_actions
from drawing_models.subjects import SUBJECTS

GUESSES = [("cat", 0.6), ("tiger", 0.2), ("lion", 0.1), ("dog", 0.05), ("owl", 0.03), ("bee", 0.01)]


class ChooseSubjectsTests(unittest.TestCase):
    def test_picks_distinct_plausible_subjects(self):
        picked = recognizer.choose_subjects(GUESSES, 3, random.Random(0))
        self.assertEqual(len(set(picked)), 3)
        self.assertTrue(set(picked) <= {"cat", "tiger", "lion", "dog", "owl"})  # "bee" is under 2%

    def test_most_obvious_guess_is_down_weighted_but_possible(self):
        firsts = Counter(recognizer.choose_subjects(GUESSES, 1, random.Random(i))[0] for i in range(4000))
        # Without the penalty "cat" would be picked ~61% of the time; with it, ~35%.
        self.assertLess(firsts["cat"] / 4000, 0.45)
        self.assertGreater(firsts["cat"], 0)
        self.assertGreater(firsts["tiger"], firsts["dog"])

    def test_returns_fewer_when_few_guesses_are_plausible(self):
        confident = [("cat", 0.97), ("dog", 0.01), ("owl", 0.01)]
        self.assertEqual(recognizer.choose_subjects(confident, 3, random.Random(0)), ["cat"])


class RasterizeTests(unittest.TestCase):
    def test_drawing_is_centered_and_fits_like_quick_draw(self):
        image = recognizer.rasterize([[(100, 100), (500, 100)], [(100, 300), (500, 300)]])
        self.assertEqual(image.shape, (28, 28))
        cols = np.nonzero(image.max(axis=0) > 0.1)[0]
        self.assertGreaterEqual(cols.min(), 0)
        self.assertLessEqual(cols.max(), 27)
        self.assertAlmostEqual((cols.min() + cols.max()) / 2, 13.5, delta=1.0)

    def test_blank_drawing(self):
        self.assertEqual(recognizer.rasterize([]).max(), 0)


@unittest.skipUnless(recognizer.available(), "recognizer.npz not trained yet")
class PickSubjectsTests(unittest.TestCase):
    def setUp(self):
        self.strokes = strokes_from_actions([{"tool": "brush", "points": [
            {"x": 300 + 100 * np.cos(t), "y": 300 + 100 * np.sin(t)} for t in np.linspace(0, 6.3, 40)]}])

    def test_recognized_subjects_come_first_then_random_fill(self):
        report = {"timings": {}}
        picked = ai_model.pick_subjects(self.strokes, 6, random.Random(0), True, report)
        guesses = [g["subject"] for g in report["recognition"]["guesses"]]
        self.assertEqual(len(set(picked)), 6)
        self.assertIn(picked[0], guesses)
        self.assertIn("recognize", report["timings"])

    def test_random_when_recognition_is_off_or_canvas_blank(self):
        for strokes, recognize in ((self.strokes, False), ([], True)):
            report = {"timings": {}}
            picked = ai_model.pick_subjects(strokes, 3, random.Random(0), recognize, report)
            self.assertEqual(len(picked), 3)
            self.assertTrue(set(picked) <= set(SUBJECTS))
            self.assertNotIn("recognition", report)


if __name__ == "__main__":
    unittest.main()
