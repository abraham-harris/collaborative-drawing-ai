"""HTTP checks for the optional AuDrA scoring endpoint."""

import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

from server import app


class ScoreEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        image = Image.new("RGB", (8, 8), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        self.data_url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    def test_returns_raw_score(self):
        with patch("audra_service.score_drawing", return_value=0.42) as scorer:
            response = self.client.post("/api/score", json={"image": self.data_url})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"score": 0.42})
        scorer.assert_called_once()

    def test_rejects_invalid_and_oversized_images(self):
        self.assertEqual(self.client.post("/api/score", json={}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/score", json={"image": "data:image/png;base64,invalid"}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/api/score", json={"image": self.data_url + "a" * 4_194_304}).status_code,
            413,
        )

    def test_model_failure_is_reported_without_checkpoint_details(self):
        with (
            patch("audra_service.score_drawing", side_effect=RuntimeError("private checkpoint missing")),
            patch.object(app.logger, "exception"),
        ):
            response = self.client.post("/api/score", json={"image": self.data_url})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private checkpoint", response.json["error"])

    def test_existing_turn_endpoint_still_responds(self):
        result = Image.new("RGB", (8, 8), "white")
        with patch("server.generate_drawing", return_value=(result, "", [], "cat")):
            response = self.client.post(
                "/api/turn",
                json={"image": self.data_url, "mode": "complete", "subject": "cat"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["image"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
