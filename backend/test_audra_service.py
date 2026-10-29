"""Checkpoint handling checks that do not require PyTorch."""

import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audra_service


class CheckpointTests(unittest.TestCase):
    def test_private_blob_is_checked_and_reused(self):
        content = b"fake checkpoint for transport test"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(audra_service, "LOCAL_CHECKPOINT", Path(directory) / "missing.ckpt"),
                patch.object(audra_service, "CHECKPOINT_SHA256", digest),
                patch.object(audra_service.tempfile, "gettempdir", return_value=directory),
                patch.dict(os.environ, {
                    "AUDRA_CHECKPOINT_BLOB_URL": "https://example.private.blob.vercel-storage.com/model.ckpt",
                    "BLOB_READ_WRITE_TOKEN": "test-token",
                }),
                patch.object(audra_service, "urlopen", return_value=io.BytesIO(content)) as download,
            ):
                path = audra_service._checkpoint_path()
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(audra_service._checkpoint_path(), path)
                self.assertEqual(download.call_count, 1)
                self.assertEqual(download.call_args.args[0].get_header("Authorization"), "Bearer test-token")


if __name__ == "__main__":
    unittest.main()
