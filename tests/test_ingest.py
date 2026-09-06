from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace

from src import ingest as ingest_module

class TestIngest(TestCase):
    def test_normalize_text(self):
        text = "Hello, world!\n\nThis is a test.\n\nThis is another test."

        self.assertEqual(ingest_module.normalize_text(text), "Hello, world!\n\nThis is a test.\n\nThis is another test.")

    def test_chunk_text(self):
        self.assertEqual(ingest_module.chunk_text("Hello, world!", ingest_module.CONFIG.chunk_config), ["Hello, world!"])


class TestEmbedBatch(TestCase):
    
    @patch("src.ingest.ollama.embed")
    def test_embed_batch(self, mock_embed):
        mock_embed.return_value = SimpleNamespace(
            embeddings=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        )

        texts = ["Hello, world!", "This is a test."]

        result = ingest_module.embed_batch(texts)

        self.assertEqual(result, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        mock_embed.assert_called_once_with(
            model=ingest_module.OLLAMA.EMBEDDING_MODEL,
        )