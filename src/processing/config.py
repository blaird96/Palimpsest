from dataclasses import dataclass

@dataclass(frozen=True)
class ChunkConfig:
    size: int = 1_200
    overlap: int = 200


@dataclass(frozen=True)
class EmbeddingConfig:
    batch_size: int = 32