"""
FineWeb-Edu data pipeline for pretraining.

Streams data from HuggingFace, tokenises on-the-fly, and packs into
fixed-length sequences for efficient GPU training.

Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import os
import torch
from torch.utils.data import IterableDataset, DataLoader

# ---------------------------------------------------------------------------
#  Tokeniser wrapper (GPT-2 BPE via tiktoken — fast, no auth needed)
# ---------------------------------------------------------------------------
_TOKENISER = None


def get_tokeniser():
    global _TOKENISER
    if _TOKENISER is None:
        import tiktoken
        _TOKENISER = tiktoken.get_encoding("gpt2")
    return _TOKENISER


def tokenise(text: str) -> list[int]:
    enc = get_tokeniser()
    return enc.encode_ordinary(text)


def decode(ids: list[int]) -> str:
    return get_tokeniser().decode(ids)


VOCAB_SIZE = 50257          # GPT-2 BPE vocabulary size
EOT_TOKEN = 50256          # <|endoftext|>


# ---------------------------------------------------------------------------
#  Streaming dataset that packs documents into fixed-length chunks
# ---------------------------------------------------------------------------
class FineWebEduDataset(IterableDataset):
    """
    Streams FineWeb-Edu from HuggingFace, tokenises each document, and
    concatenates + chunks into sequences of ``seq_len`` tokens.

    This is the standard "packing" approach used in GPT / LLaMA pretraining.
    An EOT token is inserted between documents.

    Args:
        seq_len:    context window size (tokens)
        split:      HF dataset split (default "train")
        subset:     FineWeb-Edu subset, e.g. "sample-10BT", "sample-100BT"
        seed:       shuffle seed
        max_docs:   optional cap on number of documents (for debugging)
    """

    def __init__(
        self,
        seq_len: int = 2048,
        split: str = "train",
        subset: str = "sample-10BT",
        seed: int = 42,
        max_docs: int | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.split = split
        self.subset = subset
        self.seed = seed
        self.max_docs = max_docs

    def _stream(self):
        """Yield (input_ids, labels) chunks from the HF stream."""
        from datasets import load_dataset

        # Datasets >=3.0 dropped trust_remote_code; clear the env flag if present.
        os.environ.pop("HF_DATASETS_TRUST_REMOTE_CODE", None)

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        buffer: list[int] = []
        doc_count = 0

        for example in ds:
            if self.max_docs and doc_count >= self.max_docs:
                break

            text = example.get("text", "")
            if not text.strip():
                continue

            tokens = tokenise(text)
            buffer.extend(tokens)
            buffer.append(EOT_TOKEN)
            doc_count += 1

            # Yield full chunks from buffer
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y

    def __iter__(self):
        return self._stream()


# ---------------------------------------------------------------------------
#  DataLoader factory
# ---------------------------------------------------------------------------
def create_dataloader(
    seq_len: int = 2048,
    batch_size: int = 4,
    subset: str = "sample-10BT",
    max_docs: int | None = None,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a streaming DataLoader for FineWeb-Edu pretraining."""
    dataset = FineWebEduDataset(
        seq_len=seq_len,
        subset=subset,
        seed=seed,
        max_docs=max_docs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
