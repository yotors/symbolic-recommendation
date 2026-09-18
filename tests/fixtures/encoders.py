"""Small deterministic encoder doubles shared by projection tests."""


class FakeEncoder:
    """Record encoder calls and return caller-provided vectors."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        return self.vectors
