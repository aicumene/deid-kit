# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: Zstandard frames made by the zstandard package, for the Rust decoder to read back.

    python tools/vectors_zstd.py OUT_DIR   → OUT_DIR/zstd.json   (needs `zstandard`)
"""
import base64
import json
import random
import sys
from pathlib import Path

import zstandard as zstd


def main(out):
    rnd = random.Random(14)
    words = "the rent lease tenant landlord PERSON_12345678 ORG_87654321 Hamburg Musterweg Ada Brenner clause notice".split()
    inputs = [b"", b"a", b"abc" * 1000, bytes(rnd.getrandbits(8) for _ in range(5000)), bytes(range(256)) * 40,
              " ".join(rnd.choice(words) for _ in range(40000)).encode(),
              json.dumps({"model": "gpt", "input": [{"type": "message", "content": [{"type": "input_text", "text": " ".join(rnd.choice(words) for _ in range(3000))}]}] * 30}).encode(),
              ("Ünïcödé — „quotes“ ✓ " * 3000).encode(), b"\x00" * 300000]
    for _ in range(40):
        n = rnd.randint(0, 70000)
        alphabet = rnd.choice([b"ab", b"abcdefgh", bytes(range(256)), b"0123456789 ,.\n"])
        inputs.append(bytes(rnd.choice(alphabet) for _ in range(n)))
    cases = []
    for data in inputs:
        for level in (1, 3, 9, 19, -5):
            for kw in ({}, {"write_checksum": True}, {"write_content_size": False}):
                c = zstd.ZstdCompressor(level=level, **kw).compress(data)
                cases.append({"level": level, "kw": list(kw), "plain": base64.b64encode(data).decode(), "frame": base64.b64encode(c).decode()})
    # a stream compressor writes several blocks and frames without a content size
    for data in inputs[5:8]:
        cobj = zstd.ZstdCompressor(level=3).compressobj()
        c = b"".join(cobj.compress(data[i:i + 7000]) for i in range(0, len(data), 7000)) + cobj.flush()
        cases.append({"level": 3, "kw": ["stream"], "plain": base64.b64encode(data).decode(), "frame": base64.b64encode(c).decode()})
    Path(out, "zstd.json").write_text(json.dumps(cases))
    print(len(cases), "frames")


if __name__ == "__main__":
    main(sys.argv[1])
