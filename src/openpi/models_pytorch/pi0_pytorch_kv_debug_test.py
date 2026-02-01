import os

import torch

from openpi.models_pytorch import pi0_pytorch as m


def test_summarize_past_key_values_list(monkeypatch):
    monkeypatch.setenv("OPENPI_DEBUG_KV_LAYERS", "ends")
    k0 = torch.zeros((1, 2, 5, 4), dtype=torch.float16)
    v0 = torch.zeros((1, 2, 5, 4), dtype=torch.float16)
    k1 = torch.zeros((1, 2, 5, 4), dtype=torch.float16)
    v1 = torch.zeros((1, 2, 5, 4), dtype=torch.float16)
    pkv = [(k0, v0), (k1, v1)]

    s = m._summarize_past_key_values(pkv, expected_seq_len=5)
    assert s["num_layers"] == 2
    assert s["seq_len"] == 5
    assert "0" in s["layers"]
    assert "1" in s["layers"]


def test_summarize_past_key_values_cache_like(monkeypatch):
    monkeypatch.setenv("OPENPI_DEBUG_KV_LAYERS", "all")

    class DummyCache:
        def __init__(self):
            self.key_cache = [
                torch.zeros((1, 2, 7, 4), dtype=torch.bfloat16),
                torch.zeros((1, 2, 7, 4), dtype=torch.bfloat16),
            ]
            self.value_cache = [
                torch.zeros((1, 2, 7, 4), dtype=torch.bfloat16),
                torch.zeros((1, 2, 7, 4), dtype=torch.bfloat16),
            ]

    pkv = DummyCache()
    s = m._summarize_past_key_values(pkv, expected_seq_len=7)
    assert s["num_layers"] == 2
    assert s["seq_len"] == 7
    assert set(s["layers"].keys()) == {"0", "1"}


def test_parse_debug_kv_layers_clamps(monkeypatch):
    monkeypatch.setenv("OPENPI_DEBUG_KV_LAYERS", "0,2,999,-1,foo")
    assert m._parse_debug_kv_layers(3) == [0, 2]

    # Reset to avoid leaking to other tests.
    monkeypatch.setenv("OPENPI_DEBUG_KV_LAYERS", os.environ.get("OPENPI_DEBUG_KV_LAYERS", "ends"))

