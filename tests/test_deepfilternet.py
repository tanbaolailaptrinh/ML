from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.audio.types import AudioData
from src.enhancement.factory import create_enhancer


@pytest.fixture
def fake_deepfilternet(monkeypatch):
    """Install a fake df.enhance API so tests never download model weights."""
    calls: list[tuple[int, int, float | None]] = []

    class FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class FakeDFState:
        def sr(self):
            return 48_000

    def init_df(**kwargs):
        return FakeModel(), FakeDFState(), None

    def enhance(model, state, audio, atten_lim_db=None):
        calls.append((audio.shape[0], audio.shape[1], atten_lim_db))
        return audio * 0.5

    monkeypatch.setitem(sys.modules, "df", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "df.enhance", SimpleNamespace(init_df=init_df, enhance=enhance)
    )
    return calls


def test_enhances_mono_and_restores_original_rate(fake_deepfilternet):
    enhancer = create_enhancer({"name": "deepfilternet", "params": {"device": "cpu"}})
    audio = AudioData.from_iterable(np.linspace(-0.5, 0.5, 1600), sample_rate=16_000)

    result = enhancer.enhance(audio)

    assert result.sample_rate == 16_000
    assert result.channels == 1
    assert result.frame_count == audio.frame_count
    assert np.allclose(result.samples, np.asarray(audio.samples) * 0.5, atol=1e-5)
    assert fake_deepfilternet == [(1, 4800, None)]
    assert result.metadata["enhancement"] == "deepfilternet"


def test_enhances_stereo_channels_independently(fake_deepfilternet):
    enhancer = create_enhancer({"name": "deepfilternet", "params": {"device": "cpu"}})
    left = np.full(480, 0.2, dtype=np.float32)
    right = np.full(480, -0.4, dtype=np.float32)
    audio = AudioData.from_iterable(
        np.stack((left, right), axis=1).reshape(-1), sample_rate=48_000, channels=2
    )

    result = enhancer.enhance(audio)
    output = np.asarray(result.samples).reshape(-1, 2)

    assert result.channels == 2
    assert result.sample_rate == 48_000
    assert result.frame_count == audio.frame_count
    assert np.allclose(output[:, 0], left * 0.5)
    assert np.allclose(output[:, 1], right * 0.5)
    assert fake_deepfilternet == [(1, 480, None), (1, 480, None)]


def test_empty_audio_skips_model_call(fake_deepfilternet):
    enhancer = create_enhancer({"name": "deepfilternet", "params": {"device": "cpu"}})
    audio = AudioData.from_iterable([], sample_rate=16_000)

    result = enhancer.enhance(audio)

    assert result.frame_count == 0
    assert result.sample_rate == audio.sample_rate
    assert result.metadata["enhancement"] == "deepfilternet"
    assert fake_deepfilternet == []


def test_invalid_device_is_rejected(fake_deepfilternet):
    with pytest.raises(ValueError, match="device must be"):
        create_enhancer({"name": "deepfilternet", "params": {"device": "tpu"}})


def test_negative_attenuation_limit_is_rejected(fake_deepfilternet):
    with pytest.raises(ValueError, match="attenuation_limit_db"):
        create_enhancer({
            "name": "deepfilternet",
            "params": {"device": "cpu", "attenuation_limit_db": -1},
        })
