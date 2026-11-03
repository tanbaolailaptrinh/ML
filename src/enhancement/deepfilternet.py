from __future__ import annotations

from threading import Lock

import numpy as np
import torch

from src.audio.types import AudioData

from .base import SpeechEnhancer


class DeepFilterNetEnhancer(SpeechEnhancer):
    """Pretrained DeepFilterNet speech denoiser.

    DeepFilterNet operates at the model's native sample rate (normally 48 kHz).
    The adapter preserves the input sample rate, channel count and duration.
    """

    def __init__(
        self,
        model_base_dir: str | None = None,
        device: str = "auto",
        attenuation_limit_db: float | None = None,
    ) -> None:
        try:
            from df.enhance import enhance, init_df
        except ImportError as error:
            raise RuntimeError(
                "DeepFilterNet is missing. Install it with `pip install deepfilternet`."
            ) from error

        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
        if attenuation_limit_db is not None and attenuation_limit_db < 0:
            raise ValueError("attenuation_limit_db must be non-negative or None")

        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        # DeepFilterNet selects the device used by enhance() through this
        # environment variable in df.modules.get_device(). Set it before init_df.
        import os

        os.environ["DF_DEVICE"] = selected_device
        options: dict[str, object] = {}
        if model_base_dir:
            options["model_base_dir"] = model_base_dir
        initialized = init_df(**options)
        if len(initialized) == 4:
            self.model, self.df_state, _, _ = initialized
        else:
            self.model, self.df_state, _ = initialized
        self.model.to(selected_device)
        self.model.eval()
        self.sample_rate = int(self.df_state.sr())
        self._enhance = enhance
        self.attenuation_limit_db = attenuation_limit_db
        self._lock = Lock()

    def enhance(self, audio: AudioData) -> AudioData:
        if audio.frame_count == 0:
            return audio.with_samples(audio.samples, metadata={**audio.metadata, "enhancement": "deepfilternet"})

        # DeepFilterNet uses a mono model. Preserve stereo by processing each
        # channel independently and restoring the interleaved layout.
        waveform = np.asarray(audio.samples, dtype=np.float32).reshape(-1, audio.channels)
        channel_outputs: list[np.ndarray] = []
        with self._lock, torch.inference_mode():
            for channel_index in range(audio.channels):
                channel = waveform[:, channel_index]
                tensor = torch.from_numpy(channel.copy()).reshape(1, -1)
                if audio.sample_rate != self.sample_rate:
                    tensor = torch.nn.functional.interpolate(
                        tensor.unsqueeze(0),
                        size=round(len(channel) * self.sample_rate / audio.sample_rate),
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)
                enhanced = self._enhance(
                    self.model,
                    self.df_state,
                    tensor,
                    atten_lim_db=self.attenuation_limit_db,
                )
                enhanced = enhanced.detach().float().cpu().reshape(1, -1)
                if self.sample_rate != audio.sample_rate:
                    enhanced = torch.nn.functional.interpolate(
                        enhanced.unsqueeze(0), size=audio.frame_count,
                        mode="linear", align_corners=False,
                    ).squeeze(0)
                channel_outputs.append(enhanced.squeeze(0).numpy())

        result = np.stack(channel_outputs, axis=1).reshape(-1)
        return audio.with_samples(
            result,
            metadata={
                **audio.metadata,
                "enhancement": "deepfilternet",
                "enhancement_model_sample_rate": self.sample_rate,
            },
        )
