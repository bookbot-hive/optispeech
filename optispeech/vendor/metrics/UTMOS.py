"""UTMOS MOS predictor backed by the fairseq-free utmos-pytorch package."""

from __future__ import annotations

import torch
from utmos_pytorch import UTMOSScoreTorch


class UTMOSScore:
    """Predict subjective speech quality (MOS) for 16 kHz audio."""

    def __init__(self, device: torch.device | str, verbose: bool = False) -> None:
        """Load the UTMOS strong learner on the requested device.

        Args:
            device: Torch device or device string (for example ``cuda:0``).
            verbose: If True, log missing/unexpected keys while loading weights.
        """
        device_str = str(device)
        self._model = UTMOSScoreTorch(device=device_str, verbose=verbose)

    def score(self, wavs: torch.Tensor) -> torch.Tensor:
        """Score one or more 16 kHz waveforms.

        Args:
            wavs: Audio tensor with shape ``(T,)``, ``(B, T)``, or ``(B, 1, T)``.

        Returns:
            Per-utterance MOS scores with shape ``(B,)``.
        """
        if wavs.device.type != "cpu":
            wavs = wavs.to(self._model.device)
        return self._model.score(wavs)
