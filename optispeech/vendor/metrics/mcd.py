import torch
import torchaudio


_LOG_SCALE = 10.0 / torch.log(torch.tensor(10.0))


def calculate_mcd(y: torch.Tensor, y_hat: torch.Tensor, sample_rate: int, n_mfcc: int = 13) -> float:
    """
    Calculates Mel Cepstral Distortion (MCD) between reference and generated audio.

    MCD = (10 / ln10) * sqrt(2 * sum_{k=1}^{K} (c_ref_k - c_gen_k)^2)

    C0 (energy) is excluded following standard MCD evaluation convention.

    Args:
        y:           Ground truth waveforms (B, T)
        y_hat:       Generated waveforms (B, T)
        sample_rate: Audio sample rate
        n_mfcc:      Number of MFCCs to use (excluding C0)

    Returns:
        Mean MCD in dB over the batch.
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc + 1,  # +1 because we drop C0
        melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80},
    ).to(y.device)

    log_scale = _LOG_SCALE.to(y.device)

    mcd_total = 0.0
    for ref, gen in zip(y, y_hat):
        c_ref = mfcc_transform(ref.unsqueeze(0))[:, 1:, :]  # (1, n_mfcc, T)
        c_gen = mfcc_transform(gen.unsqueeze(0))[:, 1:, :]

        min_len = min(c_ref.shape[-1], c_gen.shape[-1])
        diff = c_ref[:, :, :min_len] - c_gen[:, :, :min_len]

        # Per-frame MCD, averaged over frames
        frame_mcd = log_scale * torch.sqrt(2.0 * (diff ** 2).sum(dim=1))  # (1, T)
        mcd_total += frame_mcd.mean().item()

    return mcd_total / len(y)
