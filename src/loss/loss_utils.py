import torch


def si_snr(ests, targets):
    if targets.size() != ests.size() or targets.ndim != 3:
        raise TypeError(f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead")
    
    zero_mean = True
    EPS = 0

    # Step 1. Zero-mean norm
    if zero_mean:
        mean_source = torch.mean(targets, dim=2, keepdim=True)
        mean_est = torch.mean(ests, dim=2, keepdim=True)
        targets = targets - mean_source
        ests = ests - mean_est

    # Step 2. Pair-wise SI-SDR.
    
    # [batch, n_src]
    pair_wise_dot = torch.sum(ests * targets, dim=2, keepdim=True)
    # [batch, n_src]
    s_target_energy = torch.sum(targets**2, dim=2, keepdim=True) + EPS
    # [batch, n_src, time]
    scaled_targets = pair_wise_dot * targets / s_target_energy

    e_noise = ests - scaled_targets

    # [batch, n_src]
    pair_wise_sdr = torch.sum(scaled_targets**2, dim=2) / (torch.sum(e_noise**2, dim=2) + EPS)

    pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
    return torch.mean(pair_wise_sdr)