import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mix_with_snr(signal, noise, target_snr_db=10.0):
    sig_power = np.mean(signal ** 2) + 1e-9
    noise_power = np.mean(noise ** 2) + 1e-9
    snr_linear = 10 ** (target_snr_db / 10.0)
    scale = np.sqrt(sig_power / (snr_linear * noise_power))
    return signal + scale * noise