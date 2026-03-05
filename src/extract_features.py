import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

SR = 32000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
TOP_DB = 80

MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=20,
    f_max=8000,
)

DB_TRANSFORM = T.AmplitudeToDB(top_db=TOP_DB)

def extract_features(input_dir, output_dir):
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)

    for wav in tqdm(wav_files, desc="Extracting features"):
        rel = os.path.relpath(wav, input_dir)
        out = (Path(output_dir) / rel).with_suffix(".pt")
        out.parent.mkdir(parents=True, exist_ok=True)

        waveform, sr = torchaudio.load(wav)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != SR:
            waveform = T.Resample(sr, SR)(waveform)

        spec = DB_TRANSFORM(MEL_TRANSFORM(waveform))
        torch.save(spec, out)

    print(f"Saved {len(wav_files)} files")

def main():
    INPUT_DIR = "data/audio"
    OUTPUT_DIR = "data/features"
    extract_features(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()