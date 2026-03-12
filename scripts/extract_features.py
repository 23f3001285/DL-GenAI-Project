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

    for wav in tqdm(wav_files, desc="Extracting features", total=len(wav_files)):
        rel = os.path.relpath(wav, input_dir)
        out = (Path(output_dir) / rel).with_suffix(".pt")
        out.parent.mkdir(parents=True, exist_ok=True)

        waveform, sr = torchaudio.load(wav)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != SR:
            waveform = T.Resample(sr, SR)(waveform)

        spec = DB_TRANSFORM(MEL_TRANSFORM(waveform))   # (1, 128, T)
        torch.save(spec, out)

extract_features(SYNTH_PATH, FEATURE_PATH)
print("Total .pt:", len(glob.glob(f"{FEATURE_PATH}/**/*.pt", recursive=True)))
