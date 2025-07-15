import argparse
import onnxruntime as ort
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample
import torch.nn.functional as F

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path)
    if sr != target_sr:
        num = int(len(wav) * float(target_sr) / sr)
        wav = resample(wav, num)
    # If stereo, collapse to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav

def main():
    parser = argparse.ArgumentParser(
        description="Run AASIST ONNX model inference on a single audio file"
    )
    parser.add_argument(
        '--onnx_model_path',
        type=str,
        required=True,
        help="Path to the exported ONNX model (e.g. ./aasist.onnx)"
    )
    parser.add_argument(
        '--audio_path',
        type=str,
        required=True,
        help="Path to the audio file for inference (e.g. LA_E_1000147.flac)"
    )
    args = parser.parse_args()

    # 1. Load and preprocess audio
    wav = load_audio(args.audio_path, target_sr=16000)
    input_data = wav.astype(np.float32)[None, :]  # shape = (1, num_samples)

    # 2. Create ONNX Runtime session
    session = ort.InferenceSession(
        args.onnx_model_path,
        providers=["CPUExecutionProvider"]
    )

    # 3. Run inference
    output_names = ["embedding", "logits"]
    embedding, logits = session.run(output_names, {"input": input_data})

    # 4. Post-process
    # logits shape: (1, 2)  →  [spoof_score, bona-fide_score]
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
    pred_label = "Bona-fide" if probs[1] > probs[0] else "Spoof"

    # 5. Print results
    # print(f"Embedding shape: {embedding.shape}")
    print(f"Output:          {logits}")
    # print(f"Probabilities:   spoof={probs[0]:.4f}, bona-fide={probs[1]:.4f}")
    print(f"Predicted class:      {pred_label}")

if __name__ == "__main__":
    main()
