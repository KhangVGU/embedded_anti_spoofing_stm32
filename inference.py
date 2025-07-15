import argparse
import json
import torch
import soundfile as sf
import numpy as np
from scipy.signal import resample
import torch.nn.functional as F
from utils import str_to_bool
from AASIST import Model


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path)
    if sr != target_sr:
        num = int(len(wav) * float(target_sr) / sr)
        wav = resample(wav, num)
    return wav


def main(args):
    # 1) Load training config for consistency
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    d_args = cfg['model_config']

    # 2) Verify that filts[0] is an integer
    if not isinstance(d_args['filts'][0], int):
        raise ValueError(f"Expected d_args['filts'][0] to be int, got {type(d_args['filts'][0])}")

    # 3) Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 4) Instantiate model
    model = Model(d_args).to(device)

    # 5) Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    # Load with strict=False to handle any minor key mismatches
    load_res = model.load_state_dict(state_dict, strict=False)
    if load_res.unexpected_keys:
        print("Warning: unexpected keys in checkpoint:", load_res.unexpected_keys)
    if load_res.missing_keys:
        print("Warning: missing keys in model:", load_res.missing_keys)

    model.eval()
    torch.set_grad_enabled(False)

    # 6) Load and prepare audio
    wav = load_audio(args.audio, target_sr=16000)
    x = torch.from_numpy(wav).float().unsqueeze(0).to(device)  # (1, N)

    # 7) Run inference
    _, logits = model(x, Freq_aug=str_to_bool(cfg.get('freq_aug', 'False')))
    print("Logits:", logits.detach().cpu().numpy())
    
    probs = torch.softmax(logits, dim=1)
    top_prob, top_idx = probs.max(dim=1)

    # Map class index to label
    class_map = {0: 'Spoof', 1: 'Bona fide'}
    label = class_map.get(top_idx.item(), 'Unknown')

    print(f"Predicted class: {top_idx.item()} ({label})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AASIST inference')
    parser.add_argument('--config',     type=str, required=True,
                        help='Path to the training config JSON')
    parser.add_argument('--checkpoint', type=str, default='weights/best.pth',
                        help='Model checkpoint path')
    parser.add_argument('--audio',      type=str, default='data/sample.wav',
                        help='Input audio file (16kHz)')
    args = parser.parse_args()
    main(args)
