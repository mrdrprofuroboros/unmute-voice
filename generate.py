#!/usr/bin/env python3
"""Generate embeddings from audio files."""

from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from moshi.models import loaders
from safetensors.torch import save_file
from tqdm import tqdm

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class MimiAdapter(nn.Module):
    """David Browne's adapter from Mimi-Voice repo."""

    def __init__(self, dim=512, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(0.1),
                )
            )
        self.final_norm = nn.LayerNorm(dim)
        self.final_proj = nn.Linear(dim, dim)
        with torch.no_grad():
            self.final_proj.weight.data = torch.eye(dim) + torch.randn(dim, dim) * 0.02
            self.final_proj.bias.data.zero_()
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = x + layer(x)
        x = self.final_norm(x)
        x = self.final_proj(x)
        x = x.transpose(1, 2) + residual
        return x * self.scale


# Load Mimi encoder
print("Loading Mimi encoder...")
mimi_weight_path = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
    loaders.DEFAULT_REPO, mimi_weights=mimi_weight_path
)
checkpoint_info.lm_config = None
mimi = checkpoint_info.get_mimi(device=DEVICE)
mimi.set_num_codebooks(16)

# Load adapter
print("Loading fine-tuned weights...")
adapter = MimiAdapter().to(DEVICE)
checkpoint = torch.load("best_encoder.pt", map_location=DEVICE)
mimi.encoder.load_state_dict(checkpoint["encoder_state_dict"])
adapter.load_state_dict(checkpoint["adapter_state_dict"])

mimi.eval()
adapter.eval()


def process_audio(audio_path, mimi, adapter):
    """Load audio file, convert to embedding."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 24kHz if needed
    if sr != mimi.sample_rate:
        waveform = torchaudio.transforms.Resample(sr, mimi.sample_rate)(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Add batch dimension
    max_samples = mimi.sample_rate * 10
    wav = waveform[:, :max_samples].unsqueeze(0).to(DEVICE)

    # Generate embedding
    with torch.no_grad():
        emb = mimi.encode_to_latent(wav, quantize=False)
        emb = adapter(emb)

    return emb


# Process audio files
input_dir = Path("./input")  # Change this to your input directory
output_dir = Path("./output")  # Change this to your output directory
output_dir.mkdir(exist_ok=True)

audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3"))
print(f"Found {len(audio_files)} audio files")

for audio_file in tqdm(audio_files, desc="Processing"):
    try:
        emb = process_audio(audio_file, mimi, adapter)

        # Save embedding as safetensors (Kyutai TTS format)
        ext = ".1e68beda@240.safetensors"
        out_file = output_dir / (audio_file.name + ext)

        tensors = {"speaker_wavs": emb.cpu().contiguous()}
        metadata = {"epoch": "240", "sig": "1e68beda"}
        save_file(tensors, out_file, metadata)

        tqdm.write(f"✓ {audio_file.name} → {out_file.name}")

    except Exception as e:
        tqdm.write(f"✗ {audio_file.name}: {e}")

print("Done!")
