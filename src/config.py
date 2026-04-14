import os
import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import numpy as np
import torch

warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
torch.set_default_dtype(torch.float32)

ROOT        = Path('.')
DATA_DIR    = ROOT / 'dataset'
PLOTS_DIR   = ROOT / 'src' / 'plots'
TENSOR_NPZ  = DATA_DIR / 'physionet2012_tensor.npz'
PIVOTED_CSV = DATA_DIR / 'physionet2012_pivoted.csv'


@dataclass
class Config:
    # data
    n_hours:       int   = 48
    n_vars:        int   = 36
    horizons:      List  = field(default_factory=lambda: [6, 12, 24])
    # training
    batch_size:    int   = 128
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    n_epochs:      int   = 50
    patience:      int   = 10
    clip_grad:     float = 1.0
    # GRU-D
    grud_hidden:   int   = 64
    grud_layers:   int   = 2
    grud_dropout:  float = 0.3
    # Latent ODE
    latent_dim:    int   = 32
    ode_hidden:    int   = 64
    enc_hidden:    int   = 64
    # DeepHit
    deephit_alpha: float = 0.2
    deephit_sigma: float = 0.1
    # WandB
    entity:        str   = "ayush-patel-05-dl-genai-project"
    project:       str   = "TSA-MiniProj"


CFG = Config()


def load_env():
    env_path = Path('.env')
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()
        print('Loaded .env')


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'PyTorch: {torch.__version__}')
    print(json.dumps(asdict(CFG), indent=2))
