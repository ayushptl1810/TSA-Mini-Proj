# PhysioNet 2012 — ICU Mortality Survival Prediction

Survival analysis on the PhysioNet Computing in Cardiology Challenge 2012 dataset.
Rather than a static binary classifier ("did this patient die?"), the models answer
a clinically meaningful question at every ICU hour:

> **"Will this patient die within the next X hours from now?"**

---

## Models

### GRU-D (Gated Recurrent Unit with Decay)
Processes each patient's irregular multivariate time series hour-by-hour.
When a variable is not observed at time *t*, instead of forward-filling, the
imputed value decays exponentially toward the population mean at a rate
learned per-variable. This models the clinical reality that unmonitored
variables drift back toward baseline — and that long gaps are themselves
a risk signal.

Key components:
- **Variable-importance attention** — down-weights sparse variables (e.g. Cholesterol at 0.2% obs rate) and up-weights informative ones (HR at 90%)
- **Horizon-conditioned shared head** — a single MLP conditioned on a learned horizon embedding replaces separate per-horizon classifiers, enforcing the monotone 6h ⊂ 12h ⊂ 24h risk ordering

### Latent ODE + DeepHit Survival Head
Encodes each patient's trajectory into a continuous latent state using a Neural ODE,
then attaches a DeepHit-style discrete-time survival head.

Pipeline:
1. **RecognitionRNN** — processes observations backwards, only updating hidden state at timesteps with actual measurements → outputs `(z₀_mean, z₀_logvar)`
2. **Latent ODE** — integrates `dz/dt = f(t, z)` forward from t=0 to t=47 using `dopri5` (adaptive Runge-Kutta). The dynamics function `f` is time-conditional: it takes a sinusoidal time embedding alongside `z`, capturing circadian rhythms and clinical workflow patterns
3. **DeepHit survival head** — maps `z(t)` at each hour to a discrete hazard `h(t) = P(T=t | T≥t)`, giving a full survival curve `S(t) = ∏ₛ﹤ₜ (1 − h(s))`

---

## Windowed Survival Labels

Labels are built with a **soft ramp** to prevent models from trivially learning
"is the clock past hour 42?" instead of genuine physiological deterioration:

```
label[i, t] = 0.0   for t < T_death − 2X        (safe zone)
            = ramp   for T_death − 2X ≤ t < T_death − X   (ramp 0→1)
            = 1.0   for t ≥ T_death − X          (danger zone)
```

Horizons evaluated: **6h, 12h, 24h**

---

## TSA Concepts Applied

| Analysis | Finding |
|---|---|
| **ACF / PACF** | HR, MAP show significant autocorrelation at lags 1–6h; GCS is more step-like |
| **ADF stationarity** | HR is non-stationary in dying patients (p=0.85) but stationary in survivors (p=0.001) — temporal drift is the signal |
| **Missingness pattern** | Dead patients have shorter observation gaps (intensive monitoring = sicker patient). Missingness is MAR/MNAR, not random — GRU-D decay handles this explicitly |
| **Observation density** | 19.4% overall; HR 90%, Urine 69%, Cholesterol 0.2% — extreme heterogeneity across variables |

---

## Project Structure

```
.
├── data/
│   ├── physionet2012_timeseries.csv      raw long-format (5.2M rows)
│   ├── physionet2012_pivoted.csv         wide hourly format (538k rows)
│   ├── physionet2012_tensor.npz          model-ready tensors (12k × 48 × 36)
│   └── physionet2012_summary.xlsx        EDA summary
├── src/
│   ├── config.py                         Config dataclass, device, paths
│   ├── data/
│   │   ├── dataset.py                    ICUDataset, delta computation, loaders
│   │   └── analysis.py                   TSA plots (ACF, ADF, missingness)
│   ├── models/
│   │   ├── grud.py                       GRUDCell, GRUD, GRUDSurvivalModel
│   │   ├── latent_ode.py                 ODEFunc, RecognitionRNN, LatentODESurvival
│   │   └── attention.py                  VariableAttention, HorizonConditionedHead
│   ├── losses.py                         focal loss, windowed survival loss, DeepHit loss, KL
│   ├── metrics.py                        AUROC, AUPRC, C-index, IBS, fixed-time AUROC
│   ├── train.py                          EarlyStopping, train_grud_epoch, train_ode_epoch
│   ├── baselines.py                      LogisticRegression on 48h mean features
│   └── evaluate.py                       metric tables, calibration, IBS plots
├── main.py                               training entry point
├── baseline.ipynb                        full walkthrough notebook
└── .env                                  WANDB_API_KEY (not committed)
```

---

## Setup

```bash
python3 -m venv venv2
source venv2/bin/activate
pip install torch torchdiffeq torchcde wandb scikit-learn pandas numpy \
            matplotlib seaborn statsmodels jupyter
```

Add your W&B key to `.env`:
```
WANDB_API_KEY=your_key_here
```

Run training:
```bash
python main.py
```

Or open the full walkthrough:
```bash
jupyter notebook baseline.ipynb
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **AUROC** | Discrimination at each prediction horizon |
| **AUPRC** | Precision-recall; primary metric given 14.2% class imbalance |
| **Fixed-time AUROC** | AUROC evaluated only at a specific ICU hour (e.g. t=12), not aggregated — clinically realistic |
| **C-index** | Harrell's concordance; measures survival rank ordering |
| **IBS** | Integrated Brier Score; calibration + discrimination over the full 48h window |
| **Calibration curve** | Reliability diagram; reveals over/underconfidence that AUROC hides |

---

## Dataset

See [data/DATASET.md](data/DATASET.md) for full column-level documentation of `physionet2012_pivoted.csv`.

---

## References

- Che et al. (2018) — [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://arxiv.org/abs/1606.01865)
- Chen et al. (2018) — [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- Lee et al. (2018) — [DeepHit: A Deep Learning Approach to Survival Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/11842)
- PhysioNet Challenge 2012 — [Predicting Mortality of ICU Patients](https://physionet.org/content/challenge-2012/1.0.0/)
