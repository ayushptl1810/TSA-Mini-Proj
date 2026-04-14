# PhysioNet 2012 — Dynamic 6-Hour Mortality Risk

Dynamic mortality risk scoring on the PhysioNet Computing in Cardiology Challenge 2012 dataset.
Rather than a static binary classifier ("did this patient die?"), the models answer
a clinically actionable question at every ICU hour:

> **"What is the probability this patient dies within the next 6 hours?"**

This is a **dynamic risk score**, not a deterioration detector. The label is anchored to the
in-hospital death event — the models learn to assign rising risk in the hours immediately
preceding death. A patient who deteriorates and recovers contributes only to the negative class.

---

## Results

| Model | Global AUROC | Global AUPRC | 6h AUROC | 6h AUPRC | C-index | IBS |
|---|---|---|---|---|---|---|
| LR (first-24h mean) | 0.764 | 0.380 | — | — | — | — |
| GRU-D | 0.721 | 0.312 | 0.848 | 0.206 | — | — |
| **Latent ODE + DeepHit** | **0.837** | **0.429** | **0.937** | **0.270** | **0.837** | **0.007** |

**6h AUROC** = windowed AUROC over all observed ICU hours for the 6h horizon.
**Global AUROC** = discrimination of all-cause in-hospital mortality (p_global).
LR is a fair baseline using only the first 24h to avoid post-deterioration leakage.

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
- **Exponential decay** — per-variable `γ_x(δ) = exp(−max(0, W_γ · δ))` imputes missing values; `γ_h` decays hidden state between observations
- **Variable-importance attention** — sigmoid gate conditioned on per-patient observation rate; down-weights sparse variables (Cholesterol 0.2%) and up-weights dense ones (HR 90%)
- **6h risk head** — MLP outputting `P(death within next 6h)` at every ICU hour

### Latent ODE + DeepHit Survival Head

Encodes each patient's trajectory into a continuous latent state using a Neural ODE,
then reads off a full survival curve.

Pipeline:
1. **RecognitionRNN** — processes observations backwards; only updates hidden state at timesteps with actual measurements → outputs `(z₀_mean, z₀_logvar)`
2. **Latent ODE** — integrates `dz/dt = f(t, z)` forward from t=0 to t=47 using `dopri5` (adaptive Runge-Kutta). `f` takes a sinusoidal time embedding alongside `z`, capturing circadian rhythms and clinical workflow patterns
3. **DeepHit survival head** — maps `z(t)` at each hour to discrete hazard `h(t) = P(T=t | T≥t)`, giving full survival curve `S(t) = ∏ₛ﹤ₜ (1 − h(s))`
4. **6h risk head** — shared horizon-conditioned MLP outputting `P(death within 6h)` at each hour alongside the survival curve

---

## 6-Hour Risk Labels

Labels use a **soft ramp** anchored to each patient's last observed hour `T_death`:

```
label[i, t] = 0.0   for t < T_death − 12        (safe zone)
            = ramp   for T_death − 12 ≤ t < T_death − 6   (linear ramp 0→1)
            = 1.0   for t ≥ T_death − 6          (danger zone: within 6h of death)
```

Alive patients → all zeros for all t.

The ramp prevents the model from learning "is the clock past hour 41?" instead of
learning from physiological signal. Without it, there is a hard cliff at `T_death − 6`
that trivially leaks temporal position.

---

## TSA Concepts Applied

| Analysis | Finding |
|---|---|
| **ACF / PACF** | HR and MAP show significant autocorrelation at lags 1–6h; GCS is more step-like |
| **ADF stationarity** | HR is non-stationary in dying patients (p=0.85) but stationary in survivors (p=0.001) — temporal drift is the mortality signal |
| **Missingness pattern** | Dead patients have *shorter* observation gaps (13.2h vs 14.3h) — intensive monitoring = sicker patient. Missingness is MAR/MNAR, not random |
| **Observation density** | 19.4% overall; HR 90.1%, Urine 69.2%, Cholesterol/TroponinI 0.2% — extreme heterogeneity motivates variable-importance attention |

---

## Research Context

The PhysioNet 2012 in-hospital mortality task is a standard benchmark. Published AUROC
on the standard static task (single prediction after full 48h):

| Model | AUROC |
|---|---|
| LR (hand features) | ~0.812 |
| GRU-D (Che et al., 2018) | ~0.853 |
| SeFT (Horn et al., 2020) | ~0.858 |
| mTAND (Shukla & Marlin, 2021) | ~0.856 |
| Raindrop (Zhang et al., 2022) | ~0.874 |

All of the above solve a **retrospective** task — they see the full 48h stay and predict
a single binary outcome. This project's formulation is **prospective**: at every ICU
hour, output a probability that the patient dies in the next 6 hours. There is no
published benchmark for this exact windowed formulation on PhysioNet 2012.

---

## Project Structure

```
.
├── dataset/
│   ├── physionet2012_timeseries.csv      raw long-format (5.2M rows)
│   ├── physionet2012_pivoted.csv         wide hourly format (538k rows)
│   ├── physionet2012_tensor.npz          model-ready tensors (12k × 48 × 36)
│   └── physionet2012_summary.xlsx        EDA summary
├── src/
│   ├── config.py                         Config dataclass, device, paths
│   ├── data/
│   │   ├── dataset.py                    ICUDataset, delta/last_obs_hour, loaders
│   │   └── analysis.py                   TSA plots (ACF, ADF, missingness)
│   ├── models/
│   │   ├── grud.py                       GRUDCell, GRUD, GRUDSurvivalModel
│   │   ├── latent_ode.py                 ODEFunc, RecognitionRNN, LatentODESurvival
│   │   └── attention.py                  VariableAttention, HorizonConditionedHead
│   ├── losses.py                         focal loss, windowed survival loss, DeepHit loss (event-time), KL
│   ├── metrics.py                        AUROC, AUPRC, C-index, time-varying IBS, fixed-time AUROC
│   ├── train.py                          EarlyStopping, train_grud_epoch, train_ode_epoch
│   ├── baselines.py                      LogisticRegression on first-24h mean features
│   └── evaluate.py                       metric tables, calibration curves, IBS/survival plots
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

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **6h AUROC** | Discrimination of windowed 6h risk score across all observed ICU hours |
| **6h AUPRC** | Precision-recall for 6h risk; primary metric given ~2% positive label rate at any given hour |
| **Global AUROC** | Discrimination of all-cause in-hospital mortality using `p_global` |
| **Fixed-time AUROC** | 6h risk AUROC evaluated only at a specific ICU hour (t=6, 12, 24, 36) — tests real-time usefulness, not aggregate performance |
| **C-index** | Harrell's concordance on survival ordering |
| **IBS** | Integrated Brier Score using time-varying `I(T≤t)` labels — calibration + discrimination over the full 48h window |
| **Calibration curve** | Reliability diagram; reveals over/underconfidence that AUROC hides |

---

## Dataset

See [data/DATASET.md](data/DATASET.md) for full column-level documentation of `physionet2012_pivoted.csv`.

---

## References

- Che et al. (2018) — [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://arxiv.org/abs/1606.01865)
- Rubanova et al. (2019) — [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907)
- Chen et al. (2018) — [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- Lee et al. (2018) — [DeepHit: A Deep Learning Approach to Survival Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/11842)
- Horn et al. (2020) — [Set Functions for Time Series](https://arxiv.org/abs/1909.12064)
- Zhang et al. (2022) — [Raindrop: Graph-Guided Network for Irregularly Sampled Multivariate Time Series](https://arxiv.org/abs/2110.05357)
- PhysioNet Challenge 2012 — [Predicting Mortality of ICU Patients](https://physionet.org/content/challenge-2012/1.0.0/)
