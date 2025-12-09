# SEIRS Virulence-Transmission Trade-off Model

## Updated Project Structure

```
Drug_symptom_modelling/
├── Models/
│   ├── SEIRS_Models.py
│   ├── params.py
│   └── __init__.py
├── Scripts/
│   ├── Drug_exploration.py        # v3: baseline+drug and sweep (single heatmap of peak λ)
│   ├── Drug_three_drugs.py        # v3: baseline + 3 drugs; saves λ(t) plot
│   └── Drug_three_panels.py       # v3: three subplots; sweeps multipliers per drug
├── Figures/
│   ├── drug_v3_time_series.png
│   ├── drug_v3_sweep_heatmap_foi.png
│   ├── drug_v3_force_of_infection.png
│   └── drug_v3_three_panels.png
├── Tables/
│   └── *.csv
├── Parameter_testing.py
├── requirements.txt
└── README.md
```

Notes:
- Scripts are intended to be run from the repository root.
- Plots are saved under Figures/.

## Scripts overview

### Scripts/Drug_exploration.py
- Purpose: 
  - run: compare baseline (m_c=1, m_r=1) vs. a drug scenario from Models/params.py and save time series.
  - sweep: sweep drug multipliers and plot a single heatmap of the peak force of infection λ(t).
- Outputs:
  - Figures/drug_v3_time_series.png
  - Figures/drug_v3_sweep_heatmap_foi.png
  - Tables/drug_v3_summary.csv
- Run:
```bash
# Baseline vs params drug
python Scripts/Drug_exploration.py run --days 200

# Grid sweep → heatmap of peak λ(t)
python Scripts/Drug_exploration.py sweep --days 200
```
- Default sweep ranges (can be overridden on CLI):
  - drug_contact_multiplier (m_c): 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
  - drug_transmission_multiplier (m_r): 0.10, 0.25, 0.50, 0.75, 1.00, 1.20

Example overrides:
```bash
python Scripts/Drug_exploration.py sweep \
  --m-c 0.6 0.9 1.2 1.5 1.8 \
  --m-r 0.2 0.4 0.6 0.8 1.0
```

### Scripts/Drug_three_drugs.py
- Purpose: Compare baseline and three drug scenarios using Model v3.
- Scenarios:
  - Drug A: contact↑, transmission↓ (m_c > 1, m_r < 1)
  - Drug B: contact↑, transmission= (m_c > 1, m_r = 1)
  - Drug C: contact=, transmission↓ (m_c = 1, m_r < 1)
- Output: Figures/drug_v3_force_of_infection.png
- Run:
```bash
python Scripts/Drug_three_drugs.py --days 200
python Scripts/Drug_three_drugs.py --mc-inc 1.3 --mr-dec 0.7 --mc-inc-same-r 1.4 --mr-dec-only 0.6
```

### Scripts/Drug_three_panels.py
- Purpose: One figure with three subplots (one per drug) sweeping multipliers.
- Panels:
  - A: contact↑ and transmission↓ (vary m_c and m_r lists)
  - B: contact↑, transmission= (vary m_c list, fix m_r=1)
  - C: contact=, transmission↓ (fix m_c=1, vary m_r list)
- Output: Figures/drug_v3_three_panels.png
- Run:
```bash
python Scripts/Drug_three_panels.py --days 200 \
  --A-mc 1.0,1.2,1.4 --A-mr 0.6,0.8 \
  --B-mc 1.0,1.2,1.4 \
  --C-mr 0.6,0.8,1.0
```

## Model and params notes

- Model v3 uses:
  - contact_rate_low, transmission_probability_low, contact_rate_high, phi_transmission
  - drug_contact_multiplier (m_c), drug_transmission_multiplier (m_r)
  - kappa_base, kappa_scale, theta for split-at-onset treatment fractions
- Ensure Models/params.py defines kappa_base and kappa_scale; scripts fallback to 1.0 if absent.

## Environment setup (Linux)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Quick check:
```bash
python - <<'PY'
import numpy, scipy, pandas, matplotlib, seaborn
print("OK")
PY
```

List saved figures:
```bash
ls -1 Figures | sed -n '1,50p'
```