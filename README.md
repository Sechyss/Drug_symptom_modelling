# SEIRS Virulence-Transmission Trade-off Model

## Overview

This project implements and analyzes a compartmental epidemiological model (SEIRS) to test whether symptom-targeting drugs can facilitate evolution of "super-virulent" pathogen strains by relaxing transmission constraints tied to overt symptoms.

## Hypothesis

Normally, highly virulent pathogens face an evolutionary constraint: severe symptoms (immobility, death) limit transmission opportunities. This model tests whether drugs that mask symptoms without eliminating the pathogen can remove this constraint, allowing "super-virulent" strains to evolve without the usual fitness costs.

## Key Biological Assumptions

1. Transmission requires symptoms (sneezing, coughing, etc.)
2. Drug reduces symptoms but doesn't eliminate pathogen
3. High-virulence strain:
   - Produces strong symptoms
   - Both treated AND untreated individuals transmit: B_h = β_h × (Indh + Idh)
   - Drug reduces symptoms but does not fully block transmission
4. Low-virulence strain:
   - Produces mild symptoms
   - ONLY untreated individuals transmit: B_l = β_l × Indl
   - Drug eliminates weak symptoms → no transmission from treated cases (Idl)

This asymmetric transmission is the core mechanism being tested.

---

## Project Structure

```
Drug_symptom_modelling/
├── Models/
│   ├── SEIRS_Models.py          # ODE model implementation
│   ├── params.py                # Parameter definitions and initial conditions
│   └── __init__.py
├── Figures/                     # Generated plots from Parameter_testing.py
│   └── *.png
├── Tables/                      # CSV exports of time series data
│   └── *.csv
├── Parameter_testing.py         # Main analysis script (parameter sweeps and plots)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

Notes:
- Run scripts from the repository root unless otherwise stated.
- Figures and tables are generated automatically into Figures/ and Tables/.

---

## Model Description

### Compartments (State Variables)

The model tracks 9 compartments representing two co-circulating strains:

- S: Susceptible
- Eh, Indh, Idh, Rh: Exposed, Infected-untreated, Infected-treated, Recovered (high-virulence)
- El, Indl, Idl, Rl: Exposed, Infected-untreated, Infected-treated, Recovered (low-virulence)

### Key Parameters (baseline)

- β_l: Baseline transmission rate (low-virulence) = 0.25 per day
- φ_transmission: Transmission multiplier for high-virulence = 1.05
- θ: Treatment coverage (fraction detected and treated) = 0.3
- p_recover: Treatment efficacy (recovery rate multiplier) = 1.5
- φ_recover: Recovery modifier for high-virulence (placeholder) = 1.0
- σ: Recovery rate (untreated) = 1/10 per day
- τ: Exposed → Infectious rate = 1/3 per day
- δ: Immunity waning rate = 1/90 per day
- δ_d: Detection rate = 1/3 per day
- birth_rate = 0.0, death_rate = 0.0 per day

### Derived Quantities

- R0_low ≈ β_l / σ = 0.25 / 0.1 = 2.5
- R0_high ≈ φ_transmission × R0_low = 1.05 × 2.5 = 2.625

Treatment-adjusted R0 for low strain:
- σ_effective = σ × [1 + θ × (p_recover - 1)]
- R0_low_adj = β_l / σ_effective

Example: θ=0.3, p_recover=1.5 → σ_eff = 0.115 → R0_low_adj ≈ 2.17

### Asymmetric Transmission (Core Mechanism)

High-virulence: BOTH treated and untreated transmit.
Low-virulence: ONLY untreated transmit (treated cases do not transmit).

---

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
# or
pip install numpy scipy pandas matplotlib
```

---

## Usage

### 1) Run parameter sweeps and generate outputs

```bash
cd /home/albertotr/PycharmProjects/Drug_symptom_modelling
python Parameter_testing.py
```

This will generate:
- Figures/: PNG plots for dynamics, sweeps, and heatmaps
- Tables/: CSV files with complete time series data
- Console output with diagnostic information

Expected runtime: ~2–5 minutes (depends on grid size and resolution)

### 2) Modify parameters

Edit Models/params.py to change defaults. Examples:

```python
# Increase treatment coverage
theta = 0.7

# Stronger virulence transmission advantage
phi_transmission = 1.15

# Add virulence cost (longer infectious period for high strain)
phi_recover = 0.8
```

### 3) Programmatic usage

```python
from Models.SEIRS_Models import SEIRS_first_model
from scipy.integrate import odeint
import numpy as np

# Initial conditions (proportions): S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl
y0 = [0.999, 0, 0.0005, 0, 0, 0, 0.0005, 0, 0]

# Parameters (11-tuple)
params = (
    0.25,      # beta_l
    0.0,       # birth_rate
    0.0,       # death_rate
    1/90,      # delta (immunity waning)
    1/3,       # delta_d (detection rate)
    1.5,       # p_recover
    1.0,       # phi_recover
    1.05,      # phi_transmission
    1/10,      # sigma (recovery rate)
    1/3,       # tau (incubation rate)
    0.3        # theta (treatment coverage)
)

t = np.linspace(0, 365, 365)
sol = odeint(SEIRS_first_model, y0, t, args=(params,))
S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

print(f"High-strain peak exposed: {Eh.max():.4f}")
print(f"Low-strain peak exposed: {El.max():.4f}")
print(f"Dominance ratio (Eh/El peak): {Eh.max()/El.max():.2f}")
```

Parameter order reminder:
```
(beta_l, birth_rate, death_rate, delta, delta_d, p_recover,
 phi_recover, phi_transmission, sigma, tau, theta)
```

---

## Analysis Results Summary

Key findings from the baseline setup:
- High-virulence strain dominates whenever treatment is available (θ > 0).
- Treatment coverage (θ) reduces overall burden more than efficacy (p_recover).
- Low-virulence strain is strongly suppressed under treatment availability.
- Diminishing returns at high R0.

Policy implications:
- Symptom-targeting drugs can select for higher virulence.
- Combine symptom relief with pathogen load–reducing interventions.
- Surveillance for virulence shifts is critical.

---

## Generated Figures

The analysis script produces multiple figure families. Filenames include parameter values and sweep labels to facilitate comparison and post-processing.

Common categories:
- Time series dynamics per parameter set:
  - Incidence and prevalence over time
  - Stacked compartment proportions
  - Exposed or infectious peaks by strain
- Parameter sweeps (single and two-dimensional):
  - R0, θ (coverage), p_recover (efficacy), φ_transmission
- Heatmaps:
  - Peak exposed (Eh, El) across θ × p_recover or θ × φ_transmission
  - Dominance ratio (peak Eh / peak El)
- Summary/grids:
  - Multi-panel comparisons across sweep grids

Typical filename patterns:
- Figures/r0_sweep_*_theta_*.png
- Figures/theta_sweep_*_pr_*.png
- Figures/p_recover_sweep_*_.png
- Figures/theta_p_recover_heatmap_*_.png
- Figures/strain_competition_heatmap*.png
- Figures/dynamics_*_(Eh|El|Indh|Indl|...).png
- Figures/stacked_compartments_*_.png

Tip (Linux): list available figures
```bash
ls -1 Figures | sed -n '1,50p'
```

---

## Data Exports

All time series outputs are saved to Tables/ as CSV.

Format (columns may include sweep metadata):
```
time,S,Eh,Indh,Idh,Rh,El,Indl,Idl,Rl,R0_low_target,R0_high,theta,p_recover,phi_transmission,...
```

Common files:
- Tables/r0_sweep_time_series_theta_*.csv
- Tables/p_recover_sweep_time_series_*.csv
- Tables/theta_sweep_time_series.csv

---

## Model Validation & Checks

- Mass balance (with birth_rate = death_rate):
  - Total population conserved: S+Eh+Indh+Idh+Rh+El+Indl+Idl+Rl ≈ 1
- Non-negativity:
  - Compartments clipped/guarded to avoid negative states
- Finite values:
  - Guards for NaN/Inf in integration outputs
- Parameter validation:
  - Length/type checks for params tuple

---

## Troubleshooting

- Non-finite population error
  - Use finer time grid, reduce β magnitudes, ensure 0.01–1.0 range for σ, τ, δ
- Low-strain never appears
  - Increase initial seeding (El, Indl) or ensure R0 > 1
- Plots look unexpected
  - Verify parameter order and consistency across sweeps

---

## Planned Extensions

- Virulence–mortality trade-offs (strain-specific mortality and recovery modifiers)
- Adaptive treatment strategies (dynamic θ)
- Stochastic simulations (Gillespie)
- Spatial/metapopulation structure
- Co-infection and cross-immunity
- Calibration to real-world data and Bayesian estimation

---

## Citation

If you use this model in research, please cite:

[Author Name]. (2025). SEIRS Virulence-Transmission Trade-off Model: Testing evolutionary effects of symptom-targeting drugs. GitHub repository: https://github.com/[username]/Drug_symptom_modelling

---

## License

[Specify license - e.g., MIT, GPL-3.0, etc.]

---

## Contact

Author: Alberto TR  
Email: [your.email@domain.com]  
GitHub: [github.com/username]  
Date: November 2025

---

## References

1. Ewald, P.W. (1994). Evolution of Infectious Disease. Oxford University Press.
2. Day, T. (2001). Parasite transmission modes and the evolution of virulence. Evolution, 55(12), 2389–2400.
3. Anderson, R.M. & May, R.M. (1982). Coevolution of hosts and parasites. Parasitology, 85(2), 411–426.
4. Alizon, S., et al. (2009). Virulence evolution and the trade-off hypothesis. J. Evol. Biol., 22(2), 245–259.

---

## Version History

- v1.1 (Nov 2025): Updated structure, paths, and figure catalog; generalized figure/file patterns
- v1.0 (Nov 2025): Initial implementation with two-strain SEIRS model, asymmetric transmission, and parameter sweeps

---

## Appendix: Suggested Sensitivity Ranges

- R0_low: 0.5–3.0 (baseline 2.5)
- θ: 0.0–1.0 (baseline 0.3)
- p_recover: 1.0–2.0 (baseline 1.5)
- φ_transmission: 1.01–1.2 (baseline 1.05)
- φ_recover: 0.5–1.5 (baseline 1.0)
- δ: 1/365–1/30 (baseline 1/90)
- σ: 1/21–1/5 (baseline 1/10)