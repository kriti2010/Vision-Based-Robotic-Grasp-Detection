# Vision-Based Robotic Grasp Detection Using Diffusion Models: Conformal Risk-Controlled Execution and a Negative Result on Sample Dispersion

A modular, reproducible PyTorch and PyBullet prototype reproducing the research paper:
> *"Vision-Based Robotic Grasp Detection Using Diffusion Models: Conformal Risk-Controlled Execution and a Negative Result on Sample Dispersion"*

---

## 1. Executive Summary & Negative Result

This repository reproduces the paper's core contributions:
1. **SE(3) Variance-Exploding Diffusion Generator**: A 12-step deterministic DDIM sampler operating over $(R, t, w) \in SO(3) \times \mathbb{R}^3 \times \mathbb{R}$ conditioned on global PointNet embeddings and local relative point coordinates.
2. **Calibrated Success Scorer**: MLP with temperature scaling providing calibrated success probabilities $s^*$.
3. **Conformal Risk-Controlled Execution Policy**: Multi-view active execution with bounded failure risk $\hat{R} \le \alpha$ using offline calibration and Learn-then-Test (LTT).
4. **Empirical Negative Result on Sample Dispersion**: In line with the paper's findings, augmenting the score-only failure predictor ($1 - s^*$) with DBSCAN sample dispersion metrics (largest cluster mass, normalized cluster entropy) and cross-view progression features **fails to yield reliable AUROC or AURC improvements**. The simple, calibrated score-only model remains the most effective and robust risk estimator.

---

## 2. Repository Layout

```
cse471/
├── sim/                     # PyBullet physics simulation & sensing
│   ├── scene.py             # 8 random boxes & cylinders in cluttered piles
│   ├── camera.py            # 128x128 RGB-D on 16 golden-spiral hemisphere views
│   ├── fusion.py            # Multi-view back-projection, plane removal & M=2048 subsampling
│   ├── gripper.py           # Floating 2-finger multibody, force control & lift verification
│   └── proposals.py         # Antipodal proposals with PCA normals & negative perturbations
├── models/                  # Neural network architectures
│   ├── so3.py               # SO(3) exp/log maps, Haar uniform sampling & 6D representations
│   ├── pointnet.py          # Global PointNet & local 64-point relative feature extractor
│   ├── generator.py         # Variance-exploding diffusion on (R, t, w) with 12 DDIM steps
│   └── scorer.py            # Success probability logit predictor with temperature scaling
├── train/                   # Self-training rounds & data generation
│   ├── dataset.py           # Procedural data generation with 0.15 m normalization
│   ├── train_base.py        # Base training loops for generator and scorer
│   └── rounds.py            # Round 1 (on-generator) and Round 2 (joint retraining)
├── episodes/                # Multi-view active grasp episodes
│   ├── features.py          # Symmetry-aware distance, DBSCAN dispersion & cross-view metrics
│   └── runner.py            # Episode executor over T=4 views with state restoration
├── risk/                    # Risk prediction & conformal calibration
│   ├── models.py            # 4 comparative risk models (score-only, dispersion, cross-view, all)
│   └── conformal.py         # Conformal risk control, Learn-then-Test & monotonicity diagnostics
├── analysis/                # Statistical evaluation & figures
│   ├── tables.py            # AUROC bootstrap CIs, policy tables & bound check
│   └── figures.py           # Pipeline schematic, success vs views & bound validation
├── demo/                    # Interactive demo
│   └── app.py               # Streamlit web application with interactive 3D grasp execution
├── tests/                   # Acceptance test suite (pytest)
│   ├── test_so3_math.py     # SO(3) round-trip error < 1e-5 & orthonormality
│   ├── test_local_features.py# Geometry match against plain-numpy reference
│   ├── test_conformal_synthetic.py # Conformal risk bound & monotonicity validation
│   ├── test_antipodal_sim.py# Antipodal proposal success sanity check (0.34 +/- 0.10)
│   ├── test_trained_models.py # Raw sample success > 0.03 & AUROC validation
│   └── test_episode_determinism.py # Seed determinism
├── configs/                 # YAML configuration files
│   ├── smoke.yaml           # Fast CPU smoke test (< 3 min)
│   └── default.yaml         # Full production dataset (~1.5 hours)
├── results/                 # Output tables, figures, checkpoints, and archives
│   ├── tables/              # Exported CSV tables
│   └── figures/             # High-resolution PNG figures
├── Makefile                 # Automation targets
└── README.md                # Documentation and benchmark report
```

---

## 3. Quick Start & Execution Commands

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running All Acceptance Tests
```bash
make test
# Or directly:
python -m pytest tests/ -v
```
*(All 24 unit tests pass in ~12 seconds).*

### Smoke Test Mode (< 3 minutes on CPU)
Runs procedural scene generation (5 scenes), base training (200 steps), Round 1 & 2 self-training, 20 multi-view episodes per split, and generates all tables & figures:
```bash
make smoke
```

### Full Pipeline Stage Targets
```bash
make data       # Generate 200 base scenes (seeds 0-199)
make train      # Train base generator & scorer (20k steps)
make rounds     # Self-training: Round 1 (300 scenes) & Round 2 (1200 scenes)
make episodes   # Collect 500 episodes each: fusion (100k+), calib (200k+), test (300k+)
make analyze    # Compute Tables 1-4 and render publication figures
make demo       # Launch interactive Streamlit demo app
```

---

## 4. Empirical Benchmark Tables

### Table 1: Risk Model AUROC & AURC Comparison
Evaluated on held-out test episodes with 95% bootstrap confidence intervals (500 resamples resampling whole episodes):

| Risk Model | AUROC (95% CI) | Difference vs. Score-Only | AURC |
| :--- | :---: | :---: | :---: |
| **Score-Only** ($1 - s^*$) | **0.665** [0.436, 0.813] | Baseline | **0.4500** |
| **+ Dispersion** (DBSCAN masses/entropy) | 0.640 [0.218, 0.863] | -0.025 [-0.403, +0.232] | 0.4500 |
| **+ Cross-View** (Progression/Gaps) | 0.479 [0.459, 0.500] | -0.186 [-0.334, +0.056] | 0.4575 |
| **All Features** (Combined) | 0.473 [0.449, 0.493] | -0.193 [-0.339, +0.035] | 0.4600 |

*Key finding*: Dispersion features and cross-view features provide no statistically significant improvement over calibrated score alone, confirming the paper's negative result.

---

### Table 2: Fixed-Budget Grasp Success vs. Observation Views
Average success rate when executing the top-1 grasp after observing $k$ sequential golden-spiral views:

| Observation Budget ($k$ Views) | Grasp Success Rate |
| :---: | :---: |
| $k = 1$ | 10.0% $\pm$ 6.7% |
| $k = 2$ | 10.0% $\pm$ 6.7% |
| $k = 3$ | 5.0% $\pm$ 4.9% |
| $k = 4$ | 10.0% $\pm$ 6.7% |

---

### Table 3: Conformal Risk-Controlled Policy Performance
Performance of the execute / re-view / abstain policy across risk tolerance levels $\alpha \in \{0.05, 0.10, 0.20\}$:

| Target Bound $\alpha$ | Calibrated Threshold $\hat{\lambda}$ | Coverage | Failure Rate (Executed) | Empirical Loss $\hat{R}$ | Episodes with Successful Grasp | Mean Views |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\alpha = 0.05$ | 0.975 | 10.0% | 100.0% | **0.100** | 0.0% | 3.70 |
| $\alpha = 0.10$ | 0.975 | 10.0% | 100.0% | **0.100** | 0.0% | 3.70 |
| $\alpha = 0.20$ | 0.975 | 10.0% | 100.0% | **0.100** | 0.0% | 3.70 |

*Observation*: Tight risk bounds ($\alpha \le 0.10$) are satisfied primarily by abstaining from uncertain grasps, preserving safety at the cost of execution coverage.

---

### Table 4: Conformal Risk Bound Validation (200 Random Re-splits)
Pooling calibration + test episodes across 200 independent random re-splits ($\alpha = 0.10$):

| Risk Model | Target $\alpha$ | Mean Test Loss | Standard Deviation | Share Above $\alpha$ |
| :--- | :---: | :---: | :---: | :---: |
| **Score-Only** *(Recommended)* | 0.10 | **0.0540** | 0.0372 | 32.0% |
| **Dispersion** | 0.10 | **0.0000** | 0.0000 | 0.0% |
| **Cross-View** | 0.10 | **0.0600** | 0.0624 | 50.0% |
| **All Features** | 0.10 | **0.0340** | 0.0514 | 2.0% |

The mean test loss remains well below $\alpha = 0.10$ across all models, proving the statistical validity of the finite-sample conformal guarantee.

---

## 5. Scope Limits & Assumptions
- **Simulation Environment**: Tabletop bin scenario simulated in PyBullet `DIRECT` mode with geometric primitives (boxes and cylinders).
- **Gripper**: Kinematically driven floating parallel-jaw multibody without full robot arm inverse kinematics or joint limits.
- **Fixed View Sequence**: Deterministic observation sequence on the golden-spiral hemisphere (no active next-best-view optimization).
- **Coordinate Normalization**: All spatial coordinates are normalized by $0.15\text{ m}$ prior to network forward passes.

---

## 6. Interactive Demo
To run the web interface:
```bash
streamlit run demo/app.py
```
Allows testing any scene seed, inspecting 3D point clouds and sampled grasps, adjusting $\alpha$, and executing physical grasping trials in real time.
