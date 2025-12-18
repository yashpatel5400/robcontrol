# robcontrol

Conformal robust LQR utilities built to complement the `robbuffet` toolbox.

## Install
- Requires Python >= 3.10.
- From repo root:
  ```bash
  pip install -e .
  ```
- Installs console scripts: `robcontrol-data`, `robcontrol-model`, `robcontrol-assess`.
- Key dependency: `robbuffet` (for conformal calibration + operator-norm score).

## What it does
- Generate synthetic LQR datasets (mass-spring-damper, cartpole, DC motor) with identified dynamics.
- Train an expressive MLP dynamics predictor on design parameters `theta -> [A,B]`.
- Conformalize the predictor with operator-norm residuals and evaluate nominal vs CPC robust controllers on predicted dynamics.

## Quickstart
1) **Generate data**
   ```bash
   robcontrol-data --task cartpole --num-samples 500 --horizon 200 --seed 0 --out robcontrol/artifacts/cartpole_dataset.npz
   ```
2) **Train predictor**
   ```bash
   robcontrol-model \
     --dataset robcontrol/artifacts/cartpole_dataset.npz \
     --out-model robcontrol/artifacts/cartpole_model.pt \
     --out-meta robcontrol/artifacts/cartpole_meta.json \
     --epochs 400 --lr 1e-3 --batch-size 128
   ```
3) **Calibrate + assess nominal vs CPC**
   ```bash
   robcontrol-assess \
     --dataset robcontrol/artifacts/cartpole_dataset.npz \
     --model robcontrol/artifacts/cartpole_model.pt \
     --meta robcontrol/artifacts/cartpole_meta.json \
     --horizon 200 --rollouts 5 --trials 5 --seed 0
   ```
Outputs: calibration plot and metrics JSON (means/stds; paired t-test robust < nominal when trials > 1).

## Sample results
Cartpole (500 samples, horizon 200, rollouts 5, trials 5):
| method                | mean cost | std    |
|-----------------------|-----------|--------|
| true_opt_on_true      | 132.28    | 9.10   |
| nominal_on_true       | 3353.10   | 456.71 |
| cpc_on_true           | 3353.08   | 456.70 |
| t-test (robust < nom) | t = -4.18 | p=0.007 |

Mass-spring-damper (500 samples, horizon 200, rollouts 5, trials 5):
| method                | mean cost | std   |
|-----------------------|-----------|-------|
| true_opt_on_true      | 5.97      | 0.48  |
| nominal_on_true       | 8.98      | 0.81  |
| cpc_on_true           | 8.98      | 0.81  |
| t-test (robust < nom) | t = -14.22| p=7.1e-05 |

## Package layout
- `data.py`: dataset generation CLI.
- `model.py`: predictor training CLI (GELU/LayerNorm MLP).
- `assess.py`: calibration + evaluation CLI (operator-norm conformal, CPC vs nominal).
- `controllers/`: CPC and (optional) H-infinity controllers.
- `utils.py`: Riccati solver, rollouts, cost helper.

## Citation
If you use this in academic work, please also cite the underlying conformal robust control paper:
```
@article{patel2024conformal,
  title={Conformal robust control of linear systems},
  author={Patel, Yash and Rayan, Sahana and Tewari, Ambuj},
  journal={arXiv preprint arXiv:2405.16250},
  year={2024}
}
```
