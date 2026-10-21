# FedPAW with Informer for Vehicle Speed Prediction

An academic project exploring the computation–accuracy tradeoff in personalized vehicle speed prediction. This work extends the existing FedPAW framework with a custom Transformer predictor named `Informer` in the archived code. It uses standard multi-head attention, not the full Informer architecture or ProbSparse attention described in the report.

## Problem

Driver behavior and vehicle characteristics vary across clients. Personalized federated learning trains client-specific models without centrally collecting raw client data. The prediction backbone also needs to balance forecasting accuracy and computation cost.

## Project contributions

- Reproduced and evaluated the existing FedPAW framework on CarlaVSP.
- Integrated a custom Transformer predictor as an alternative to the LSTM-based predictor.
- Compared computation time, mean absolute error (MAE), and root mean square error (RMSE).

## Reported results

| Model | Time per iteration (s) | MAE (m/s) | RMSE (m/s) |
| --- | ---: | ---: | ---: |
| FedPAW + SSLSTM | 7.07 | 1.65 | 2.02 |
| FedPAW + BiSSLSTM | 7.86 | 1.62 | 1.98 |
| FedPAW + Informer | 4.53 | 1.75 | 2.12 |

Relative to SSLSTM, Informer reduced the reported time per iteration by **35.9%**, with an increase in MAE from 1.65 to 1.75 m/s. These are archived project results, not newly rerun benchmarks. Iteration time should not be interpreted as isolated inference latency.

The report describes 10 simulated clients, 300 federated rounds, and 5- and 10-second prediction horizons. The archived notebook currently contains different settings; see the reproduction notes below.

## Repository contents

```text
main.ipynb          Colab experiment notebook
models.py           Modified model definitions
data_utils.py       Modified dataset utilities
requirements.txt   Dependencies identified from the archived notebook
results/           Archived plots and experiment logs
docs/              Project report, presentation, and original run instructions
```

## Running the archived notebook

The notebook was written for **Google Colab** and uses `/content/PFLlibVSP` paths.

1. Open `main.ipynb` in Google Colab and select a GPU runtime.
2. Run its clone cell for the original repository:
   `https://github.com/heyuepeng/PFLlibVSP.git`.
3. Follow the upstream dataset setup instructions to obtain and prepare CarlaVSP. Dataset files are not included here.
4. Replace `/content/PFLlibVSP/system/utils/data_utils.py` with this repository's `data_utils.py`.
5. Replace `/content/PFLlibVSP/system/flcore/trainmodel/models.py` with this repository's `models.py`.
6. Review the model, dataset, feature dimensions, prediction horizon, and round count before running the experiment cells.

### Reproduction status

This repository preserves an academic experiment and is not yet a one-click reproduction package.

- The saved notebook defaults to `global_rounds=1`, but the archived Informer experiment log records 300 rounds.
- Its default dataset is `driving10_fg34`. The constructor is called with `out_len=5`, but the model ignores that argument and hard-codes `self.out_len=10`. Confirm the intended horizon before modifying either setting.
- The archived `InformerLayer` uses standard `nn.MultiheadAttention`; it does not implement ProbSparse attention. The report's sparse-attention complexity claim does not describe the supplied implementation.
- The algorithm is named `FedAWA` in the saved code. Confirm its mapping to the reported FedPAW experiment in the upstream implementation.
- Local execution requires adapting hard-coded Colab paths.
- Dependency versions are only partially recorded. The supplied requirements are a starting point, not a tested environment lockfile.

## Attribution

This project builds on [PFLlibVSP / the original FedPAW implementation](https://github.com/heyuepeng/PFLlibVSP) and the paper *FedPAW: Federated Learning with Personalized Aggregation Weights for Urban Vehicle Speed Prediction* by Yuepeng He and coauthors ([paper](https://arxiv.org/abs/2412.01281)). FedPAW is prior work; this project's contribution is the predictor modification and experimental comparison.

The supplied Python files retain PFLlib copyright notices for Jianqing Zhang and GNU GPL version 2-or-later notices. Preserve those notices and include the applicable upstream license text when distributing the derived code. No new MIT or other replacement license is asserted here.

## Author

Raju Santhani — M.Tech Smart Mobility, IIT Hyderabad.
