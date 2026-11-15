# Reassessing Machine-Learning Day-Ahead AQI Forecasting in India

**Data leakage, persistence baselines and transferability across monitoring stations**

Ashwani Sah (IIT Jodhpur) · Aleena Swetapadma (KIIT University)

This repository contains the code, processed data and result files for our study of day-ahead Air Quality Index (AQI) forecasting in India. Seven pollutants are forecast one day ahead at R K Puram, Delhi, converted to AQI with the official CPCB formula, and compared with persistence ("tomorrow = today") on the same days. The models are then transferred to seven stations 9–1,740 km away.

## Main findings

| | Finding |
|---|---|
| **N1** | Same-day AQI classifiers reach 92.6–97.7% accuracy because they approximate the CPCB formula; their errors cluster at category breakpoints (pooled p = 3.6 × 10⁻¹⁶). |
| **N2** | Day-ahead models improve PM2.5 and PM10 forecasts at the source station (skill up to 0.35), but the 5.6-point gain in AQI category accuracy is not confirmed by a paired test. |
| **N3** | Transfer follows similarity in pollution level, not distance. A level-free model brings zero-shot transfer close to persistence. |
| **N4** | The stubble-burning window has the lowest skill; satellite fire counts add no consistent improvement. |

## Repository structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── COMMITS.md                         # commit hashes, timestamps and messages
├── CHOSSING_STATION_BASED _ON_DISTANCE.ipynb   # station selection, distances and station map
├── Fig1_station_map.png
├── Utility_fun/                       # aqi_utils.py: AQI formula, scores, bootstrap, tests
├── DATASETS/                          # raw hourly CPCB pollutant files
├── Data_Other/                        # other input data
├── XY_Bengaluru_Silk.csv              # Bengaluru feature matrix
├── results/                           # early result files
└── MODEL/
    ├── FINAL-MODEL-WORKING/           # source station: R K Puram, Delhi
    │   ├── DELHI_FINAL.ipynb
    │   ├── best_params.json           # raw Optuna result
    │   ├── final_params.json          # settings used (defaults kept where tuning did not help)
    │   ├── delhi_saved/
    │   ├── era5_delhi_rkpuram.csv
    │   ├── firms_punjab_haryana.csv
    │   ├── figures/
    │   └── *.csv                      # result tables
    ├── ITO-DELHI-Same-region-validation/
    ├── Ludhiana/
    ├── Dehradun/
    ├── Jodhpur/
    ├── Bhubneshwar/                   # Patia, Bhubaneswar
    ├── Guwahati/
    ├── Benguluru/                     # Silk Board, Bengaluru
    └── summary/                       # transfer, similarity, level-free model
```

Each station folder inside `MODEL/` contains one notebook, its daily data (`daily_*.csv`, `X_*.csv`), its ERA5 weather file, its result tables and its figures.

## Environment

- Python 3.x (tested on macOS, CPU only; no GPU needed)
- Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

All data are public. The raw hourly CPCB files are included in `DATASETS/`.

| Data | Source | How to get it |
|---|---|---|
| Hourly pollutant concentrations | CPCB Air Quality Portal, https://airquality.cpcb.gov.in/ | Included in `DATASETS/`. To download again: select the station, 1 Jan 2023 – 30 Sep 2026, hourly, parameters PM2.5, PM10, NO2, SO2, NH3, CO, O3 |
| Hourly weather (ERA5) | Open-Meteo Historical Weather API, https://open-meteo.com | Downloaded automatically by each notebook; the resulting `era5_*.csv` files are included |
| Active fires (VIIRS S-NPP) | NASA FIRMS, https://firms.modaps.eosdis.nasa.gov/ | Included as `firms_punjab_haryana.csv`. To download again, set your own key: `export FIRMS_MAP_KEY=your_key` |
| India state boundaries (`Admin2.*`, map only) | DataMeet India maps (`States/Admin2.*`), https://github.com/datameet/maps | Not included; download and place next to `CHOSSING_STATION_BASED _ON_DISTANCE.ipynb`. Needed only for the station map |

**Station data used:** R K Puram and ITO (Delhi), Punjab Agricultural University (Ludhiana), Doon University (Dehradun), Mandor (Jodhpur), Patia (Bhubaneswar), Railway Colony (Guwahati), Silk Board (Bengaluru).

## How to reproduce

The notebooks were run with fixed random seeds (0–4), and their outputs are saved, so every number in the paper can be checked without rerunning.

To rerun:

1. Update the data paths at the top of each notebook to your local folder.
2. Run the notebooks in this order:
   1. `MODEL/FINAL-MODEL-WORKING/DELHI_FINAL.ipynb`: source station, model selection, tuning, leakage tests, persistence benchmark, weather sensitivity, seasons and fire (writes `final_params.json`, used by all other notebooks)
   2. The seven station notebooks in `MODEL/`, in any order: ITO, Ludhiana, Dehradun, Jodhpur, Patia, Guwahati, Bengaluru
   3. `MODEL/summary/`: cross-station transfer, similarity vs distance, level-free model
   4. `CHOSSING_STATION_BASED _ON_DISTANCE.ipynb`: station map (optional)

### Where each result in the paper comes from

| Paper | Notebook |
|---|---|
| Leakage table, boundary test | Each station notebook (leakage section) + `MODEL/summary/` (pooled test) |
| R K Puram vs persistence, weather sensitivity | `MODEL/FINAL-MODEL-WORKING/DELHI_FINAL.ipynb` |
| Own-station results (all stations) | Each station notebook (Part E) |
| Transfer, level-free, similarity figure, ITO/Patia figure | `MODEL/summary/` |
| Seasons and fire | `MODEL/FINAL-MODEL-WORKING/DELHI_FINAL.ipynb` |

## Checkpoints

No model checkpoints are included. All models are classical machine-learning regressors (Lasso, ExtraTrees, LightGBM, HistGradientBoosting) that retrain from the code in minutes on a standard laptop CPU, using fixed random seeds (0–4). Saving them for 7 pollutants × 5 seeds × 8 stations would take several hundred MB with no benefit over retraining. Instead, the repository includes the tuned hyperparameters (`final_params.json`) and the result tables, so every number in the paper can be checked against the notebook outputs without retraining.

## Reproducibility statement

We declare that all results reported in the paper were produced with the code in this repository, using the public data sources listed above, a chronological train–test split and fixed random seeds (0–4). All notebooks are provided with their saved outputs, and the result tables (`*.csv`) and hyperparameters (`final_params.json`) used in the paper are included. Small numerical differences may occur with different library versions.

## Commit history

The commit log (hashes, timestamps and messages) is in `COMMITS.md`, generated with:

```bash
git log --pretty=format:"%h  %ad  %s" --date=iso > COMMITS.md
```

## License

MIT License. See `LICENSE`.
