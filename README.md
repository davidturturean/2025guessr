# 2025guessr

This repository contains utilities for experimenting with a real-time forecast of the 2025 Romanian presidential runoff. It uses precinct level results and demographics to predict how votes will be distributed between George Simion and Nicușor Dan as results arrive.

## Requirements

- Python 3.11+
- `pandas`
- `torch`

## Training

A training script is provided to build the model using past election data. Usage example:

```bash
python train.py \
  --demo data/demographics_11102019.csv \
  --first data/pv_part_cntry_prsd_11102019.csv \
  --second data/pv_part_cntry_prsd_11242019.csv \
  --simion-col IOHANNIS_KLAUS_WERNER \
  --dan-col DANCILA_VIORICA
```

To train on multiple past elections at once, simply pass several paths to each
option:

```bash
python train.py \
  --demo data/demographics_11102019.csv data/demographics_11242024.csv \
  --first data/pv_part_cntry_prsd_11102019.csv data/pv_part_cntry_prsd_11242024.csv \
  --second data/pv_part_cntry_prsd_11242019.csv data/pv_part_cntry_prsd_11242024.csv \
  --simion-col IOHANNIS_KLAUS_WERNER \
  --dan-col DANCILA_VIORICA
```

This command trains the model and saves the weights to `model.pt`.

## Forecasting

The `forecast.py` script can combine multiple first round result files and optionally demographics before running the forecast. Pass `--features` for each results CSV and `--demo` for the demographics table:

```bash
python forecast.py \
  --features data/pv_part_cntry_prsd_05062025.csv \
  --features data/pv_part_cntry_prsd_c_05062025.csv \
  --demo data/demographics_05042025.csv \
  --model model.pt
```

Real-time updates can be performed by using the `ElectionForecaster` class from `runoff_model.py` and calling `update_precinct` as results come in.
