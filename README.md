# 2025guessr

This repository contains code to forecast the 2025 Romanian presidential runoff results on a per-precinct basis.

## Pre-election pipeline

`pre_election_model.py` trains a neural network using precinct-level data from the 2019 and 2024 elections. It uses demographic breakdowns, turnout, first round vote shares, and spatial neighbour information to predict second round vote counts for each finalist. The script saves the trained model and preprocessing objects for use on election night.

### Usage

1. Place CSV files with the following names under a `data/` directory:
   - `2019_precincts.csv` – first and second round results for 2019 with demographics.
   - `2024_precincts.csv` – first and second round results for 2024 with demographics.
   - `2025_first_round.csv` – first round results for 2025 with demographics.
   - `presidentialcandidates.txt` – mapping of candidate names to their party (used for feature engineering).

   Each file should include columns referenced in `pre_election_model.py` (e.g. `registered_voters`, vote counts for each candidate, `county`, `uat_name`, `urban_rural`, and age/sex counts such as `M_18`, `F_25`, etc.).

2. Run the script:

```bash
python pre_election_model.py
```

The script trains the model and outputs `2025_runoff_predictions.csv` containing predicted vote counts for the two runoff candidates in every precinct.


### Live election-night updates

`live_update_model.py` refines the baseline predictions as results arrive from individual precincts. Provide a CSV with the reported vote counts so far (columns: `precinct_id`, `candA_votes_reported`, `candB_votes_reported`). Run:

```bash
python live_update_model.py partial_results.csv
```

The script outputs updated precinct predictions in `runoff_updated_predictions.csv` and prints the projected national totals.
