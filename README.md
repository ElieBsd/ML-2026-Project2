# CS-433 Project 2 - Copy Detection Pattern authentication (ML4Science, SIP lab UNIGE)

EPFL CS-433 Machine Learning, Fall 2026 - Project 2 (ML4Science).

- **Team:** Elie Bussod, Gabin Prunet, Antoine Mandly
- **Host lab:** Stochastic Information Processing (SIP) group, Université de Genève - Prof. Slava Voloshynovskiy, mentor Vitaliy Kinakh

## Topic
Copy detection patterns (CDP) are small printed random binary codes used against counterfeiting: printing and scanning blur them in a way that a copy cannot reproduce exactly. Joakim Tutt's PhD thesis (UNIGE, 2025) models this with the **binary pattern-based channel (BPC)**: the probability that a printed symbol is read wrong depends only on its 3x3 neighbourhood in the template, and symbols are independent.

**Question: do the guarantees of the BPC model hold on real, independent data?** We work on the public Scantrust CDP dataset and test three things:
- **H1 - independence.** Are flips over-dispersed or spatially correlated compared to what the BPC predicts?
- **H2 - a score without fakes.** A locally most powerful (LMP) test derived from the BPC, with a per-image quality correction, compared to Hamming distance and the thesis scores.
- **H3 - realistic thresholds.** Thresholds are set on originals only (conformal calibration). We report the false acceptance rate at a fixed 1 % / 5 % miss rate with 95 % confidence intervals, on unseen attacks, against supervised baselines that do see fakes (oracle references).

All choices (thresholds, hyperparameters) are made on calibration or validation splits grouped by template, never on the test split.

## Setup
Python 3.11.
```bash
uv venv --python 3.11 .venv      # or: python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"  # or: pip install -e ".[dev]"
pytest
```

## Data
Both datasets are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (non-commercial, no derivatives). They may not be redistributed and are **not included in this repository**. Everything goes under `dataset/`, which is listed in `.gitignore` together with `*.zip`.

One script downloads everything. No account, no token, standard library only:

```bash
python scripts/download_data.py --list      # what each group contains
python scripts/download_data.py scantrust   # one or several groups
python scripts/download_data.py all         # everything, about 44 GB
```

Downloads resume where they stopped, and files already present are skipped.

| Group | Source | Content | Size |
|---|---|---|---|
| `scantrust` | Kaggle | Scantrust CDP dataset: 5000 templates printed once with four kinds of fakes (`unique_cdp`), 50 templates printed 50 times with fakes (`batch_cdp`) | 0.5 GB |
| `mobile` | Yareta | 2023 Indigo 1x1 mobile: 1440 templates, originals and fakes, iPhone 12 Pro and Samsung Galaxy Note 20 Ultra, 6 captures each (the dataset of the thesis) | 9.4 GB |
| `scanner` | Yareta | 2021 Indigo 1x1 base at 2400 dpi: 720 templates, originals from two printers (HPI55, HPI76), four kinds of fakes, estimated templates | 4.6 GB |
| `variability` | Yareta | 2022 Indigo 1x1 variability, InvercoteG paper: the same templates printed again (10 print runs on HPI55, 3 on HPI76) and the other print sessions of both printers | 29.3 GB |

### Scantrust CDP dataset
https://www.kaggle.com/datasets/scantrust/copy-detection-pattern-dataset

The script downloads the public zip and unpacks it into `dataset/unique_cdp/` and `dataset/batch_cdp/` (28550 PNG files).

Citation: E. Khermaza, I. Tkachenko, J. Picard, "Can Copy Detection Patterns be copied? Evaluating the performance of attacks and highlighting the role of the detector", IEEE WIFS 2021.

By hand, if the script fails: sign in to Kaggle, click **Download** > **Download dataset as zip**, move `archive.zip` to the root of this repository and unzip it there.

### SIP Indigo datasets
https://yareta.unige.ch/archives/2eab9a72-dd60-4a5f-ba95-0144cc2983aa (doi:[10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma](https://doi.org/10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma), R. Chaban and S. Voloshynovskiy, University of Geneva, 2023).

The SIP group publishes all its CDP data as one public 123 GB archive, and we only need part of it. The script reads the index of the remote zip with HTTP range requests and fetches the selected files into `dataset/sip_yareta/`. Each file is checked against its CRC. The files stay zipped and are read directly from the zips.

By hand, if the range requests ever stop working: download the archive from the page above and unzip the same folders into `dataset/sip_yareta/`.

## Repository layout
```
configs/       one YAML config per experiment
scripts/       data download and data manifest
dataset/       Scantrust and SIP data (not versioned)
src/cdpbpc/    library: binarization, 3x3 patterns, codebook, scores, metrics, splits
experiments/   one script per experiment, writes results/ and figures
tests/         pytest
notebooks/     exploration only, no reported number comes from here
report/        LaTeX report and generated figures
references/    project description and thesis
```

## References
- `references/project2_description.pdf` - official CS-433 Project 2 description (Fall 2026).
- `references/Tutt_2025_PhD_thesis_CDP.pdf` - J. Tutt, *Un modèle mathématique d'impression et d'imagerie pour l'authentification des codes de détection de copie*, PhD thesis, Université de Genève, 2025. doi:[10.13097/archive-ouverte/unige:192628](https://doi.org/10.13097/archive-ouverte/unige:192628). Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); redistributed unmodified.
- SIP group code: https://github.com/sip-group/snf-it-dis-code (read as a specification only; no code is copied).
