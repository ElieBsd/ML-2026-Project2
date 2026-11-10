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

### Scantrust CDP dataset (Kaggle, about 490 MB, by hand)
Kaggle needs an account, so this one is downloaded by hand:
1. Sign in to Kaggle (free account) and open https://www.kaggle.com/datasets/scantrust/copy-detection-pattern-dataset
2. Click **Download** > **Download dataset as zip**.
3. Move the downloaded `archive.zip` to the root of this repository and unzip it there. It creates `dataset/unique_cdp/` and `dataset/batch_cdp/`. You can then delete `archive.zip`.

Citation: E. Khermaza, I. Tkachenko, J. Picard, "Can Copy Detection Patterns be copied? Evaluating the performance of attacks and highlighting the role of the detector", IEEE WIFS 2021.

### SIP Indigo datasets (University of Geneva, Yareta, by script)
The SIP group publishes all its CDP data as one public 123 GB archive:
https://yareta.unige.ch/archives/2eab9a72-dd60-4a5f-ba95-0144cc2983aa (doi:[10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma](https://doi.org/10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma), R. Chaban and S. Voloshynovskiy, 2023).

We only need part of it. `scripts/download_sip.py` reads the index of the remote zip with HTTP range requests and fetches the selected files into `dataset/sip_yareta/`. No account, standard library only. Each file is checked against its CRC and an interrupted download resumes where it stopped.

```bash
python scripts/download_sip.py --list      # sizes of each group
python scripts/download_sip.py mobile      # one or several groups
python scripts/download_sip.py all         # everything below, about 43 GB
```

| Group | Content | Size |
|---|---|---|
| `mobile` | 2023 Indigo 1x1 mobile: 1440 templates, originals and fakes, iPhone 12 Pro and Samsung Galaxy Note 20 Ultra, 6 captures each (the dataset of the thesis) | 9.4 GB |
| `scanner` | 2021 Indigo 1x1 base at 2400 dpi: 720 templates, originals from two printers (HPI55, HPI76), four kinds of fakes, estimated templates | 4.6 GB |
| `variability` | 2022 Indigo 1x1 variability, InvercoteG paper: the same templates printed again (10 print runs on HPI55, 3 on HPI76) and the other print sessions of both printers | 29.3 GB |

The files stay zipped and are read directly from the zips. If the range requests ever stop working, download the archive from the page above and unzip the same folders into `dataset/sip_yareta/`.

## Repository layout
```
configs/       one YAML config per experiment
scripts/       data download (SIP) and data manifest
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
