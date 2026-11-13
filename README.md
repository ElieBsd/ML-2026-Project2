# CS-433 Project 2 - Copy Detection Pattern authentication (ML4Science, SIP lab UNIGE)

EPFL CS-433 Machine Learning, Fall 2026 - Project 2 (ML4Science).

- **Team:** Elie Bussod, Gabin Prunet, Antoine Mandly
- **Host lab:** Stochastic Information Processing (SIP) group, Université de Genève - Prof. Slava Voloshynovskiy, mentor Vitaliy Kinakh

## Topic
Copy detection patterns (CDP) are small printed random binary codes used against counterfeiting: printing and scanning blur them in a way that a copy cannot reproduce exactly. To check a product, the code is captured, binarized and compared to its secret template. A symbol read with the wrong colour is called a flip.

Joakim Tutt's PhD thesis (UNIGE, 2025) models this with the **binary pattern-based channel (BPC)**: the probability that a symbol flips depends only on its 3x3 neighbourhood in the template (a table of 512 values, the codebook, learned on originals), and symbols flip independently. Its main result: counting flips only on the symbols that rarely flip (the reliability mask) detects fakes that plain error counting misses.

**Question: do the guarantees of the BPC model hold on real data?** We study three things, on two datasets.

1. **What the 3x3 model misses.** With repeated prints of the same template, we measure how much of the noise predicted by the model is in fact reproducible from print to print, and how much comes from printing or from the capture. We then learn flip-probability models with a larger context (a 5x5 table with back-off, logistic regression, gradient-boosted trees) and compare them to the 3x3 table by held-out log-likelihood. Originals only.
2. **How far a threshold holds.** A threshold is set on originals only (conformal calibration). We measure how many originals it rejects when the scan, the print run, the printing session, the printer or the phone changes, and how many originals of the new condition are needed to recalibrate.
3. **Which test against which attacker.** We re-evaluate the scores of the thesis at fixed operating points, with confidence intervals. We test a simple mechanism: a fake is the attacker's estimate of the template, printed a second time, so its error profile can be predicted from the codebook of originals. Supervised classifiers that see fakes are reported as oracle references.

Two datasets with two roles:
- the **SIP Indigo datasets** (the data of the thesis) for measurement: repeated prints and scans, three sessions, two printers, two phones, 1440 templates;
- the **Scantrust CDP dataset** as an independent dataset, and the only one with an averaging attack (the attacker combines 50 prints of the same code).

Rules we follow:
- nothing is chosen on test data: thresholds, hyperparameters and model choices come from calibration, validation or development subsets;
- all images of a template stay in the same split, and a codebook never scores a template it was fitted on;
- the defender has no fakes: thresholds use originals only, and methods trained with fakes are labelled oracle;
- every false acceptance rate is reported with the realized miss rate and a 95 % confidence interval;
- fakes of the test subsets are opened once, after the methods are frozen.

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

The `scantrust` group alone is enough for the Scantrust part of each study. The Indigo part needs the three other groups.

### Scantrust CDP dataset
https://www.kaggle.com/datasets/scantrust/copy-detection-pattern-dataset

The script downloads the public zip and unpacks it into `dataset/unique_cdp/` and `dataset/batch_cdp/` (28550 PNG files).

Citation: E. Khermaza, I. Tkachenko, J. Picard, "Can Copy Detection Patterns be copied? Evaluating the performance of attacks and highlighting the role of the detector", IEEE WIFS 2021.

By hand, if the script fails: sign in to Kaggle, click **Download** > **Download dataset as zip**, move `archive.zip` to the root of this repository and unzip it there.

### SIP Indigo datasets
https://yareta.unige.ch/archives/2eab9a72-dd60-4a5f-ba95-0144cc2983aa (doi:[10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma](https://doi.org/10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma), R. Chaban and S. Voloshynovskiy, University of Geneva, 2023).

The SIP group publishes all its CDP data as one public 123 GB archive, and we only need part of it. The script reads the index of the remote zip with HTTP range requests and fetches the selected files into `dataset/sip_yareta/`. Each file is checked against its CRC. The files stay zipped and are read directly from the zips.

By hand, if the range requests ever stop working: download the archive from the page above and unzip the same folders into `dataset/sip_yareta/`.

### Compact Indigo pack
The experiments only use binarized images, so the Indigo zips are turned once into a compact pack of about 0.5 GB:

```bash
python scripts/make_compact.py              # reads dataset/sip_yareta/, writes dataset/indigo_compact/
```

One `.npz` file per print run and scan run, per phone and capture run, and per set of templates. Each holds the binarized 228x228 images (Otsu per image, then a 3x3 majority vote for scans), the `pcor` score, the mean grey level and the Otsu threshold. Until the methods freeze, `pcor` is left as NaN for the fakes of the test templates; after it, `python scripts/make_compact.py --test-pcor --overwrite --only "*fake*"` fills it in. `cdpbpc.io.load_compact` reads a file; with `stage="dev"` it only returns the fakes of the development templates (1 to 288). The pack falls under the same license as the data: it is not versioned and not redistributed.

### Image manifests
```bash
python scripts/make_manifest.py             # writes dataset/manifest.csv and dataset/manifest_indigo.csv
```

One row per image: template, original or fake, attack, session, print run, scan or capture run, and so on. Experiments select images by filtering these tables (`cdpbpc.io.read_manifest`).

## Repository layout
```
configs/       one YAML config per experiment
scripts/       data download, compact data pack, data manifests
dataset/       Scantrust and SIP data (not versioned)
src/cdpbpc/    library: binarization, 3x3 patterns, codebook, context models, scores, calibration, metrics, splits
experiments/   one script per experiment, writes results/ and figures
tests/         pytest
notebooks/     exploration only, no reported number comes from here
report/        LaTeX report and generated figures
references/    project description and thesis
```

## References
- `references/project2_description.pdf` - official CS-433 Project 2 description (Fall 2026).
- `references/Tutt_2025_PhD_thesis_CDP.pdf` - J. Tutt, *Un modèle mathématique d'impression et d'imagerie pour l'authentification des codes de détection de copie*, PhD thesis, Université de Genève, 2025. doi:[10.13097/archive-ouverte/unige:192628](https://doi.org/10.13097/archive-ouverte/unige:192628). Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); redistributed unmodified.
- R. Chaban et al., "Printing variability of copy detection patterns", IEEE WIFS 2022 - describes the Indigo variability dataset.
- SIP group code: https://github.com/sip-group/snf-it-dis-code (read as a specification only; no code is copied).
