# CS-433 Project 2 — Copy Detection Pattern authentication (ML4Science, SIP lab UNIGE)

EPFL CS-433 Machine Learning, Fall 2026 — Project 2 (ML4Science).

- **Team:** Elie Bussod, Gabin Prunet, Antoine Mandly
- **Host lab:** Stochastic Information Processing (SIP) group, Université de Genève — Prof. Slava Voloshynovskiy, mentor Vitaliy Kinakh
- **Deadline:** 2026-12-17, 16:00 (report + link to the final commit of this repository)

## Topic (draft)
Authentication of printed **copy detection patterns (CDP)**, building on Joakim Tutt's PhD thesis (UNIGE, 2025): reproduce the pattern-codebook baselines (binary pattern-based channel model, masked Hamming / log-likelihood scores), then study learned classifiers on these features under realistic protocols (unseen fakes, cross-device, cross-printer).

## Repository layout
```
references/   project description and background reading
```
Code, data loaders and experiments will be added as the project starts.

## Data
The datasets belong to the SIP lab and are **not** stored in this repository. Instructions to obtain them will be added here.

## References
- `references/project2_description.pdf` — official CS-433 Project 2 description (Fall 2026).
- `references/Tutt_2025_PhD_thesis_CDP.pdf` — J. Tutt, *Un modèle mathématique d'impression et d'imagerie pour l'authentification des codes de détection de copie*, PhD thesis, Université de Genève, 2025. doi:[10.13097/archive-ouverte/unige:192628](https://doi.org/10.13097/archive-ouverte/unige:192628). Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); redistributed unmodified.
- SIP group code: https://github.com/sip-group/snf-it-dis-code
