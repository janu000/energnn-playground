# Project Guidelines

## Code Style
- Nutze Python 3.11+ lokal entsprechend [pyproject.toml](../pyproject.toml) (`requires-python = ">=3.11"`).
- Halte Zeilenlaenge bei maximal 127 Zeichen (Black/Flake8), siehe [pyproject.toml](../pyproject.toml).
- Bevorzuge kleine, gezielte Aenderungen ohne Reformatierung unbeteiligter Bereiche.

## Architecture
- Kernpaket ist [src/energnn](../src/energnn) mit klaren Domainen:
- `graph`: Datenstrukturen und Operationen auf Graphen.
- `graph/jax`: JAX-kompatible Spiegelung der Graph-Strukturen (PyTree, Konvertierung).
- `model` und `gnn`: Modellbausteine (Encoder/Coupler/Decoder/Normalizer).
- `problem`: Problem-, Dataset- und Loader-Abstraktionen.
- `trainer`: Trainings-Orchestrierung (inkl. Checkpointing/Tracking).
- `feature_store`: externer HTTP-Client und Konfig-Integration.
- Achte auf Boundary-Trennung: Datenmodell in `graph`, Modelllogik in `model`/`gnn`, Trainingsfluss in `trainer`, externe IO in `feature_store`.

## Build and Test
- Installation (CPU): `uv sync`
- Installation (GPU): `uv sync --extra gpu`
- CI-nahe Dev-Umgebung:
- `uv venv`
- `source .venv/bin/activate`
- `uv sync --group dev --extra docs`
- Lint: `uv run flake8 src/energnn --count --show-source --max-complexity=10 --max-line-length=127 --statistics`
- Tests: `uv run pytest --cov=. tests`
- Paket bauen: `uv build`
- Doku (streng wie CI): `cd docs && uv run make html SPHINXOPTS="-W"`

## Conventions
- Viele Kernobjekte sind dict-basierte Container statt Dataclasses; respektiere bestehende Zugriffsmuster.
- NumPy- und JAX-Graphpfade existieren parallel; konvertiere explizit zwischen Repräsentationen statt impliziter Annahmen.
- Flax NNX und JAX-Typkompatibilitaet haben Vorrang bei Modell- und Trainer-Code.
- Aendere Tests bevorzugt im gleichen Funktionsbereich wie der Codefix (z. B. Graph/JAX-Paritaet in [tests/graph](../tests/graph)).

## Pitfalls
- Versionshinweis: CI testet auch 3.10, waehrend [pyproject.toml](../pyproject.toml) `>=3.11` setzt. Bei Python-Versionsthemen zuerst diese Diskrepanz pruefen.
- Feature-Store-Code ist netzwerkabhaengig; in Tests externe Calls mocken und keine echten Endpunkte voraussetzen.
- Doku-Warnungen koennen CI brechen (`SPHINXOPTS="-W"`), auch wenn lokaler Build ohne `-W` erfolgreich ist.
