# Data Directory — DeepMIMO Scenario

This directory is managed automatically by the `deepmimo` Python package.
The dataset is **auto-downloaded** when you run `python src/train.py` for the first time.

---

## Scenario: `asu_campus_3p5`

| Property | Value |
|----------|-------|
| Environment | Outdoor campus (Arizona State University) |
| Frequency | 3.5 GHz |
| Total UE positions | 131,931 |
| Valid users (non-blocked) | ~54,068 |
| BS antennas | 64-element ULA (configured at load time) |
| UE antennas | 1 (single-antenna) |
| Download size | ~29 MB |
| Source | [deepmimo.net](https://deepmimo.net) |

---

## Auto-Download (Recommended)

The pipeline downloads the scenario automatically on first use:

```bash
python src/train.py        # downloads and trains
python src/dataset.py --real  # downloads and runs smoke test
```

DeepMIMO v4 stores downloaded scenarios under:
```
C:\Projects\6G-ML\deepmimo_scenarios\asu_campus_3p5\
```

---

## Manual Download

If the auto-download fails:

1. Visit **[https://deepmimo.net/scenarios/asu-campus-35/](https://deepmimo.net)**
2. Download `asu_campus_3p5.zip`
3. Extract and place the folder at:
   ```
   C:\Projects\6G-ML\deepmimo_scenarios\asu_campus_3p5\
   ```
4. Verify with: `python -c "import deepmimo as dm; dm.load('asu_campus_3p5')"`

---

## Channel Configuration

When loaded by `load_deepmimo_v4()`, channels are generated with:

```python
params.bs_antenna['shape']  = [64, 1]   # 64-element ULA
params.ue_antenna['shape']  = [1, 1]    # single-antenna UE
params.num_paths            = 10        # multipath components
params.ofdm['subcarriers']  = 64
params.ofdm['selected_subcarriers'] = [0]  # baseline: subcarrier 0
```

Output: `channel.shape = (131931, 1, 64, 1)` → squeezed to `(131931, 1, 64)`

---

## Upgrading to O1_28B (mmWave)

The PRD originally specified O1_28B (28 GHz mmWave). To use it when available:

```bash
python src/train.py --scenario O1_28B
```

The code automatically handles both scenarios via `load_deepmimo_v4(scenario)`.
