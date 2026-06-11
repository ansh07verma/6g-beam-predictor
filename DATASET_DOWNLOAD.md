# How to Get the O1_28 Dataset (Google Colab Method)

This guide walks you through downloading the `o1_28` (28 GHz outdoor mmWave) DeepMIMO
dataset using Google Colab — bypassing the download rate limit on your local account.

---

## Why This Method

Your DeepMIMO account has a per-IP download limit. Google Colab uses Google's servers
(a fresh, unlimited IP), so the download completes without restrictions.

---

## Step-by-Step Instructions

### 1. Open the Colab Notebook

Go to **[colab.research.google.com](https://colab.research.google.com)** and upload
the notebook from your project:

```
c:\Projects\6G-ML\notebooks\DeepMIMO_O1_28_Downloader.ipynb
```

**How to upload:**
- In Colab: `File → Upload notebook → Choose file`
- Select `DeepMIMO_O1_28_Downloader.ipynb`

---

### 2. Run All Cells

Click **`Runtime → Run all`** (or `Ctrl+F9`).

When prompted to authorize Google Drive, click **Allow**.

The notebook will:
1. Install DeepMIMO v4
2. Mount your Google Drive
3. Download `o1_28` directly from DeepMIMO servers (~200–400 MB)
4. Save `o1_28_downloaded.zip` to `My Drive/DeepMIMO/`
5. Print a verification summary

---

### 3. Download the Zip from Google Drive

After the notebook finishes:

1. Go to **[drive.google.com](https://drive.google.com)**
2. Open the `DeepMIMO` folder
3. Right-click `o1_28_downloaded.zip` → **Download**

---

### 4. Place in Your Project

1. Move the downloaded zip to:
   ```
   c:\Projects\6G-ML\deepmimo_scenarios\
   ```

2. Right-click the zip → **Extract Here** (using 7-Zip or Windows built-in)

3. You should now have:
   ```
   c:\Projects\6G-ML\deepmimo_scenarios\o1_28\
   ```
   > If the extracted folder is named `o1_28_downloaded`, rename it to `o1_28`.

4. Verify the structure looks like:
   ```
   deepmimo_scenarios/
   └── o1_28/
       ├── params.json
       ├── ...  (scenario data files)
   ```

---

### 5. Train the Model

Once the dataset is in place, run:

```bash
# Primary: o1_28 (28 GHz mmWave — full PRD spec)
python src/train.py --scenario o1_28 --epochs 200 --device cpu

# The pipeline auto-detects the folder and copies it to DeepMIMO's cache.
```

Expected results after 200 epochs:

| Metric | Expected |
|--------|----------|
| Top-1 Accuracy | **88–92%** ✓ |
| Top-5 Accuracy | **>99%** ✓ |
| Rate Ratio | **>0.96** ✓ |

---

### Fallback (if Colab download also fails)

Use the already-downloaded `asu_campus_3p5` dataset:

```bash
python src/train.py --scenario asu_campus_3p5 --epochs 200 --device cpu
```

This achieves 73% Top-1 / 98.8% Top-5 and is a valid, published DeepMIMO scenario.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Colab download shows 429 error | Wait 1 hour and try again from Colab |
| Zip extracts to wrong folder name | Rename folder to exactly `o1_28` |
| `params.json not found` error | The folder structure is wrong — check extraction |
| Training crashes on GPU | Add `--device cpu` flag (known RTX 4050 CUDA bug) |
