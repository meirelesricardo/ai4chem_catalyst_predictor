# Catalyst & Yield Prediction

This project aims to predict:
- reaction yield
- optimal catalyst

Given two reactants as SMILES strings, the pipeline recommends the best combination of **ligand, base, and solvent** for a Suzuki–Miyaura cross-coupling reaction, along with the predicted UV yield for each combination.

---

## Chemical Context

The **Suzuki–Miyaura coupling** is a palladium-catalyzed cross-coupling reaction between an aryl halide and a boronic acid (or ester) in the presence of a base:

$$\text{Ar-X} + \text{Ar'-B(OH)}_2 \xrightarrow[\text{base, solvent}]{\text{Pd(0)}} \text{Ar-Ar'} + \text{X-B(OH)}_2$$

This reaction is critical in pharmaceutical synthesis (e.g. Valsartan, Losartan). Identifying optimal reaction conditions currently relies on costly trial-and-error. This project automates that search using machine learning.

### Dataset

The dataset (`aap9112_data_file_s1.xlsx`) comes from **Perera et al. (*Science*, 2018)**: *"A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow"*. It contains **5,760 reactions** from a High-Throughput Experimentation (HTE) platform, covering:

| Component | Options |
|---|---|
| Electrophile (Reactant 1) | 7 aryl halides / pseudohalides |
| Nucleophile (Reactant 2) | 4 organoboron species |
| Catalyst | Pd(OAc)₂ (constant) |
| Ligand | 11 phosphines + "no ligand" |
| Base | 7 bases + "no base" |
| Solvent | 6 solvents |

The target variable is `Product_Yield_PCT_Area_UV` — the reaction yield measured by HPLC-UV (%).

---

## Project Structure

```
.
├── data/
│   ├── raw/                   # Original dataset (aap9112_data_file_s1.xlsx)
│   ├── processed/             # Cleaned dataset (suzuki_cleaned.csv)
│   └── final/                 # Feature matrices ready for model training (.npy)
│       └── drfp/
│           └── Data V2/
│               └── Data V2 Optimized/
├── features/                  # Feature engineering outputs and scaler
├── models/                    # Saved models and utilities
│   ├── model_utils.py         # Shared evaluation functions and NN classes
│   ├── RandomForest.pth
│   ├── XGBoost.pth
│   └── lightning_nn.ckpt
├── notebooks/                 # Experiment notebooks (run in order)
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_features_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_testing_and_comparing.ipynb
└── README.md
```

---

## Pipeline Overview

The project is structured as a sequential pipeline across 5 notebooks.

### `01_data_exploration.ipynb` — Data Exploration
Exploratory analysis of the raw HTE dataset. Key findings:
- 5,760 reactions across a full combinatorial grid of conditions
- Target variable `yield_uv` spans 0–100%, with a median of ~33.6% and a right-skewed distribution
- 480 reactions run without ligand and 720 without base, these are **intentional controls**, not missing data
- `THF` and `THF_V2` appear as duplicate solvent entries with a ~12% median yield difference

### `02_data_cleaning.ipynb` — Data Cleaning
Prepares the dataset for feature engineering. Operations performed:

| Step | Detail |
|---|---|
| Missing values | NaN ligand/base → `"None"` (categorical), equivalents → `0.0` |
| Constant columns dropped | `Catalyst_1_Short_Hand`, `Catalyst_1_eq`, `Reactant_1_eq`, `Reactant_2_eq`, `Reactant_1_mmol` |
| Redundant columns dropped | `Reactant_1_Short_Hand`, `Product_Yield_Mass_Ion_Count` (r ≈ 0.35 with UV yield) |
| THF_V2 removed | 96 rows removed (~1.7%) to avoid batch inconsistency |
| Columns renamed | Standardized to lowercase snake_case |
| Outliers | None removed, all yields within [0, 100]%; low yields are valid failed experiments |

Output: `data/processed/suzuki_cleaned.csv`

### `03_features_engineering.ipynb` — Feature Engineering
Converts the cleaned tabular data into a numeric feature matrix using two complementary representations:

**DRFP (Differential Reaction FingerPrint)**
Encodes the reaction itself as a 2048-bit binary vector based on bond changes between reactants. Parameters: `radius=3`, `min_radius=0`, `rings=True`.

**RDKit Molecular Descriptors**
209 physicochemical descriptors (MW, LogP, TPSA, H-bond donors/acceptors, etc.) computed for the ligand, base, and solvent via their SMILES strings. Multi-fragment SMILES (salts, metallocenes) are handled by selecting the largest organic fragment.

The final feature matrix concatenates the DRFP vector with the descriptor vectors for ligand, base, and solvent. A `StandardScaler` is fitted on the training set and saved for inference.

### `04_model_training.ipynb` — Model Training
Trains three regressors on the feature matrix. Each model takes a reaction descriptor vector as input and predicts the UV yield. The trained models are evaluated across multiple train/test splits (50/50 to 90/10) for robustness assessment.

**Strategy:** for a given pair of reactants, all 576 condition combinations (12 ligands × 8 bases × 6 solvents) are scored, then sorted by predicted yield to recommend optimal conditions.

**Models trained:**

| Model | Description |
|---|---|
| **Random Forest** | Ensemble of decision trees; tuned with `HalvingGridSearchCV` (10-fold CV) |
| **XGBoost** | Gradient-boosted trees; tuned with `HalvingGridSearchCV` (5-fold CV) |
| **Neural Network** | PyTorch Lightning feedforward network with early stopping |

Saved to `models/`.

### `05_model_testing_and_comparing.ipynb` — Evaluation & Comparison
Full comparative evaluation of all three models on the held-out test set. Includes:
- Parity plots (predicted vs. actual yield)
- Error histograms and boxplots
- Side-by-side metrics bar charts (RMSE, MAE, R², Explained Variance)
- SHAP-based feature importance analysis

---

## Installation

```bash
git clone https://github.com/meirelesricardo/ai4chem_catalyst_predictor.git
cd ai4chem_catalyst_predictor
pip install -r requirements.txt
```

Key dependencies: `rdkit`, `drfp`, `scikit-learn`, `xgboost`, `torch`, `pytorch-lightning`, `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`

---

## Usage

Run the notebooks in order:

```
01 → 02 → 03 → 04 → 05
```

Each notebook reads from the output of the previous one. Start with the raw dataset in `data/raw/`.

---

## Evaluation Metrics

Models are assessed with four metrics:

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **MAE** | Mean Absolute Error — interpretable in yield % |
| **R²** | Coefficient of determination — variance explained |
| **Explained Variance** | Similar to R², robust to mean shifts |

---

## References

- Perera, D. et al. *A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow.* **Science**, 2018.
- Probst, D. et al. *Reaction classification and yield prediction using the differential reaction fingerprint DRFP.* **Digital Discovery**, 2022.