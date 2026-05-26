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

The target variable is `Product_Yield_PCT_Area_UV`, the reaction yield measured by HPLC-UV (%).

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
├── predictor/                 # Inference tools (app + CLI)
│   ├── app.py                 # Streamlit web application
│   ├── run_predictor.py       # Command-line predictor script
│   └── smiles_data.py         # SMILES dictionaries for all components
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

## Model Performance

Model performance was assessed through four metrics: RMSE, MAE, R² score, and Explained Variance. Parity plots (predicted vs. actual yield) were generated for each model.

| Model | RMSE | MAE | R² | Explained Variance |
|---|---|---|---|---|
| **Random Forest** | 0.1076 | 0.0760 | 0.8566 | 0.8566 |
| **XGBoost** | 0.1089 | 0.0800 | 0.8531 | 0.8532 |
| **Neural Network** | 0.1146 | 0.0842 | 0.8374 | 0.8378 |

All three models produce similar parity plots, with predictions tightly distributed around the ideal diagonal. The Neural Network shows a slightly more dispersed point cloud, consistent with its marginally lower R² and higher error metrics. One shared behaviour across all models is a tendency to **overestimate yields when the actual yield is near zero**, likely because very low-yield reactions are harder to distinguish from moderately low-yield ones in the feature space.

Conversely, all models show a mild tendency to **underestimate high yields**, though analysis of high-confidence predictions (where the model outputs a high yield) shows relatively strong precision in that regime. This asymmetry is actually desirable for the intended use case: when the model predicts a high yield, it is likely to be correct, which is exactly what matters when screening for optimal conditions.

In terms of raw metrics, **Random Forest is the best-performing model**, closely followed by XGBoost. However, **XGBoost is the preferred model for deployment**: it achieves nearly identical performance while being orders of magnitude smaller on disk (~3 MB vs. ~150 MB for Random Forest), resulting in faster load times and leaner inference. This is why XGBoost is used in both the Streamlit app and the CLI predictor.

---

## Inference Tools

Two ready-to-use tools are provided in the `predictor/` directory for running predictions on new reactant pairs without going through the notebooks.

### `app.py` — Streamlit Web Application

A browser-based graphical interface for exploring reaction conditions interactively.

**Features:**
- Dropdown menus to select the first and second reactant from the dataset
- Live rendering of both molecular structures (SVG via RDKit) with stereo annotations
- One-click prediction over all 576 ligand–base–solvent combinations
- Results displayed as an interactive ranked table with a progress bar for each predicted yield
- Model and descriptor caches are loaded once at startup and reused across predictions (`@st.cache_resource`)

**To launch:**
```bash
cd predictor
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Select two reactants, click **"Predict top 5 conditions"**, and the table will display the five highest-yielding ligand–base–solvent combinations along with their predicted yields.

> The app uses the **XGBoost** model with DRFP fingerprints and RDKit molecular descriptors, identical to the training pipeline.

---

### `run_predictor.py` — Command-Line Predictor

A terminal-based alternative for environments where a browser interface is not available or preferred.

**Features:**
- Interactive numbered menus to select reactants from the dataset (7 type-1 × 4 type-2 reactants)
- Displays all available options with indices for easy selection
- Evaluates all 576 combinations and prints the top 5 results as a formatted table
- Supports multiple predictions in sequence without restarting the script (loop with `y/n` prompt)
- Identical feature engineering and model inference as the Streamlit app

**To run:**
```bash
cd predictor
python run_predictor.py
```

**Example output:**
```
===========================================
Suzuki Coupling - Catalytic Yield Predictor
===========================================

Choose your first reactant:
   1. ...
   2. ...

...

===================================================================
  Top 5 predicted catalytic conditions
    Reactant 1 : <selected>
    Reactant 2 : <selected>
===================================================================
Rank  Ligand           Base       Solvent                Predicted Yield
----------------------------------------------------------------------
1     ...              ...        ...                           87.3 %
2     ...              ...        ...                           84.1 %
...
===================================================================
```

Both tools share the same `smiles_data.py` module containing the SMILES dictionaries for all reactants, ligands, bases, and solvents.

---

## Installation

```bash
git clone https://github.com/meirelesricardo/ai4chem_catalyst_predictor.git
cd ai4chem_catalyst_predictor
pip install -r requirements.txt
```

Key dependencies: `rdkit`, `drfp`, `scikit-learn`, `xgboost`, `torch`, `pytorch-lightning`, `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `streamlit`

---

## Usage

Run the notebooks in order:

```
01 → 02 → 03 → 04 → 05
```

Each notebook reads from the output of the previous one. Start with the raw dataset in `data/raw/`.

Once the models are trained, use either inference tool from the `predictor/` directory:

```bash
# Web interface
streamlit run app.py

# Command-line interface
python run_predictor.py
```

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
- Schwaller, P. et al. *Prediction of Chemical Reaction Yields Using Deep Learning*. 2020.