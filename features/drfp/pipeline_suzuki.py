"""
pipeline_suzuki.py
==================
End-to-end feature engineering pipeline for Suzuki yield prediction.

Input  : data_suzuki_cleaned.csv
Outputs: X_features.npy, y_target.npy,
         X_train.npy, X_test.npy, y_train.npy, y_test.npy

Feature vector layout (per row):
  [ 2048-bit DRFP | 217 ligand RDKit descs | 217 base descs | 217 solvent descs ]
  = 2699 raw features  →  N features after cleaning
"""

import numpy as np
import pandas as pd
import math
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# 1. SMILES DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────

# --- Reactant 1 (halide / pseudo-halide / boronic acid coupling partner) ---
REACTANT_1_SMILES = {
    '6-chloroquinoline':                    'Clc1ccc2ncccc2c1',
    '6-Bromoquinoline':                     'Brc1ccc2ncccc2c1',
    '6-triflatequinoline':                  'FC(F)(F)S(=O)(=O)Oc1ccc2ncccc2c1',
    '6-Iodoquinoline':                      'Ic1ccc2ncccc2c1',
    '6-quinoline-boronic acid hydrochloride':'[Cl-].OB(O)c1ccc2[nH+]cccc2c1',
    'Potassium quinoline-6-trifluoroborate': '[K+].F[B-](F)(F)c1ccc2ncccc2c1',
    '6-Quinolineboronic acid pinacol ester': 'CC1(C)OB(c2cc3cccnc3cc2)OC1(C)C',
}

# --- Reactant 2 (pyrazole boronic acid / ester / trifluoroborate / bromide) ---
REACTANT_2_SMILES = {
    '2a, Boronic Acid':   'Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O',
    '2b, Boronic Ester':  'Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1',
    '2c, Trifluoroborate':'Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]',
    '2d, Bromide':        'Cc1ccc2c(cnn2C2CCCCO2)c1Br',
}

# --- Ligands ---
# Experiments without a ligand (ligand_eq == 0, ligand == NaN) are encoded
# with a zero vector — this is a deliberate experimental condition, not missing data.
#
# Note on Fe-containing ligands (dtbpf, dppf):
#   Simplified SMILES without the cyclopentadienyl rings are used here.
#   As a consequence, ~12 charge/BCUT descriptors return NaN for these two ligands.
#   These columns are dropped in the cleaning step (Step 5).
#
# Note on Xantphos:
#   Complex bisphosphine ligand — SMILES manually verified against the literature.
LIGAND_SMILES = {
    'P(tBu)3':     'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
    'P(Ph)3 ':     'P(c1ccccc1)(c1ccccc1)c1ccccc1',   # trailing space matches CSV
    'AmPhos':      'CN(C)c1ccc(P(C(C)(C)C)C(C)(C)C)cc1',
    'P(Cy)3':      'P(C1CCCCC1)(C1CCCCC1)C1CCCCC1',
    'P(o-Tol)3':   'P(c1ccccc1C)(c1ccccc1C)c1ccccc1C',
    'CataCXium A': 'CCCCP(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2',
    'SPhos':       'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
    'dtbpf':       'CC(C)(C)P([Fe]P(C(C)(C)C)C(C)(C)C)C(C)(C)C',
    'XPhos':       'CC(C)c1cc(C(C)C)cc(C(C)C)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
    'dppf':        'O=P(c1ccccc1)(c1ccccc1)[Fe]P(=O)(c1ccccc1)c1ccccc1',
    'Xantphos':    'O=P(c1ccccc1)(c1ccccc1)c1c(oc2c(C(C)(C)C3)cccc12)cccc3P(=O)(c1ccccc1)c1ccccc1',
}

# --- Bases ---
# Experiments without a base (base_eq == 0, base == NaN) → zero vector.
# Inorganic salts are represented by their active species (anion / neutral proxy):
#   NaOH / KOH  → water (proxy for OH⁻)
#   CsF         → HF   (proxy for F⁻)
#   K3PO4       → H3PO4 (neutral phosphoric acid proxy)
#   NaHCO3      → H2CO3 (neutral carbonic acid proxy)
#   LiOtBu      → tert-butanol (neutral form of tBuO⁻)
#   Et3N        → encoded directly (neutral organic base)
BASE_SMILES = {
    'NaOH':   'O',
    'NaHCO3': 'OC(=O)O',
    'CsF':    'F',
    'K3PO4':  'OP(=O)(O)O',
    'KOH':    'O',
    'LiOtBu': 'OC(C)(C)C',
    'Et3N':   'CCN(CC)CC',
}

# --- Solvents ---
# MeOH/H2O_V2 9:1 → represented by MeOH SMILES (majority solvent).
#   Alternative: weighted average  0.9 * desc_MeOH + 0.1 * desc_H2O (not implemented).
# THF_V2 → identical to THF (same molecule, different experimental run).
SOLVENT_SMILES = {
    'MeCN':             'CC#N',
    'THF':              'C1CCOC1',
    'DMF':              'CN(C)C=O',
    'MeOH':             'OC',
    'MeOH/H2O_V2 9:1': 'OC',
    'THF_V2':           'C1CCOC1',
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. RDKIT DESCRIPTOR CALCULATOR (217 descriptors)
# ─────────────────────────────────────────────────────────────────────────────

DESC_NAMES = [d[0] for d in Descriptors.descList]   # 217 descriptor names
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def smiles_to_descriptors(smi: str) -> np.ndarray:
    """
    Compute 217 RDKit descriptors for a given SMILES string.

    For salts (SMILES with '.'), the largest fragment is used.
    Returns an array of floats; some values may be NaN for organometallic
    structures (e.g., dtbpf, dppf — partial charges / BCUT descriptors
    not supported for Fe complexes).
    """
    parts = smi.split('.')
    best = max(
        parts,
        key=lambda s: (Chem.MolFromSmiles(s) or Chem.MolFromSmiles('C')).GetNumAtoms()
    )
    mol = Chem.MolFromSmiles(best)
    if mol is None:
        return np.full(len(DESC_NAMES), np.nan)
    return np.array(calculator.CalcDescriptors(mol), dtype=float)


# Pre-compute descriptors for each unique molecule (avoids repeated RDKit calls)
ligand_desc_cache  = {k: smiles_to_descriptors(v) for k, v in LIGAND_SMILES.items()}
base_desc_cache    = {k: smiles_to_descriptors(v) for k, v in BASE_SMILES.items()}
solvent_desc_cache = {k: smiles_to_descriptors(v) for k, v in SOLVENT_SMILES.items()}

# Zero vector used for absent ligand/base (deliberate experimental condition)
ZERO_DESC = np.zeros(len(DESC_NAMES))


def get_descriptor(cache: dict, key) -> np.ndarray:
    """Return descriptor vector from cache; return zero vector if key is NaN (absent species)."""
    if isinstance(key, float) and math.isnan(key):
        return ZERO_DESC
    return cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD DATA & BUILD REACTION SMILES
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("data_suzuki_cleaned.csv")
print(f"Dataset loaded: {df.shape[0]} reactions, columns: {df.columns.tolist()}")

# Map string names to SMILES for the two reactants
smiles_1 = df["reactant_1"].map(REACTANT_1_SMILES)
smiles_2 = df["reactant_2"].map(REACTANT_2_SMILES)

# Format as reaction SMILES  "reactant1.reactant2>>"  (no product known)
reaction_smiles = (smiles_1 + "." + smiles_2 + ">>").tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 4. DRFP FINGERPRINTS (2048-bit)
# ─────────────────────────────────────────────────────────────────────────────

print("Computing DRFP fingerprints...")
fps = DrfpEncoder.encode(
    reaction_smiles,
    n_folded_length=2048,   # binary vector length (power of 2)
    radius=3,               # max circular substructure radius
    min_radius=0,           # include single atoms (radius 0)
    rings=True,             # include whole rings as substructures
    show_progress_bar=True,
)
drfp_array = np.array(fps)          # shape: (N, 2048)
print(f"DRFP shape: {drfp_array.shape}")

# Sanity checks
zero_rows = np.where(drfp_array.sum(axis=1) == 0)[0]
print(f"  All-zero rows (invalid SMILES): {len(zero_rows)}")   # expected: 0
print(f"  Mean bit density: {drfp_array.mean():.4f}")          # typical: 0.04–0.15

# ─────────────────────────────────────────────────────────────────────────────
# 5. RDKIT DESCRIPTOR MATRICES (ligand / base / solvent)
# ─────────────────────────────────────────────────────────────────────────────

ligand_array  = np.array([get_descriptor(ligand_desc_cache,  l) for l in df["ligand"]])
base_array    = np.array([get_descriptor(base_desc_cache,    b) for b in df["base"]])
solvent_array = np.array([get_descriptor(solvent_desc_cache, s) for s in df["solvent"]])

print(f"Ligand descriptors:  {ligand_array.shape}")
print(f"Base descriptors:    {base_array.shape}")
print(f"Solvent descriptors: {solvent_array.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CONCATENATION & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

X_raw = np.concatenate([drfp_array, ligand_array, base_array, solvent_array], axis=1)
print(f"\nRaw feature matrix: {X_raw.shape}")  # expected: (N, 2699)

# Drop columns that are:
#   - NaN    : caused by organometallic ligands (dtbpf, dppf) — ~12 columns
#   - Inf    : numerical overflow in some RDKit descriptors
#   - Constant: zero variance → useless for any ML model
nan_mask  = np.isnan(X_raw).any(axis=0)
inf_mask  = np.isinf(X_raw).any(axis=0)
var_mask  = np.var(X_raw, axis=0) == 0

drop_mask = nan_mask | inf_mask | var_mask
X = X_raw[:, ~drop_mask]

print(f"Dropped columns: {drop_mask.sum()} "
      f"(NaN: {nan_mask.sum()}, Inf: {inf_mask.sum()}, Constant: {var_mask.sum()})")
print(f"Final feature matrix: {X.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. TARGET & TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

y = df["yield_uv"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nX_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}   |  y_test: {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE .npy FILES
# ─────────────────────────────────────────────────────────────────────────────

np.save("X_features.npy", X)
np.save("y_target.npy",   y)
np.save("X_train.npy",    X_train)
np.save("X_test.npy",     X_test)
np.save("y_train.npy",    y_train)
np.save("y_test.npy",     y_test)

print("\nSaved: X_features.npy, y_target.npy, X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("Pipeline complete ✅")
