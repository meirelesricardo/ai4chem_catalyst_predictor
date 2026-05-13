import os
import numpy as np
import pandas as pd
import math
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import cdist
import joblib

# ─────────────────────────────────────────────
# Reaction component SMILES
# ─────────────────────────────────────────────

REACTANT_1_SMILES = {
    '6-chloroquinoline':                     'Clc1ccc2ncccc2c1',
    '6-Bromoquinoline':                      'Brc1ccc2ncccc2c1',
    '6-triflatequinoline':                   'FC(F)(F)S(=O)(=O)Oc1ccc2ncccc2c1',
    '6-Iodoquinoline':                       'Ic1ccc2ncccc2c1',
    '6-quinoline-boronic acid hydrochloride': '[Cl-].OB(O)c1ccc2[nH+]cccc2c1',
    'Potassium quinoline-6-trifluoroborate':  '[K+].F[B-](F)(F)c1ccc2ncccc2c1',
    '6-Quinolineboronic acid pinacol ester':  'CC1(C)OB(c2cc3cccnc3cc2)OC1(C)C',
}

REACTANT_2_SMILES = {
    '2a, Boronic Acid':    'Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O',
    '2b, Boronic Ester':   'Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1',
    '2c, Trifluoroborate': 'Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]',
    '2d, Bromide':         'Cc1ccc2c(cnn2C2CCCCO2)c1Br',
}

LIGAND_SMILES = {
    'P(tBu)3':     'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
    'P(Ph)3 ':     'P(c1ccccc1)(c1ccccc1)c1ccccc1',
    'AmPhos':      'CN(C)c1ccc(P(C(C)(C)C)C(C)(C)C)cc1',
    'P(Cy)3':      'P(C1CCCCC1)(C1CCCCC1)C1CCCCC1',
    'P(o-Tol)3':   'P(c1ccccc1C)(c1ccccc1C)c1ccccc1C',
    'CataCXium A': 'CCCCP(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2',
    'SPhos':       'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
    'dtbpf':       'CC(C)(C)P(C1=CC=C[CH-]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH-]1)C(C)(C)C.[Fe+2]',
    'XPhos':       'CC(C)c1cc(C(C)C)cc(C(C)C)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
    'dppf':        '[CH-]1C=CC(=C1)P(c2ccccc2)c3ccccc3.[CH-]1C=CC(=C1)P(c2ccccc2)c3ccccc3.[Fe+2]',
    'Xantphos':    'CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21',
}

BASE_SMILES = {
    'NaOH':   '[Na+].[OH-]',
    'NaHCO3': '[Na+].O=C([O-])O',
    'CsF':    '[Cs+].[F-]',
    'K3PO4':  '[K+].[K+].[K+].[O-]P(=O)([O-])[O-]',
    'KOH':    '[K+].[OH-]',
    'LiOtBu': '[Li+].CC(C)(C)[O-]',
    'Et3N':   'CCN(CC)CC',
}

SOLVENT_SMILES = {
    'MeCN': 'CC#N', 'THF': 'C1CCOC1', 'DMF': 'CN(C)C=O',
    'MeOH': 'OC', 'MeOH/H2O_V2 9:1': 'CO.O', 'THF_V2': 'C1CCOC1',
}

# ─────────────────────────────────────────────
# RDKit descriptor calculator
# ─────────────────────────────────────────────

DESC_NAMES = [d[0] for d in Descriptors.descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)

def smiles_to_descriptors(smi):
    parts = smi.split('.')
    best = max(parts, key=lambda s: (Chem.MolFromSmiles(s) or Chem.MolFromSmiles('C')).GetNumAtoms())
    mol = Chem.MolFromSmiles(best)
    return np.array(calculator.CalcDescriptors(mol), dtype=float) if mol else np.full(len(DESC_NAMES), np.nan)

ligand_cache  = {k: smiles_to_descriptors(v) for k, v in LIGAND_SMILES.items()}
base_cache    = {k: smiles_to_descriptors(v) for k, v in BASE_SMILES.items()}
solvent_cache = {k: smiles_to_descriptors(v) for k, v in SOLVENT_SMILES.items()}
ZERO = np.zeros(len(DESC_NAMES))

def get_desc(cache, key):
    return ZERO if isinstance(key, float) and math.isnan(key) else cache[key]

# ─────────────────────────────────────────────
# Main featurization
# ─────────────────────────────────────────────

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "suzuki_cleaned.csv"))

# Build reaction SMILES and encode with DRFP
reaction_smiles = (
    df["reactant_1"].map(REACTANT_1_SMILES) + "." +
    df["reactant_2"].map(REACTANT_2_SMILES) + ">>"
).tolist()

print("Encoding DRFP fingerprints...")
fps = DrfpEncoder.encode(
    reaction_smiles, n_folded_length=2048, radius=3,
    min_radius=0, rings=True, show_progress_bar=True
)
drfp_array = np.array(fps)

# Build condition descriptors (ligand / base / solvent)
ligand_array  = np.array([get_desc(ligand_cache,  l) for l in df["ligand"]])
base_array    = np.array([get_desc(base_cache,    b) for b in df["base"]])
solvent_array = np.array([get_desc(solvent_cache, s) for s in df["solvent"]])

# Concatenate and clean
X_raw = np.concatenate([drfp_array, ligand_array, base_array, solvent_array], axis=1)
drop  = np.isnan(X_raw).any(0) | np.isinf(X_raw).any(0) | (np.var(X_raw, 0) == 0)
X = X_raw[:, ~drop]
y = df["yield_uv"].values / 100.0

# Track how many DRFP columns remain after cleaning
n_drfp_clean = (~drop[:2048]).sum()
print(f"DRFP dims after cleaning: {n_drfp_clean} / 2048")

# ─────────────────────────────────────────────
# Stratified diversity-based train/test split
# Maximin + Jaccard within each yield bin ensures
# both chemical diversity and yield balance.
# ─────────────────────────────────────────────

print("\nRunning stratified diversity-based train/test split...")
n_bins = 10
bin_labels = pd.qcut(y, q=n_bins, labels=False)

train_idx, test_idx = [], []
for b in range(n_bins):
    bin_mask = np.where(bin_labels == b)[0]
    n_train_bin = int(len(bin_mask) * 0.8)

    X_bin = X[bin_mask, :n_drfp_clean]
    distances = cdist(X_bin, X_bin, metric='jaccard')
    selected = [np.random.default_rng(42).integers(0, len(X_bin))]
    for _ in range(n_train_bin - 1):
        min_dists = distances[:, selected].min(axis=1)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))

    train_idx.extend(bin_mask[selected])
    test_idx.extend(bin_mask[np.setdiff1d(np.arange(len(bin_mask)), selected)])

train_idx = np.array(train_idx)
test_idx  = np.array(test_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ─────────────────────────────────────────────
# RobustScaler — fit on train, apply to both
# ─────────────────────────────────────────────

scaler = RobustScaler()
X_train_scaled = np.clip(scaler.fit_transform(X_train), -10, 10)
X_test_scaled  = np.clip(scaler.transform(X_test), -10, 10)

# ─────────────────────────────────────────────
# Save everything
# ─────────────────────────────────────────────

out_dir = os.path.dirname(__file__)

np.save(os.path.join(out_dir, "X_features.npy"), X)
np.save(os.path.join(out_dir, "y_target.npy"),   y)
np.save(os.path.join(out_dir, "X_train.npy"), X_train_scaled)
np.save(os.path.join(out_dir, "X_test.npy"),  X_test_scaled)
np.save(os.path.join(out_dir, "y_train.npy"), y_train)
np.save(os.path.join(out_dir, "y_test.npy"),  y_test)
joblib.dump(scaler, os.path.join(out_dir, "feature_scaler.pth"))

print(f"\nDone ✅")
print(f"  Full dataset  — X: {X.shape}, y: {y.shape}")
print(f"  Train set     — {X_train_scaled.shape[0]} samples (80%)")
print(f"  Test set      — {X_test_scaled.shape[0]} samples (20%)")
print(f"  Split method  — stratified maximin | metric: jaccard")
print(f"  Scaler saved  — feature_scaler.pth")