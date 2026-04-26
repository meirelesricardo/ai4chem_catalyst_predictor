"""
build_features.py
=================
Pipeline complet : DRFP (réactifs 1 & 2) + descripteurs RDKit (ligand, base, solvant)
→ matrice de features finale pour modélisation du yield Suzuki.

Structure du vecteur final par ligne :
  [  2048 bits DRFP  |  217 desc ligand  |  217 desc base  |  217 desc solvant  ]
  = 2699 features au total (avant suppression des NaN/colonnes constantes)
"""

import pandas as pd
import numpy as np
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# ─────────────────────────────────────────────────────────────────────────────
# 1. SMILES des conditions réactionnelles
# ─────────────────────────────────────────────────────────────────────────────

# --- Ligands ---
# Certaines expériences sont conduites SANS ligand (ligand_eq = 0, ligand = NaN
# dans le CSV). Ces lignes sont encodées avec un vecteur de zéros (217 valeurs),
# représentant l'absence totale de ligand. Ce n'est PAS une donnée manquante —
# c'est une condition expérimentale délibérée.
# Note : dtbpf et dppf contiennent un atome de Fe.
#   → Le SMILES utilisé ici est simplifié (sans les ligands cyclopentadiényles
#     du ferrocène). La vraie structure de dtbpf serait :
#     [Fe+2].[cH-]1cc(P(C(C)(C)C)C(C)(C)C)ccc1.[cH-]1cc(P(C(C)(C)C)C(C)(C)C)ccc1
#   → Conséquence : 12 descripteurs de charge/BCUT donnent NaN pour ces deux
#     ligands (MaxPartialCharge, MinPartialCharge, BCUT2D_*). Ces colonnes
#     seront supprimées en étape 5 (imputation ou drop).
# Note : Xantphos — SMILES complexe vérifié manuellement, correspond bien
#     au ligand bisphosphine utilisé dans le protocole Suzuki.
LIGAND_SMILES = {
    'P(tBu)3':      'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
    'P(Ph)3 ':      'P(c1ccccc1)(c1ccccc1)c1ccccc1',   # espace trailing conservé pour matcher le CSV
    'AmPhos':       'CN(C)c1ccc(P(C(C)(C)C)C(C)(C)C)cc1',
    'P(Cy)3':       'P(C1CCCCC1)(C1CCCCC1)C1CCCCC1',
    'P(o-Tol)3':    'P(c1ccccc1C)(c1ccccc1C)c1ccccc1C',
    'CataCXium A':  'CCCCP(C12CC3CC(CC(C3)C1)C2)C12CC3CC(CC(C3)C1)C2',
    'SPhos':        'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
    'dtbpf':        'CC(C)(C)P([Fe]P(C(C)(C)C)C(C)(C)C)C(C)(C)C',  # voir note ci-dessus
    'XPhos':        'CC(C)c1cc(C(C)C)cc(C(C)C)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1',
    'dppf':         'O=P(c1ccccc1)(c1ccccc1)[Fe]P(=O)(c1ccccc1)c1ccccc1',  # voir note ci-dessus
    'Xantphos':     'O=P(c1ccccc1)(c1ccccc1)c1c(oc2c(C(C)(C)C3)cccc12)cccc3P(=O)(c1ccccc1)c1ccccc1',
}

# --- Bases ---
# Certaines expériences sont conduites SANS base (base_eq = 0, base = NaN).
# Même logique que pour les ligands : encodage avec un vecteur de zéros.
# Les bases inorganiques sont des sels. On encode la partie active (anion/molécule
# neutre) car RDKit ne calcule pas de descripteurs moléculaires pertinents sur
# les contre-ions métalliques.
#   NaOH / KOH  → OH- représenté par l'eau (proxy le plus proche)
#   CsF         → F- représenté par HF
#   K3PO4       → acide phosphorique H3PO4 comme proxy neutre
#   NaHCO3      → acide carbonique H2CO3 comme proxy neutre
#   LiOtBu      → tert-butanol (forme protonée, neutre)
#   Et3N        → molécule neutre, encodée directement
BASE_SMILES = {
    'NaOH':     'O',            # proxy : OH- → eau
    'NaHCO3':   'OC(=O)O',     # proxy : HCO3- → acide carbonique
    'CsF':      'F',            # proxy : F- → HF
    'K3PO4':    'OP(=O)(O)O',  # proxy : PO4^3- → acide phosphorique
    'KOH':      'O',            # proxy : OH- → eau (même proxy que NaOH)
    'LiOtBu':   'OC(C)(C)C',   # tert-butanol (forme neutre de tBuO-)
    'Et3N':     'CCN(CC)CC',    # triéthylamine, base organique neutre
}

# --- Solvants ---
# MeOH/H2O_V2 9:1 : mélange 90% MeOH / 10% H2O.
#   → Encodé avec le SMILES du MeOH (solvant majoritaire). L'eau est ignorée.
#   → Alternative possible : moyenne pondérée des descripteurs MeOH et H2O
#     (0.9 * desc_MeOH + 0.1 * desc_H2O), non implémentée ici.
# THF_V2 : deuxième condition expérimentale avec du THF (même réactif,
#   conditions opératoires potentiellement différentes). SMILES identique à THF.
SOLVENT_SMILES = {
    'MeCN':             'CC#N',
    'THF':              'C1CCOC1',
    'DMF':              'CN(C)C=O',
    'MeOH':             'OC',
    'MeOH/H2O_V2 9:1': 'OC',      # proxy MeOH majoritaire — voir note ci-dessus
    'THF_V2':           'C1CCOC1', # identique à THF — voir note ci-dessus
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Calcul des descripteurs RDKit (217 descripteurs)
# ─────────────────────────────────────────────────────────────────────────────

DESC_NAMES = [d[0] for d in Descriptors.descList]  # 217 descripteurs
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def smiles_to_descriptors(smi: str) -> np.ndarray:
    """
    Calcule les 217 descripteurs RDKit pour un SMILES.
    Pour les sels (SMILES avec '.'), prend le fragment le plus lourd.
    Retourne un array de float (peut contenir des NaN pour les molécules
    avec métaux de transition comme dtbpf et dppf).
    """
    parts = smi.split('.')
    best_part = max(
        parts,
        key=lambda s: Chem.MolFromSmiles(s).GetNumAtoms()
                      if Chem.MolFromSmiles(s) is not None else 0
    )
    mol = Chem.MolFromSmiles(best_part)
    if mol is None:
        return np.full(len(DESC_NAMES), np.nan)
    return np.array(calculator.CalcDescriptors(mol), dtype=float)


# Précalcul des descripteurs pour chaque valeur unique (évite les recalculs)
ligand_desc_cache  = {name: smiles_to_descriptors(smi) for name, smi in LIGAND_SMILES.items()}
base_desc_cache    = {name: smiles_to_descriptors(smi) for name, smi in BASE_SMILES.items()}
solvent_desc_cache = {name: smiles_to_descriptors(smi) for name, smi in SOLVENT_SMILES.items()}

# Vecteur "absent" : utilisé quand ligand=NaN (ligand_eq=0) ou base=NaN (base_eq=0).
# Représente chimiquement l'absence de la molécule → tous les descripteurs à 0.
# Ce n'est PAS une imputation : c'est une condition expérimentale délibérée.
ZERO_DESCRIPTOR = np.zeros(len(DESC_NAMES))

def get_descriptor(cache: dict, key) -> np.ndarray:
    """Récupère le vecteur de descripteurs, gère les NaN (absent = zéro vecteur)."""
    import math
    if isinstance(key, float) and math.isnan(key):
        return ZERO_DESCRIPTOR
    return cache[key]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Chargement des données et génération des DRFP
# ─────────────────────────────────────────────────────────────────────────────

df_smiles  = pd.read_csv("data_suzuki_smiles_only.csv")
df_cleaned = pd.read_csv("data_suzuki_cleaned.csv")

# Construction des reaction SMILES (format DRFP : "reactant1.reactant2>>")
reaction_smiles = (
    df_smiles["Smiles 1"] + "." + df_smiles["Smiles 2"] + ">>"
).tolist()

print("Génération des fingerprints DRFP...")
fps = DrfpEncoder.encode(
    reaction_smiles,
    n_folded_length=2048,
    radius=3,
    min_radius=0,
    rings=True,
    show_progress_bar=True,
)
drfp_array = np.array(fps)  # (5760, 2048)
print(f"DRFP shape : {drfp_array.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Construction des matrices de descripteurs ligand / base / solvant
# ─────────────────────────────────────────────────────────────────────────────

ligand_array  = np.array([get_descriptor(ligand_desc_cache,  l) for l in df_cleaned["ligand"]])   # (5760, 217)
base_array    = np.array([get_descriptor(base_desc_cache,    b) for b in df_cleaned["base"]])     # (5760, 217)
solvent_array = np.array([get_descriptor(solvent_desc_cache, s) for s in df_cleaned["solvent"]])  # (5760, 217)

print(f"Ligand descriptors shape  : {ligand_array.shape}")
print(f"Base descriptors shape    : {base_array.shape}")
print(f"Solvent descriptors shape : {solvent_array.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Concaténation et nettoyage
# ─────────────────────────────────────────────────────────────────────────────

X_raw = np.concatenate([drfp_array, ligand_array, base_array, solvent_array], axis=1)
print(f"\nShape avant nettoyage : {X_raw.shape}")  # attendu : (5760, 2699)

# Suppression des colonnes avec NaN ou variance nulle (constantes)
# Les 12 descripteurs NaN viennent de dtbpf et dppf (charges partielles
# et BCUT2D non calculables sur les complexes organométalliques Fe).
nan_mask      = np.isnan(X_raw).any(axis=0)
inf_mask      = np.isinf(X_raw).any(axis=0)
var_mask      = np.var(X_raw, axis=0) == 0  # colonnes constantes → inutiles pour ML

drop_mask = nan_mask | inf_mask | var_mask
X = X_raw[:, ~drop_mask]

print(f"Colonnes supprimées : {drop_mask.sum()} "
      f"(NaN: {nan_mask.sum()}, inf: {inf_mask.sum()}, constantes: {var_mask.sum()})")
print(f"Shape finale : {X.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Récupération des targets et split train/test
# ─────────────────────────────────────────────────────────────────────────────

y = df_cleaned["yield_uv"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nX_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"y_train : {y_train.shape}   |  y_test : {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

np.save("X_features.npy", X)
np.save("y_target.npy",   y)
np.save("X_train.npy",    X_train)
np.save("X_test.npy",     X_test)
np.save("y_train.npy",    y_train)
np.save("y_test.npy",     y_test)

print("\nFichiers sauvegardés : X_features.npy, y_target.npy, X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("Pipeline terminé ✅")
