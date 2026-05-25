"""
How to run the predictor on terminal:

cd predictor
python run_predictor2.py


This script allows the user to run a prediction by choosing from the possible
reagents in the dataset (7 type 1 reagents and 4 type 2 reagents). The model 
used is XGBoost; the predictions provide the 5 best catalytic systems with their yields.
"""

# Importing libraries
import itertools
import math
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from smiles_data import REACTANT_1_SMILES, REACTANT_2_SMILES, LIGAND_SMILES, BASE_SMILES, SOLVENT_SMILES

# Defining paths to data and model (XGBoost and scaler)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CSV_PATH       = os.path.join(ROOT_DIR, "data", "processed", "suzuki_cleaned.csv")
XGB_PATH       = os.path.join(MODELS_DIR, "XGBoost.pth")
SCALER_PATH    = os.path.join(ROOT_DIR, "data", "final", "scaler.pth")
KEEP_MASK_PATH = os.path.join(ROOT_DIR, "data", "final", "keep_mask.npy")

# RDKit descriptor setup
_DESC_NAMES  = [d[0] for d in Descriptors.descList]
_CALCULATOR  = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
_N_DESC      = len(_DESC_NAMES)
_ZERO_DESC   = np.zeros(_N_DESC)


def _smiles_to_desc(smi: str) -> np.ndarray:
    """
    This function allows the user to convert a given SMILES in string form into a vector
    of descriptors using RDKit.

    Input:
        smi: a SMILES string representing a molecule
    Output:
        a numpy array of molecular descriptors for the molecule, or a zero vector if 
        the SMILES is invalid
    """
    parts = smi.split(".")
    best  = max(parts, key=lambda s: (Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C")).GetNumAtoms())
    mol   = Chem.MolFromSmiles(best)
    return np.array(_CALCULATOR.CalcDescriptors(mol), dtype=float) if mol else np.full(_N_DESC, np.nan)


def _get_desc(cache: dict, key) -> np.ndarray:
    """
    This function handles cases where the reaction does not contain a base/ligand. 
    In these specific cases, the function returns a vector of 0s instead of searching 
    for a non-existent molecule.

    Input: 
        cache: dictionary containing precomputed descriptor vectors
        key: key to look up in the cache
    Output: 
        a numpy array of molecular descriptors for the given key, or a zero vector if 
        key is None or NaN
    """
    if key is None or (isinstance(key, float) and math.isnan(key)):
        return _ZERO_DESC
    return cache.get(key, _ZERO_DESC)


def _build_desc_caches() -> tuple[dict, dict, dict]:
    """
    This function allows the script to perform a pre-calculation for all known RDKit 
    descriptor vectors (ligands, bases, solvents). With this cache, the calculation 
    is performed only once per molecule at startup.

    Input: 
        None
    Output: 
        Three dictionaries containing the precomputed descriptor vectors for ligands, 
        bases, and solvents, respectively.
    """
    print("Computing molecular descriptors for all catalytic components …", flush=True)
    ligand_cache  = {k: _smiles_to_desc(v) for k, v in LIGAND_SMILES.items()}
    base_cache    = {k: _smiles_to_desc(v) for k, v in BASE_SMILES.items()}
    solvent_cache = {k: _smiles_to_desc(v) for k, v in SOLVENT_SMILES.items()}
    return ligand_cache, base_cache, solvent_cache


def _pick_from_list(prompt: str, options: list) -> str:
    """
    This function allows the user to easily choose their combination of two reactants.

    Input:
        prompt: a string to display to the user
        options: a list of options for the user to choose from
    Output:
        The selected option as a string
    """
    print(f"\n{prompt}")
    for i, name in enumerate(options, 1):
        print(f"  {i:2d}. {name}")
    while True:
        raw = input("Your choice (number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(options)}.")


def main() -> None:
    """
    Main function for the Suzuki Coupling — Full Combination Predictor (XGBoost).

    This function handles the user interaction, loads the model and scaler,
    builds descriptor caches, and performs predictions for the chosen reactants.
    It displays the top 5 predicted catalytic conditions based on the user's choice of reactants.
    """
    # Display the header
    print("=" * 43)
    print("Suzuki Coupling - Catalytic Yield Predictor")
    print("=" * 43)

    # Load the model and the scaler
    print("\nLoading model and scaler …", flush=True)
    with warnings.catch_warnings():
        """
        Suppress UserWarning from XGBoost about feature names.
        """
        warnings.simplefilter("ignore", UserWarning)
        xgb = joblib.load(XGB_PATH)
    scaler = torch.load(SCALER_PATH, weights_only=False)
    df     = pd.read_csv(CSV_PATH)

    # Build descriptor caches and load the saved drop-mask
    ligand_cache, base_cache, solvent_cache = _build_desc_caches()
    keep_mask = np.load(KEEP_MASK_PATH)

    # Pre-compute combination catalogue (same for every prediction)
    unique_ligands  = [None] + sorted(l for l in df["ligand"].dropna().unique())
    unique_bases    = [None] + sorted(b for b in df["base"].dropna().unique())
    unique_solvents = sorted(df["solvent"].unique().tolist())
    combos          = list(itertools.product(unique_ligands, unique_bases, unique_solvents))

    # Define options for reactant selection
    r1_options = sorted(df["reactant_1"].unique().tolist())
    r2_options = sorted(df["reactant_2"].unique().tolist())

    # Main loop to allow multiple predictions without restarting the script
    while True:
        # Reactant selection
        r1 = _pick_from_list("Choose your first reactant:", r1_options)
        r2 = _pick_from_list("Choose your second reactant:", r2_options)

        # Compute DRFP once for the chosen reactant pair
        print(f"\nComputing DRFP fingerprint for {r1} + {r2} …", flush=True)
        rxn_smi = REACTANT_1_SMILES[r1] + "." + REACTANT_2_SMILES[r2] + ">>"
        drfp = np.array(
            DrfpEncoder.encode([rxn_smi], n_folded_length=2048, radius=3,
                               min_radius=0, rings=True)[0],
            dtype=float,
        )

        n_combos = len(combos)
        print(
            f"\nTesting {n_combos} combinations  "
            f"({len(unique_ligands)} ligands × {len(unique_bases)} bases × {len(unique_solvents)} solvents) …",
            flush=True,
        )

        # Build feature matrix: one row per combination
        rows = []
        for ligand, base, solvent in combos:
            x_raw = np.concatenate([
                drfp,
                _get_desc(ligand_cache,  ligand),
                _get_desc(base_cache,    base),
                _get_desc(solvent_cache, solvent),
            ])
            rows.append(x_raw[keep_mask])
        X = np.vstack(rows)

        # Apply the same single StandardScaler.transform used during training
        X_ready = scaler.transform(X)

        # Predict (model output is in [0, 1] fractions)
        y_pred_pct = xgb.predict(X_ready) * 100.0

        # Display top 5
        top5_idx = np.argsort(y_pred_pct)[::-1][:5]

        print("\n" + "=" * 65)
        print("  Top 5 predicted catalytic conditions")
        print(f"    Reactant 1 : {r1}")
        print(f"    Reactant 2 : {r2}")
        print("=" * 65)
        print(f"{'Rank':<5} {'Ligand':<16} {'Base':<10} {'Solvent':<22} {'Predicted Yield':>16}")
        print("-" * 70)
        for rank, idx in enumerate(top5_idx, 1):
            ligand, base, solvent = combos[idx]
            l_str = ligand if ligand is not None else "None"
            b_str = base   if base   is not None else "None"
            print(f"{rank:<5} {l_str:<16} {b_str:<10} {solvent:<22} {y_pred_pct[idx]:>15.1f} %")
        print("=" * 65)

        # Ask whether to run another prediction
        while True:
            print("\nWould you like to make another prediction?")
            again = input("  [y] Yes   [n] No  → ").strip().lower()
            if again in ("y", "n"):
                break
            print("  Please enter 'y' or 'n'.")
        if again != "y":
            break


if __name__ == "__main__":
    main()