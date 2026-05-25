"""
Test suite for run_predictor.py

Run with:
    cd predictor
    python -m pytest test_run_predictor.py -v
  or simply:
    python test_run_predictor.py
"""


# Importing libraries
import math
import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch
import numpy as np

# Make sure the predictor directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_predictor as rp
from smiles_data import (
    BASE_SMILES,
    LIGAND_SMILES,
    REACTANT_1_SMILES,
    REACTANT_2_SMILES,
    SOLVENT_SMILES,
)


# ---------------------------------------------------------------------------
# 1. Helper: _smiles_to_desc
# ---------------------------------------------------------------------------
class TestSmilesToDesc(unittest.TestCase):

    def test_valid_smiles_returns_array(self):
        """A valid SMILES should produce a finite numpy array."""
        desc = rp._smiles_to_desc("c1ccccc1")  # benzene
        self.assertIsInstance(desc, np.ndarray)
        self.assertEqual(desc.shape[0], rp._N_DESC)
        # At least some descriptors should be non-zero for benzene
        self.assertTrue(np.any(desc != 0))

    def test_invalid_smiles_returns_nan_vector(self):
        """An invalid SMILES should return a vector of NaN."""
        desc = rp._smiles_to_desc("NOT_A_SMILES!!!")
        self.assertIsInstance(desc, np.ndarray)
        self.assertEqual(desc.shape[0], rp._N_DESC)
        self.assertTrue(np.all(np.isnan(desc)))

    def test_multi_component_smiles_picks_largest(self):
        """Dot-separated SMILES: the largest fragment should drive the descriptors."""
        # Ethanol vs benzene: benzene is larger
        desc_multi  = rp._smiles_to_desc("CCO.c1ccccc1")
        desc_single = rp._smiles_to_desc("c1ccccc1")
        np.testing.assert_array_almost_equal(desc_multi, desc_single, decimal=6)

    def test_all_known_smiles_parseable(self):
        """Every SMILES in smiles_data should parse without NaN."""
        all_smiles = (
            list(REACTANT_1_SMILES.values())
            + list(REACTANT_2_SMILES.values())
            + list(LIGAND_SMILES.values())
            + list(BASE_SMILES.values())
            + list(SOLVENT_SMILES.values())
        )
        for smi in all_smiles:
            with self.subTest(smi=smi):
                desc = rp._smiles_to_desc(smi)
                # Allow some NaN descriptors but not a fully-NaN vector
                self.assertFalse(
                    np.all(np.isnan(desc)),
                    msg=f"All-NaN descriptor vector for SMILES: {smi}",
                )


# ---------------------------------------------------------------------------
# 2. Helper: _get_desc
# ---------------------------------------------------------------------------
class TestGetDesc(unittest.TestCase):

    def setUp(self):
        self.cache = {"benzene": np.ones(rp._N_DESC)}

    def test_existing_key(self):
        result = rp._get_desc(self.cache, "benzene")
        np.testing.assert_array_equal(result, np.ones(rp._N_DESC))

    def test_none_key_returns_zero_vector(self):
        result = rp._get_desc(self.cache, None)
        np.testing.assert_array_equal(result, rp._ZERO_DESC)

    def test_nan_key_returns_zero_vector(self):
        result = rp._get_desc(self.cache, float("nan"))
        np.testing.assert_array_equal(result, rp._ZERO_DESC)

    def test_missing_key_returns_zero_vector(self):
        result = rp._get_desc(self.cache, "not_in_cache")
        np.testing.assert_array_equal(result, rp._ZERO_DESC)


# ---------------------------------------------------------------------------
# 3. _build_desc_caches
# ---------------------------------------------------------------------------
class TestBuildDescCaches(unittest.TestCase):

    def test_cache_keys_match_smiles_dicts(self):
        ligand_cache, base_cache, solvent_cache = rp._build_desc_caches()
        self.assertEqual(set(ligand_cache.keys()),  set(LIGAND_SMILES.keys()))
        self.assertEqual(set(base_cache.keys()),    set(BASE_SMILES.keys()))
        self.assertEqual(set(solvent_cache.keys()), set(SOLVENT_SMILES.keys()))

    def test_cache_values_are_arrays_of_correct_shape(self):
        ligand_cache, base_cache, solvent_cache = rp._build_desc_caches()
        for name, vec in {**ligand_cache, **base_cache, **solvent_cache}.items():
            with self.subTest(name=name):
                self.assertIsInstance(vec, np.ndarray)
                self.assertEqual(vec.shape[0], rp._N_DESC)


# ---------------------------------------------------------------------------
# 4. Files exist
# ---------------------------------------------------------------------------
class TestFilePaths(unittest.TestCase):

    def test_xgb_model_exists(self):
        self.assertTrue(os.path.isfile(rp.XGB_PATH), f"XGBoost model not found: {rp.XGB_PATH}")

    def test_scaler_exists(self):
        self.assertTrue(os.path.isfile(rp.SCALER_PATH), f"Scaler not found: {rp.SCALER_PATH}")

    def test_keep_mask_exists(self):
        self.assertTrue(os.path.isfile(rp.KEEP_MASK_PATH), f"keep_mask not found: {rp.KEEP_MASK_PATH}")

    def test_csv_exists(self):
        self.assertTrue(os.path.isfile(rp.CSV_PATH), f"CSV not found: {rp.CSV_PATH}")


# ---------------------------------------------------------------------------
# 5. Full end-to-end prediction (non-interactive, first reactants, one pass)
# ---------------------------------------------------------------------------
class TestFullPrediction(unittest.TestCase):

    def test_main_runs_and_outputs_top5(self):
        """
        Simulate a full run of main() by patching `input` so that:
          - Reactant 1 → choice 1 (first in sorted list)
          - Reactant 2 → choice 1 (first in sorted list)
          - 'Run again?' → 'n'
        We verify the printed output contains the Top-5 table header.
        """
        # Provide answers: r1=1, r2=1, no repeat
        input_values = iter(["1", "1", "n"])

        with patch("builtins.input", lambda _: next(input_values)):
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                rp.main()
                output = mock_out.getvalue()

        self.assertIn("Top 5 predicted catalytic conditions", output)
        self.assertIn("Predicted Yield", output)
        # Should print exactly 5 data rows (rank 1-5)
        for rank in range(1, 6):
            self.assertIn(str(rank), output)

    def test_predicted_yields_are_in_valid_range(self):
        """
        Run a second prediction and verify yield values look reasonable
        (model output * 100 should be within [0, 120] %).
        """
        import warnings
        import joblib
        import torch
        import pandas as pd
        import itertools
        from drfp import DrfpEncoder

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb = joblib.load(rp.XGB_PATH)

        scaler    = torch.load(rp.SCALER_PATH, weights_only=False)
        df        = pd.read_csv(rp.CSV_PATH)
        keep_mask = np.load(rp.KEEP_MASK_PATH)

        ligand_cache, base_cache, solvent_cache = rp._build_desc_caches()

        # Use the first reactant pair in the dataset
        r1 = sorted(df["reactant_1"].unique())[0]
        r2 = sorted(df["reactant_2"].unique())[0]

        rxn_smi = REACTANT_1_SMILES[r1] + "." + REACTANT_2_SMILES[r2] + ">>"
        drfp = np.array(
            DrfpEncoder.encode([rxn_smi], n_folded_length=2048, radius=3,
                               min_radius=0, rings=True)[0],
            dtype=float,
        )

        unique_ligands  = [None] + sorted(l for l in df["ligand"].dropna().unique())
        unique_bases    = [None] + sorted(b for b in df["base"].dropna().unique())
        unique_solvents = sorted(df["solvent"].unique().tolist())
        combos          = list(itertools.product(unique_ligands, unique_bases, unique_solvents))

        rows = []
        for ligand, base, solvent in combos:
            x_raw = np.concatenate([
                drfp,
                rp._get_desc(ligand_cache, ligand),
                rp._get_desc(base_cache,   base),
                rp._get_desc(solvent_cache, solvent),
            ])
            rows.append(x_raw[keep_mask])
        X = np.vstack(rows)
        X_ready = scaler.transform(X)

        y_pred_pct = xgb.predict(X_ready) * 100.0
        top5_yields = np.sort(y_pred_pct)[::-1][:5]

        for yld in top5_yields:
            self.assertTrue(
                0.0 <= yld <= 120.0,
                msg=f"Yield {yld:.1f}% is outside the expected [0, 120] range",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
