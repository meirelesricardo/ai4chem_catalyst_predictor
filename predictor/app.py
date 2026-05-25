"""
How to run the predictor on terminal:

cd predictor
streamlit run app.py


This script implements a Streamlit web application that uses the XGBoost model 
trained on a Suzuki coupling dataset to predict the best catalytic conditions for 
a given pair of reactants. The application allows users to select two reactants 
from dropdown menus, visualizes their molecular structures, and then predicts 
and displays the top 5 ligand–base–solvent combinations ranked by predicted yield. 
The app uses DRFP fingerprints for reaction encoding and RDKit molecular descriptors for
capturing ligand/base/solvent properties.
"""

# Import libraries
import itertools
import math
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smiles_data import BASE_SMILES, LIGAND_SMILES, REACTANT_1_SMILES, REACTANT_2_SMILES, SOLVENT_SMILES

# Defining paths to data and model (XGBoost and scaler)
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR      = os.path.dirname(_BASE_DIR)
_CSV_PATH      = os.path.join(_ROOT_DIR, "data", "processed", "suzuki_cleaned.csv")
_XGB_PATH      = os.path.join(_ROOT_DIR, "models", "XGBoost.pth")
_SCALER_PATH   = os.path.join(_ROOT_DIR, "data", "final", "scaler.pth")
_KEEP_MASK_PATH = os.path.join(_ROOT_DIR, "data", "final", "keep_mask.npy")

# RDKit descriptor setup
_DESC_NAMES = [d[0] for d in Descriptors.descList]
_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
_N_DESC     = len(_DESC_NAMES)
_ZERO_DESC  = np.zeros(_N_DESC)

# Streamlit page configuration
st.set_page_config(
    page_title="Suzuki Coupling - Catalytic Yield Predictor",
    page_icon="⚗️",
    layout="centered",
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main card */
    section.main > div { padding-top: 2rem; }

    /* Title */
    h1 { letter-spacing: -0.5px; }

    /* Selectbox labels */
    label { font-size: 0.85rem !important; color: #9a9aaa !important; }

    /* Primary button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4a9eff 0%, #2563eb 100%);
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.65rem 1.2rem;
        color: #ffffff;
        transition: opacity 0.2s;
    }
    div.stButton > button[kind="primary"]:hover { opacity: 0.88; }

    /* Result card */
    .result-header {
        background: #2c2c2e;
        border-left: 4px solid #4a9eff;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
    }
    .result-header p { margin: 0; color: #9a9aaa; font-size: 0.85rem; }
    .result-header span { color: #e0e0e0; font-weight: 600; }

    /* Divider */
    hr { border-color: #3a3a3c; }
</style>
""", unsafe_allow_html=True)


def _mol_svg(smi: str, width: int = 300, height: int = 200) -> str:
    """
    This function converts a SMILES string into an image of the molecule 
    using RDKit's drawing capabilities.

    Input: 
        smi: a SMILES string representing a molecule
        width: the width of the output SVG image
        height: the height of the output SVG image
    Output: 
        an SVG string representing the drawn molecule
    """
    parts = smi.split(".")
    best  = max(parts, key=lambda s: (Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C")).GetNumAtoms())
    mol   = Chem.MolFromSmiles(best)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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
    This function retrieves a descriptor vector from a cache based on a given key.
    If the key is None or NaN, it returns a zero vector.
    Input:
        cache: a dictionary mapping keys to descriptor vectors
        key: the key to look up in the cache
    Output:
        a numpy array of molecular descriptors for the given key, or a zero vector if 
        the key is invalid
    """
    if key is None or (isinstance(key, float) and math.isnan(key)):
        return _ZERO_DESC
    return cache.get(key, _ZERO_DESC)


# Streamlit caching for loading model and computing descriptors
@st.cache_resource(show_spinner="Loading model and computing molecular descriptors…")
def _load_resources():
    """
    This function loads the XGBoost model, scaler, keep_mask, and pre-computed descriptor caches for ligands, bases, and solvents.
    Output:
        xgb: the loaded XGBoost model for prediction
        scaler: the loaded scaler for feature normalization
        keep_mask: the loaded boolean mask for selecting features
        ligand_cache: a dictionary mapping ligand names to their descriptor vectors
        base_cache: a dictionary mapping base names to their descriptor vectors
        solvent_cache: a dictionary mapping solvent names to their descriptor vectors
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        xgb = joblib.load(_XGB_PATH)
    scaler    = torch.load(_SCALER_PATH, weights_only=False)
    keep_mask = np.load(_KEEP_MASK_PATH)
    ligand_cache  = {k: _smiles_to_desc(v) for k, v in LIGAND_SMILES.items()}
    base_cache    = {k: _smiles_to_desc(v) for k, v in BASE_SMILES.items()}
    solvent_cache = {k: _smiles_to_desc(v) for k, v in SOLVENT_SMILES.items()}
    return xgb, scaler, keep_mask, ligand_cache, base_cache, solvent_cache


@st.cache_data
def _load_options():
    """
    This function loads the options for reactants, ligands, bases, and solvents
    from the CSV file and returns them along with all possible combinations.

    Output:
        r1_options: a list of unique options for the first reactant
        r2_options: a list of unique options for the second reactant
        combos: a list of all possible combinations of ligand, base, and solvent
    """
    df = pd.read_csv(_CSV_PATH)
    unique_ligands  = [None] + sorted(l for l in df["ligand"].dropna().unique())
    unique_bases    = [None] + sorted(b for b in df["base"].dropna().unique())
    unique_solvents = sorted(df["solvent"].unique().tolist())
    combos          = list(itertools.product(unique_ligands, unique_bases, unique_solvents))
    r1_options      = sorted(df["reactant_1"].unique().tolist())
    r2_options      = sorted(df["reactant_2"].unique().tolist())
    return r1_options, r2_options, combos


# Importing the prediction function from run_predictor.py
def _predict(r1, r2, xgb, scaler, keep_mask, ligand_cache, base_cache, solvent_cache, combos):
    """
    This function takes the selected reactants and pre-loaded resources to predict
    the top 5 catalytic conditions

    Input:
        r1: the first reactant selected by the user
        r2: the second reactant selected by the user
        xgb: the pre-loaded XGBoost model for prediction
        scaler: the pre-loaded scaler for feature normalization
        keep_mask: the pre-loaded boolean mask for selecting features
        ligand_cache: a dictionary mapping ligand names to their descriptor vectors
        base_cache: a dictionary mapping base names to their descriptor vectors
        solvent_cache: a dictionary mapping solvent names to their descriptor vectors
        combos: a list of all possible combinations of ligand, base, and solvent
    Output:
        a pandas DataFrame containing the top 5 predicted catalytic conditions and their predicted yields
    """
    rxn_smi = REACTANT_1_SMILES[r1] + "." + REACTANT_2_SMILES[r2] + ">>"
    drfp = np.array(
        DrfpEncoder.encode([rxn_smi], n_folded_length=2048, radius=3, min_radius=0, rings=True)[0],
        dtype=float,
    )
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
    X_ready = scaler.transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred_pct = xgb.predict(X_ready) * 100.0
    top5_idx = np.argsort(y_pred_pct)[::-1][:5]
    records = []
    for rank, idx in enumerate(top5_idx, 1):
        ligand, base, solvent = combos[idx]
        records.append({
            "Rank":    rank,
            "Ligand":  ligand if ligand is not None else "—",
            "Base":    base   if base   is not None else "—",
            "Solvent": solvent,
            "Yield (%)": round(float(y_pred_pct[idx]), 1),
        })
    return pd.DataFrame(records)


# Load model, scaler, keep_mask, and descriptor caches once at the start
xgb, scaler, keep_mask, ligand_cache, base_cache, solvent_cache = _load_resources()
r1_options, r2_options, combos = _load_options()

# Streamlit UI
st.title("⚗️ Suzuki Coupling - Catalytic Yield Predictor")
st.markdown(
    "This tool uses a **XGBoost model** trained on a Suzuki cross-coupling dataset to predict "
    "the best catalytic conditions for a given pair of reactants (present in the dataset). "
    "Reactions are encoded with **DRFP fingerprints** and ligand/base/solvent properties are "
    "captured via **RDKit molecular descriptors**. "
    "Select your two reactants below and the model will rank all ligand–base–solvent combinations "
    "by predicted yield."
)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    r1 = st.selectbox("FIRST REACTANT", r1_options)
with col2:
    r2 = st.selectbox("SECOND REACTANT", r2_options)

# Display molecular structures for the selected reactants
with st.expander("Show molecular structures", expanded=True):
    v1, arrow_col, v2 = st.columns([5, 1, 5])
    svg_style = (
        "background:#f5f5f7; border-radius:8px; padding:6px; "
        "display:flex; justify-content:center; align-items:center;"
    )
    with v1:
        st.markdown(
            f'<p style="color:#9a9aaa; font-size:0.78rem; margin:0 0 4px 0; text-align:center;">Reactant 1</p>'
            f'<div style="{svg_style}">{_mol_svg(REACTANT_1_SMILES[r1])}</div>',
            unsafe_allow_html=True,
        )
    with arrow_col:
        st.markdown(
            '<div style="height:100%; display:flex; align-items:center; justify-content:center; '
            'font-size:1.8rem; color:#4a9eff; padding-top:1.4rem;">+</div>',
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            f'<p style="color:#9a9aaa; font-size:0.78rem; margin:0 0 4px 0; text-align:center;">Reactant 2</p>'
            f'<div style="{svg_style}">{_mol_svg(REACTANT_2_SMILES[r2])}</div>',
            unsafe_allow_html=True,
        )

st.markdown("")
run = st.button("🔍 &nbsp; Predict top 5 conditions", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Testing {len(combos):,} combinations…"):
        results_df = _predict(r1, r2, xgb, scaler, keep_mask, ligand_cache, base_cache, solvent_cache, combos)

    st.markdown("---")
    st.markdown(
        f'<div class="result-header">'
        f'<p>Reactant 1 &nbsp;→&nbsp; <span>{r1}</span></p>'
        f'<p>Reactant 2 &nbsp;→&nbsp; <span>{r2}</span></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Ligand":  st.column_config.TextColumn("Ligand"),
            "Base":    st.column_config.TextColumn("Base"),
            "Solvent": st.column_config.TextColumn("Solvent"),
            "Yield (%)": st.column_config.ProgressColumn(
                "Predicted Yield",
                format="%.1f %%",
                min_value=0,
                max_value=100,
            ),
        },
    )