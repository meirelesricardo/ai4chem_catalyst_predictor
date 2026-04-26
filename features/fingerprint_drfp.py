import pandas as pd

df = pd.read_csv("data_suzuki_smiles_only.csv")
# Colonnes : "Smiles 1"  (reactant_1)  |  "Smiles 2"  (reactant_2)
# Shape attendue : (5760, 2)
print(df.shape, df.head(2))


# DRFP encode des réactions, pas des molécules isolées.
# On formate chaque ligne en reaction SMILES.
# Pas de produit connu → on laisse la partie droite vide.

df["reaction_smiles"] = (
    df["Smiles 1"] + "." + df["Smiles 2"] + ">>"
)

from drfp import DrfpEncoder
import numpy as np

reaction_list = df["reaction_smiles"].tolist()

fps = DrfpEncoder.encode(
    reaction_list,
    n_folded_length=2048,   # taille du vecteur binaire (puissance de 2)
    radius=3,               # rayon max des sous-structures circulaires
    min_radius=0,           # inclut les atomes seuls (rayon 0)
    rings=True,             # inclut les cycles entiers comme sous-structures
    show_progress_bar=True,
)

fps_array = np.array(fps)  # shape : (5760, 2048)
print(fps_array.shape)     # → (5760, 2048)

# Vérifie qu'aucun fingerprint n'est entièrement nul (SMILES invalide)
zero_rows = np.where(fps_array.sum(axis=1) == 0)[0]
print(f"Lignes nulles : {len(zero_rows)}")   # attendu : 0

# Densité moyenne (% de bits à 1)
density = fps_array.mean()
print(f"Densité : {density:.4f}")            # typiquement 0.04–0.15

# Option A — numpy binaire (recommandé, plus rapide)
np.save("drfp_fingerprints.npy", fps_array)
# Rechargement : fps = np.load("drfp_fingerprints.npy")


# Note drfp il faudrait aussi des fingerprint pour les condition chimiques (ligands, base, solvent) 
# car juste avec les réactifs, bah y'a pas bcp de combinaisons possibles 