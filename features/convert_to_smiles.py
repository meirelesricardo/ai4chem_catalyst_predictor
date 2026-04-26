import pandas as pd

INPUT_CSV = "data_suzuki_cleaned.csv"
OUTPUT_CSV = "data_suzuki_smiles_only.csv"

reactant_1_smiles = {
    '6-chloroquinoline': 'Clc1ccc2ncccc2c1',
    '6-Bromoquinoline': 'Brc1ccc2ncccc2c1',
    '6-triflatequinoline': 'FC(F)(F)S(=O)(=O)Oc1ccc2ncccc2c1',
    '6-Iodoquinoline': 'Ic1ccc2ncccc2c1',
    '6-quinoline-boronic acid hydrochloride': '[Cl-].OB(O)c1ccc2[nH+]cccc2c1',
    'Potassium quinoline-6-trifluoroborate': '[K+].F[B-](F)(F)c1ccc2ncccc2c1',
    '6-Quinolineboronic acid pinacol ester': 'CC1(C)OB(c2cc3cccnc3cc2)OC1(C)C'
}

reactant_2_smiles = {
    '2a, Boronic Acid': 'Cc1ccc2c(cnn2C2CCCCO2)c1B(O)O',
    '2b, Boronic Ester': 'Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1',
    '2c, Trifluoroborate': 'Cc1ccc2c(cnn2C2CCCCO2)c1[B-](F)(F)F.[K+]',
    '2d, Bromide': 'Cc1ccc2c(cnn2C2CCCCO2)c1Br'
}

df = pd.read_csv(INPUT_CSV)

df_out = pd.DataFrame({
    "Smiles 1": df["reactant_1"].map(reactant_1_smiles),
    "Smiles 2": df["reactant_2"].map(reactant_2_smiles),
})

df_out.to_csv(OUTPUT_CSV, index=False)
print(df_out.head())