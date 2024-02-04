import pandas as pd

import pickle
from main import create_vector, create_co_matrix, create_text_features, create_counts

# alpha reduction mappings
hydropathy_reduction = {"RNDEQHK": 0,
                        "PWY": 1,
                        "GST": 2,
                        "ACM": 3,
                        "ILFV": 4}

conformational_reduction = {"REQKACML": 0, "HWYF": 1, "TIV": 2, "ND": 3,
                            "P": 4, "S": 5, "G": 6}

# import the dataset
df = pd.read_csv(f"dataset/amp.csv")

# 1. seq to vector

# 1.1 without reduction
stoi = {s: i for (i, s) in enumerate(sorted(set("".join(list(df["seq"])))))}
df["no_reduction"] = df['seq'].apply(lambda row: create_vector(row, stoi))
# 1.2 with reduction
df['hydro_reduction'] = df['seq'].apply(lambda row: create_vector(row, hydropathy_reduction))
df['conform_reduction'] = df['seq'].apply(lambda row: create_vector(row, conformational_reduction))

# 2. creating counts vector

df["no_reduction_counts"] = df['no_reduction'].apply(lambda row: create_counts(row, len(stoi)))
df["hydro_counts"] = df['hydro_reduction'].apply(lambda row: create_counts(row, len(hydropathy_reduction)))
df["conform_counts"] = df['conform_reduction'].apply(lambda row: create_counts(row, len(conformational_reduction)))

# 3. create co-occurance matrices and texture feature

offsets = [1, 2, 3]

for off in offsets:
    df[f"no_reduction_mat_{off}"] = df['no_reduction'].apply(lambda row: create_co_matrix(row, len(stoi), offset=off))
    df[f'hydro_mat_{off}'] = df['hydro_reduction'].apply(
        lambda row: create_co_matrix(row, len(hydropathy_reduction), offset=off))
    df[f'conform_mat_{off}'] = df['conform_reduction'].apply(
        lambda row: create_co_matrix(row, len(conformational_reduction), offset=off))

    df[f'no_reduction_tfeat_{off}'] = df[f"no_reduction_mat_{off}"].apply(lambda row: create_text_features(row))
    df[f'hydro_tfeat_{off}'] = df[f'hydro_mat_{off}'].apply(lambda row: create_text_features(row))
    df[f'conform_tfeat_{off}'] = df[f'conform_mat_{off}'].apply(lambda row: create_text_features(row))

    df[f'no_reduction_t2feat_{off}'] = df[f"no_reduction"].apply(
        lambda row: create_text_features(create_co_matrix(row, len(stoi), offset=off, bigram=True)))
    df[f'hydro_t2feat_{off}'] = df["hydro_reduction"].apply(
        lambda row: create_text_features(create_co_matrix(row, len(hydropathy_reduction), offset=off, bigram=True)))
    df[f'conform_t2feat_{off}'] = df["conform_reduction"].apply(
        lambda row: create_text_features(create_co_matrix(row, len(conformational_reduction), offset=off, bigram=True)))

# 4. Pro2vec
import numpy as np


def condense_embeddings_map(dataset):
    # Concatenate all columns into a single column
    d = dataset.drop(columns=["Seq", "label", "is_Train"])
    d = d.apply(lambda row: np.array(row.values), axis=1)
    map = {}
    for i in range(len(d)):
        map[dataset.loc[i, "Seq"]] = d[i]

    return map


vect_df = pd.read_csv(f"dataset/AMP_protvec.csv")
protvecmap = condense_embeddings_map(vect_df)
df['no_reduction_protvec'] = df['seq'].map(protvecmap)

vect_df = pd.read_csv(f"dataset/AMP_Hydropathy.csv")
protvecmap = condense_embeddings_map(vect_df)
df['hydro_protvec'] = df['seq'].map(protvecmap)

vect_df = pd.read_csv(f"dataset/AMP_Conf_sim.csv")
protvecmap = condense_embeddings_map(vect_df)
df['conform_protvec'] = df['seq'].map(protvecmap)

with open(f"processed_dataset/features_amp.pkl", "wb") as f:
    pickle.dump(df, f)
