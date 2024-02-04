"""Methods for Deriving features from the string of alphabets"""
import numpy as np


# 1. Feature reduction, string to vector
def create_vector(string, mappings):
    vector = []
    for char in string:
        entry = -1
        for key, value in mappings.items():
            if char in key:
                entry = int(value)
                break
        if entry == -1:
            print(char)
        vector.append(entry)
    return np.array(vector)


# 2. create normalized counts feature
def create_counts(vector, mapping_chars=5, normalize=True):
    counts = np.zeros((mapping_chars))
    for i in vector:
        counts[i] += 1
    if normalize == True:
        counts = counts / np.sum(counts)
    return counts


# 3. Create co-occurance mat with offsets
def create_co_matrix(vector, n=5, offset=1, normalize=True, bigram=False):
    # Initialize an empty co-occurrence matrix
    if bigram == True:
        co_matrix = np.zeros((n * n, n))
        a = 1
    else:
        co_matrix = np.zeros((n, n))
        a = 0
    k = offset
    # Iterate over the words using a sliding window of size 2
    for i in range(len(vector) - 1):
        if i + k + a < len(vector):
            if bigram:
                w1 = vector[i:i + 2]
                w2 = vector[i + k + 1]
                co_matrix[w1[0] * n + w1[1]][w2] += 1
            else:
                w1 = vector[i]
                w2 = vector[i + k]
                co_matrix[w1][w2] += 1

    # Convert the co-occurrence matrix to a numpy array
    co_matrix = co_matrix / np.sum(co_matrix)

    return co_matrix


# 3. Create texture features

def create_text_features(matrix):
    energy = np.sum(matrix ** 2)
    entropy = np.sum(-matrix * np.log(matrix + np.finfo(float).eps))
    contrast = 0
    homogen = 0
    diss = 0
    rows, cols = matrix.shape

    # Iterate through each element in the matrix
    for i in range(rows):
        for j in range(cols):
            element = matrix[i, j]
            homogen = homogen + element / (1 + (i - j) ** 2)
            contrast = contrast + element * ((i - j) ** 2)
            diss = diss + element * np.abs(i - j)
    return np.array([energy, entropy, homogen, contrast, diss])






# For creating the model input array after extracting features using the "extract_feat.py" script

def create_model_data(df, include=["no_reduction", "hydro", "conform"],
                      type=["counts", "mat", "tfeat", "protvec"], offsets=[1], is_train=None):
    """
    Concatenate all the features to have a model input arrays : X (datapoints as rows),Y (labels for classification)
    :param df: Processed dataset
    :param include: list containing representations to use
    :param type: extraction feature type
    :param offsets: list of offsets for creating co-occurrence matrices
    :param is_train: is it the training data (dataset has a column "is_Train"
    :return: numpy array of extracted features (1 data point per row) and classification labels
    """

    feature_names = []
    # if is_train = None, keep the entire dataset(do nothing)
    if is_train == True:
        df = df[df["is_Train"] == True]  # reduce the dataset to contain only train samples
    if is_train == False:
        df = df[df["is_Train"] == False]  # reduce the dataset to contain only test samples

    input = None
    column_names = df.columns
    for index, row in df.iterrows():
        row_input = np.array([])
        # extract features row wise
        for i in include:
            for t in type:
                if t in ["counts", "protvec"]:
                    row_input = np.hstack([row_input, row[f"{i}_{t}"].flatten()])
                else:
                    for o in offsets:
                        row_input = np.hstack([row_input, row[f"{i}_{t}_{o}"].flatten()])

        # stack row features
        if input is None:
            input = row_input
        else:
            input = np.vstack([input, row_input])

    return input, df["label"]
