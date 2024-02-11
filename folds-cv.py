import pickle
from main import create_model_data
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedKFold
from evaluation import classification_report
from models import Model



"""
This script runs 10-folds cross validation over all combinations of features (610) with a given model and returns
the result in form of a csv (1 row: 1 feature set). 
"""


def get_combinations(lst):  # for creating combinations of feature selection
    combinations = []
    for r in range(1, len(lst) + 1):
        combinations.extend(list(itertools.combinations(lst, r)))
    return combinations


# arguments
# ----------------------------------------------------------------------------------------------------------------------
folds = 10
eval_mcc = True
eval_counts = True
model = None
# Select -> random forest : "rf", support vector machine : "svm", logistic regression : "linear", Neural net : "ann"
model_name = "ann"
# ----------------------------------------------------------------------------------------------------------------------

with open(f"processed_dataset/features_amp.pkl", "rb") as f:
    df = pickle.load(f)

if __name__ == "__main__":
    r_df = pd.DataFrame()
    for includes in get_combinations(["conform", "no_reduction", "hydro"]):
        for types in get_combinations(["counts", "mat", "tfeat", "protvec"]):
            for offsets in get_combinations([1, 2, 3]):  # [1,2,3]
                print(includes, types, offsets)
                if types in get_combinations(["counts", "protvec"]) and offsets != tuple(
                        [1]):  # redundant combinations
                    break
                data, y_data = create_model_data(df, include=includes, type=types, offsets=offsets,
                                                 is_train=None)
                kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
                evaluation = {}  # a dict to maintain results
                for train_index_vals, test_index_vals in kf.split(data, y_data):
                    train_X = data[train_index_vals]
                    train_y = y_data.iloc[train_index_vals]
                    test_X = data[test_index_vals]
                    test_y = y_data.iloc[test_index_vals]
                    feature_len = train_X.shape[-1]
                    if model_name == "ann":
                        model = Model(model_name, input_dim=train_X.shape[1])
                    else:
                        model = Model(model_name)
                    model.fit(train_X, train_y)
                    pred = model.predict(test_X)
                    prob = model.predict_proba(test_X)
                    # Calculate metrics
                    # Generate classification report
                    report = classification_report(test_y, pred, prob)
                    for key in report:
                        evaluation[key] = evaluation.get(key, 0) + report[key]

                for key in evaluation:
                    evaluation[key] = evaluation.get(key) / folds
                print(evaluation)

                new_entry = {"model": model_name, "string_conversions": includes, "feat_types": types,
                             "dataset": f"{folds}-folds cv", "offsets": offsets, "nfeat": feature_len}
                new_entry.update(evaluation)

                r_df = r_df._append(new_entry, ignore_index=True)

    r_df = r_df.sort_values(['f1_score', 'nfeat'], ascending=[False, True])

    r_df.to_csv(f"results/results_cv_stratified{folds}_folds_{model_name}_amp.csv", index=False)
