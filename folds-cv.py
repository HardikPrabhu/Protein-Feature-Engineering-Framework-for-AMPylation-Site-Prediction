import pickle
from main import create_model_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from evaluation import classification_report

def get_combinations(lst):
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
model_name = "rf"
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
                        start_time = time.time()
                        if model_name == "rf":
                            model = RandomForestClassifier(random_state=42)
                        model.fit(train_X, train_y)
                        pred = model.predict(test_X)
                        prob = model.predict_proba(test_X)
                        prob = prob[:, 1]
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

    r_df.to_csv(f"results/results_cv_stratified{folds}_folds{model_name}_amp_testing.csv", index=False)
