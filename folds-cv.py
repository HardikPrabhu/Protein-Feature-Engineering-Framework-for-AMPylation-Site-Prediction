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


def calculate_counts(y_actual, y_predict):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for actual, predict in zip(y_actual, y_predict):
        if actual == 1 and predict == 1:
            tp += 1
        elif actual == 0 and predict == 1:
            fp += 1
        elif actual == 0 and predict == 0:
            tn += 1
        elif actual == 1 and predict == 0:
            fn += 1

    return tp, fp, tn, fn


def get_combinations(lst):
    combinations = []
    for r in range(1, len(lst) + 1):
        combinations.extend(list(itertools.combinations(lst, r)))
    return combinations


folds = 10

# arguments : ------------------
eval_mcc = True
eval_counts = True
model_name = "rf"  # name used to store the results
model = RandomForestClassifier(random_state=42)


with open(f"processed_dataset/features_amp.pkl", "rb") as f:
    df = pickle.load(f)

if __name__ == "__main__":
    r_df = pd.DataFrame()
    for includes in get_combinations(["conform", "no_reduction", "hydro"]):
        for types in get_combinations(["counts", "mat", "tfeat", "protvec"]):
          #if len(r_df) < 3:
            for offsets in get_combinations([1, 2, 3]):  # [1,2,3]
                print(includes, types, offsets)

                data, y_data = create_model_data(df, include=includes, type=types, offsets=offsets,
                                                 is_train=None)
                kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

                eval = dict()
                classes = list(df["label"].unique())
                for i in classes:
                    eval[f"sn_{i}"] = 0
                eval[f"OA"] = 0
                eval[f"AA"] = 0
                if eval_mcc:
                    eval["mcc"] = 0
                eval["F1"] = 0

                for train_index_vals, test_index_vals in kf.split(data, y_data):

                    train_X = data[train_index_vals]
                    train_y = y_data.iloc[train_index_vals]
                    test_X = data[test_index_vals]
                    test_y = y_data.iloc[test_index_vals]

                    feature_len = train_X.shape[-1]

                    start_time = time.time()
                    model.fit(train_X, train_y)
                    pred = model.predict(test_X)
                    # Calculate metrics
                    # Generate classification report
                    report = classification_report(test_y, pred, output_dict=True)
                    classes = list(df["label"].unique())

                    for i in classes:
                        eval[f"sn_{i}"] += report[str(i)]["recall"]
                    eval[f"OA"] += report["accuracy"]
                    eval[f"AA"] += report["macro avg"]["recall"]
                    if eval_mcc:
                        eval["mcc"] += matthews_corrcoef(test_y, pred)
                    eval["F1"] += f1_score(test_y, pred)

                for key in eval:
                    eval[key] /= folds

                print(eval)

                new_entry = {"model": model_name, "string_conversions": includes, "feat_types": types,
                             "dataset": " 10-folds cv", "offsets": offsets, "nfeat": feature_len}
                new_entry.update(eval)

                r_df = r_df._append(new_entry, ignore_index=True)

    r_df = r_df.sort_values(['F1', 'nfeat'], ascending=[False, True])

    r_df.to_csv(f"results/results_cv_stratified{folds}_folds{model_name}_amp_testing.csv", index=False)
