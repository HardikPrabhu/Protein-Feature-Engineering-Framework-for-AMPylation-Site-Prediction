import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
import itertools
from sklearn.svm import SVC
from main import create_model_data


def get_combinations(lst):
    combinations = []
    for r in range(1, len(lst) + 1):
        combinations.extend(list(itertools.combinations(lst, r)))
    return combinations


# arguments : ------------------
raw_name = "amp"
eval_mcc = True
eval_counts = True
model_name = "rf"

with open(f"processed_dataset/features_{raw_name}.pkl", "rb") as f:
    df = pickle.load(f)

if __name__ == "__main__":
    r_df = pd.DataFrame()

    for includes in get_combinations(["conform", "no_reduction", "hydro"]):
        for types in get_combinations(["counts", "mat", "tfeat", "t2feat", "protvec"]):
            if "t2feat" in types and "tfeat" not in types:
                pass
            else:
                for offsets in get_combinations([1, 2, 3]):  # [1,2,3]

                    train = create_model_data(df, include=includes, type=types, offsets=offsets, is_train=True)
                    test = create_model_data(df, include=includes, type=types, offsets=offsets, is_train=False)

                    feature_len = train[0].shape[-1]

                    if model_name == "rf":
                        model = RandomForestClassifier(random_state=42)

                    else:
                        model = SVC(random_state=42)

                    start_time = time.time()

                    model.fit(train[0], train[1])
                    pred = model.predict(test[0])
                    # Calculate metrics
                    # Generate classification report
                    report = classification_report(test[1], pred, output_dict=True)
                    classes = list(df["label"].unique())
                    eval = dict()
                    for i in classes:
                        eval[f"sn_{i}"] = report[str(i)]["recall"]
                    eval[f"OA"] = report["accuracy"]
                    eval[f"AA"] = report["macro avg"]["recall"]
                    # mcc
                    # f1
                    if eval_counts:
                        eval["TP"], eval["FP"], eval["TN"], eval["FN"] = calculate_counts(test[1], pred)
                    if eval_mcc:
                        eval["mcc"] = matthews_corrcoef(test[1], pred)

                    end_time = time.time()
                    execution_time = end_time - start_time

                    new_entry = {"model": model_name, "string_conversions": includes, "feat_types": types,
                                 "dataset": "test", "offsets": offsets, "nfeat": feature_len,
                                 "time_taken": execution_time, }
                    new_entry.update(eval)

                    r_df = r_df.append(new_entry, ignore_index=True)

                    if types == ("counts"):
                        break

    if eval_mcc:
        r_df = r_df.sort_values(['mcc', 'nfeat'], ascending=[False, True])
    else:
        r_df = r_df.sort_values(['AA', 'nfeat'], ascending=[False, True])

    r_df.to_csv(f"results/results_{model_name}_{raw_name}_2.csv", index=False)
