import pickle
import pandas as pd
from main import create_model_data
from models import Model
from evaluation import classification_report


# script to perform experiment on models from table 1. for 80:20 train-test split.

configs = {"rf": {"includes": ["conform", "no_reduction", "hydro"], "types": ["counts", "mat"], "offsets": [1, 3]},
           "ann": {"includes": ["conform", "no_reduction"], "types": ["mat"], "offsets": [1, 3]},
           "xgb": {"includes": ["hydro", "no_reduction"], "types": ["counts","mat"], "offsets": [1, 3]},
           "lgbm": {"includes": ["conform", "no_reduction"], "types": ["mat"], "offsets": [1, 2,3]},
           "svm": {"includes": ["no_reduction"], "types": ["counts","mat"], "offsets": [1, 2]},
           "linear": {"includes": ["hydro", "no_reduction"], "types": ["tfeat","protvec"], "offsets": [3]}
           }

r_df = pd.DataFrame()

with open(f"processed_dataset/features_amp.pkl", "rb") as f:
    df = pickle.load(f)

for m_name in configs:
    param_dict = configs[m_name]
    train = create_model_data(df, include=param_dict["includes"], type=param_dict["types"],
                              offsets=param_dict["offsets"], is_train=True)
    test = create_model_data(df, include=param_dict["includes"], type=param_dict["types"],
                             offsets=param_dict["offsets"], is_train=False)
    feature_len = train[0].shape[-1]
    if m_name == "ann":
        model = Model(m_name, input_dim=feature_len)
    else:
        model = Model(m_name)
    model.fit(train[0], train[1])
    pred = model.predict(test[0])
    pred_prob = model.predict_proba(test[0])
    # Calculate metrics
    # Generate classification report
    report = classification_report(test[1], pred, pred_prob)
    new_entry = {"model": m_name, "string_conversions": param_dict["includes"], "feat_types": param_dict["types"],
                 "offsets": param_dict["offsets"], "nfeat": feature_len}
    new_entry.update(report)

    r_df = r_df._append(new_entry, ignore_index=True)


r_df.to_csv("results_train_test.csv")