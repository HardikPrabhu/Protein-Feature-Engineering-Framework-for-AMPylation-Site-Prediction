from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
import itertools
from sklearn.metrics import f1_score
import pickle
from main import create_model_data
import numpy as np
import shap
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def get_combinations(lst):
    combinations = []
    for r in range(1, len(lst) + 1):
        combinations.extend(list(itertools.combinations(lst, r)))
    return combinations



# arguments : ------------------
raw_name = "amp"
eval_mcc = True
model_name = "rf"



with open(f"processed_dataset/features_{raw_name}.pkl", "rb") as f:
    df = pickle.load(f)

df = df.rename(columns={'labels': 'label'})

if raw_name == "ion channel vs non-ion channel":
  class_to_int = {c:i for i,c in enumerate(df["label"].unique())}
  df["label"] = df['label'].apply(lambda row: class_to_int[row])


names = {}
names["no_reduction"] = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
names["hydro"] = ['H_0','H_1','H_2','H_3','H_4']
names["conform"] = ['C_1','C_2','C_3','C_4','C_5','C_6','C_7']

includes = ["conform", "no_reduction", "hydro"]
types = ["counts", "mat"]
offsets = [1, 3]
feature_names = []


for i in includes:
        for t in types:
             if t in ["counts"]:
                    feature_names.extend(names[i])

             else:
                    for o in offsets:
                          # create a matrix using names[i]
                          # Convert the list of characters to a 2D numpy array with concatenated strings
                          matrix = np.array([[ch1 + ch2 + f"({o})" for ch2 in names[i]] for ch1 in names[i]])
                          # Flatten the 2D array to get a 1D array
                          flattened_array = matrix.ravel()
                          feature_names.extend(list(flattened_array))


if __name__ == "__main__":

     train = create_model_data(df,include=includes,type=types,offsets=offsets,is_train=True)
     test = create_model_data(df,include=includes,type=types,offsets=offsets,is_train=False)
     feature_len = train[0].shape[-1]
     model = RandomForestClassifier(random_state=42)
     model.fit(train[0],train[1])
     pred = model.predict(test[0])
            # Calculate metrics
            # Generate classification report
     report = classification_report(test[1], pred, output_dict=True)
     classes =  list(df["label"].unique())
     eval = dict()
     for i in classes:
            eval[f"sn_{i}"] = report[str(i)]["recall"]
            eval[f"OA"] = report["accuracy"]
            eval[f"AA"] = report["macro avg"]["recall"]
            # mcc
    # f1
     eval["F1"] = f1_score(test[1], pred)

     if eval_mcc:
                eval["mcc"] = matthews_corrcoef(test[1], pred)

     print(eval)


     all_data = np.vstack((train[0],test[0]))

     # Create an explainer object using the trained model and the feature values
     explainer = shap.TreeExplainer(model,data=train[0],model_output="probability",feature_names=feature_names)



     shap_values = explainer(all_data, check_additivity=False)


     """
     # barplot
     shap.plots.bar(shap_values[:,:,1], show=False, max_display=25)
     plt.savefig(f"shap_stuff/shap_bar.png", bbox_inches='tight', dpi=300)
     """

     """
     # scatterplot
     selected =  ["H_2","K","T","H_2H_2(1)"]  #names["hydro"]
     for i in range(len(feature_names)):
        if feature_names[i] in selected:
         fig = plt.figure()
         shap.plots.scatter(shap_values[:,:,1][:, feature_names[i]], show=False)
         plt.savefig(f"shap_stuff/selected_{feature_names[i]}.png", bbox_inches='tight', dpi=300)
         plt.close()
     """

     """
     data_point_index = []

     # get data_points
     for i in range(len(all_data)):
         if model.predict_proba(all_data[i].reshape(1, -1))[0][1] >0.93 or model.predict_proba(all_data[i].reshape(1, -1))[0][1]<0.07:
             print(model.predict_proba(all_data[i].reshape(1, -1))[0][1],i)
             data_point_index.append(i)

     for i in data_point_index:
         fig = plt.figure()
         # Create the waterfall plot for the specified data point
         shap.plots.waterfall(shap_values[:,:,1][i], show=False)
         plt.savefig(f"shap_stuff/local_{i}.png", bbox_inches='tight', dpi=300)
         plt.close()

         # Display the plot
         plt.show()
     """

     # Calculate SHAP values
     mean_abs_shap = np.mean(np.abs(shap_values.values[:,:,1]), axis=0)
     top_n_indices = np.argsort(mean_abs_shap)[::-1]


     # sort  features
     top_n_feature_names = [feature_names[i] for i in top_n_indices]
     print("topn : ", top_n_feature_names)





     f1 = []
     mcc = []
     f_used = []
     s = [x for x in range(1,15)]
     s.extend([980//32,980//16,980//4,980//2,980])
     folds = 10
     kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

     data, y_data = create_model_data(df, include=includes, type=types, offsets=offsets)

     for i in s:
         if i == 980:
             selected_columns = data
         else:
             features = top_n_indices[:i]
             selected_columns = data[:, features]
         f_len = len(features)
         f_used.append(f_len)
         f1_ = 0
         mcc_ = 0
         for train_index_vals, test_index_vals in kf.split(data, y_data):

             train_X = selected_columns[train_index_vals]
             train_y = y_data.iloc[train_index_vals]
             test_X = selected_columns[test_index_vals]
             test_y = y_data.iloc[test_index_vals]
             rf = RandomForestClassifier(random_state=42)
             rf.fit(train_X,train_y)
             y_pred = rf.predict(test_X)
             f1_ +=f1_score(test_y, y_pred)
             mcc_ +=matthews_corrcoef(test_y, y_pred)

         f1.append(f1_/folds)
         print(f1)
         mcc.append(mcc_/folds)


fig = plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(f_used)),mcc,"bo-",label ="MCC")
plt.plot(np.arange(len(f_used)),f1,"rx-",label = "F1")

plt.xlabel("Number of features selected")
plt.ylabel("F1, MCC")
plt.xticks(np.arange(len(f_used)), f_used)
plt.legend()
plt.savefig(f'shap_stuff/_feat_select.png',dpi=300,bbox_inches='tight')






