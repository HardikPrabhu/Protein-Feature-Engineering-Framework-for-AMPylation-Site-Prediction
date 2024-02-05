# Protein-Feature-Engineering-Framework-for-AMPylation-Site-Prediction

## Dataset 

The data that support this study is available at https://github.com/MehediAzim/DeepAmp. It has been converted in csv format and present in the dataset folder.

Dataset also includes protein sequence embeddings for i. regular protein alphabet sequences (ProtVec) ii. reduced protein alphabet sequences (Ra2vec).

## Feature engineering

![img.png](img.png)

All the feature engineering methods could be found in "main.py".

## Quick setup

1. Install the required python packages : `pip install -r requirement.txt`
2. Run the script "extract_feat.py". It will create a pickle file "features_amp.pkl" 
3. Run "folds-cv.py" to perform 10 folds cross validation as given in the manuscript. It will return a csv file in results directory containing several metrics used for evaluation.

## How results are stored
The experiments are done for all the combinations of features possible. (For one model there are 610 possible feature engineering combinations)

The results for a given model are stored the following way:

| model | string_conversions                       | feat_types                       | dataset     | offsets | nfeat | accuracy | precision | recall  | f1_score | auc_roc | mcc    |
|-------|------------------------------------------|-----------------------------------|-------------|---------|-------|----------|-----------|---------|----------|---------|--------|
| rf    | ('conform', 'no_reduction', 'hydro')    | ('counts', 'mat')                | 10-folds cv | (1, 3)  | 980   | 0.809744 | 0.829522  | 0.606044| 0.687253 | 0.885388| 0.580302|
| rf    | ('no_reduction',)                        | ('counts', 'tfeat', 'protvec')   | 10-folds cv | (1,)    | 75    | 0.791110 | 0.746792  | 0.639560| 0.681438 | 0.851257| 0.537578|
| rf    | ('no_reduction', 'hydro')                | ('counts', 'mat')                | 10-folds cv | (1,)    | 450   | 0.793670 | 0.763735  | 0.619231| 0.677726 | 0.867166| 0.540653|
| rf    | ('no_reduction',)                        | ('counts', 'tfeat', 'protvec')   | 10-folds cv | (1, 2)  | 80    | 0.780370 | 0.718014  | 0.648352| 0.674373 | 0.853790| 0.517564|
| rf    | ('no_reduction',)                        | ('mat', 'protvec')               | 10-folds cv | (1, 3)  | 850   | 0.799147 | 0.812106  | 0.589560| 0.673608 | 0.879957| 0.556312|