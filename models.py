import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Model options
--------------
  Models that can be used, model_name to be used while constructing Model object
  - Random Forest : "rf"
  - support vector machine : "svm"
  - Logistic regression : "linear"
  - Neural net : "ann"
  - Xgboost : "xgb"
  - LightGBM : "lgbm"

"""


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, max(10, input_dim // 2))
        self.fc3 = nn.Linear(max(10, input_dim // 2), 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

    def fit(self, X_train, Y_train, epochs=100, learning_rate=0.01, batch_size=32):
        """
        Trains the neural network using the provided training data and labels.

        Parameters:
            X_train (Tensor): The training data features.
            Y_train (Tensor): The training data labels.
            epochs (int): The number of times to iterate over the entire dataset.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The size of each batch to be used during training.
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        dataset_size = X_train.shape[0]
        num_batches = (dataset_size + batch_size - 1) // batch_size  # Calculate the number of batches

        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                inputs = X_train[start:end]
                labels = Y_train[start:end]
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / num_batches}")

    def predict(self, X_batch):
        with torch.no_grad():
            return self(X_batch).round()

    def predict_proba(self, X_batch):
        with torch.no_grad():
            return self(X_batch)


class Model():
    """
    A wrapper class for machine learning models that simplifies the process of fitting,
    predicting, and obtaining prediction probabilities.
    Attributes:
        model_name (str): The name of the model to use.
        kwargs: key word arguments specific to the model.
    """

    def __init__(self, model_name="rf", **kwargs):
        self.model_name = model_name
        if self.model_name == "rf":
            self.model = RandomForestClassifier(**kwargs,random_state=42)
        if self.model_name == "svm":
            self.model = SVC(probability=True, **kwargs,random_state=42)
        if self.model_name == "linear":
            self.model = LogisticRegression(**kwargs,random_state=42)
        if self.model_name == "ann":
            self.model = MyNet(**kwargs)
            self.model.to(device)
        if self.model_name == "xgb":
            self.model = XGBClassifier(**kwargs,random_state=42)
        if self.model_name == "lgbm":
            self.model = LGBMClassifier(**kwargs,random_state=42)


    def fit(self, Xtrain, Ytrain, **kwargs):
        if self.model_name == "ann":
            Xtrain = torch.tensor(Xtrain,device=device).float()
            Ytrain = torch.tensor(np.array(Ytrain),device=device).float().view(-1, 1)
        self.model.fit(Xtrain, Ytrain)

    def predict(self, X):
        if self.model_name == "ann":
            X = torch.tensor(X,device=device).float()
            return self.model.predict(X).cpu()
        return self.model.predict(X)
    def predict_proba(self, Xtest):
        if self.model_name == "ann":
            Xtest = torch.tensor(Xtest,device=device).float()
            prob = self.model.predict_proba(Xtest).cpu()
        if self.model_name in ["rf", "svm", "linear","xgb","lgbm"]:
            prob = self.model.predict_proba(Xtest)
            prob = prob[:, 1]
        return prob
