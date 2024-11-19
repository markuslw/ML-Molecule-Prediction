import os
os.environ["LOKY_MAX_CPU_COUNT"] = "32"
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
    Prepare train dataset

    Use at least one method for data pre-processing

    Create a train and validation set from the train dataset.
    Drop lipophilicity from the X values.
"""
def preprocess_data():
    train_data_path = 'data/train.csv'
    train_data = pd.read_csv(train_data_path)
    train_data.dropna(inplace=True)
    train_data.dropna(axis=1, inplace=True)

    train_features = train_data.drop(columns=['lipophilicity', 'Id'])
    train_target = train_data['lipophilicity']

    train_X, val_X, train_y, val_y = train_test_split(train_features, train_target, random_state=42)
    
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)

    return train_X_scaled, val_X_scaled, train_y, val_y, scaler

"""
    Train dataset

    Analyse the data with at least 3 different classification methods
"""
def train_data(train_X, val_X, train_y, val_y):
    rfc_model = RandomForestClassifier(bootstrap=True, max_depth=8, min_samples_split=10,
                       n_estimators=150, random_state=42)  # Supervised
    rfc_model.fit(train_X, train_y)
    rfc_model_pred = rfc_model.predict(val_X)
    rfc_model_pred_accuracy = accuracy_score(val_y, rfc_model_pred)

    lr_model = LogisticRegression(C=0.025, max_iter=450, random_state=42)  # Supervised
    lr_model.fit(train_X, train_y)
    lr_model_pred = lr_model.predict(val_X)
    lr_model_pred_accuracy = accuracy_score(val_y, lr_model_pred)

    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=4, random_state=42)  # Supervised
    dt_model.fit(train_X, train_y)
    dt_model_pred = dt_model.predict(val_X)
    dt_mode_pred_accuracy = accuracy_score(val_y, dt_model_pred)

    models = {
        rfc_model: rfc_model_pred_accuracy,
        lr_model: lr_model_pred_accuracy,
        dt_model: dt_mode_pred_accuracy
    }

    best_model = max(models, key=models.get)
    models.pop(best_model)
    second_best_model = max(models, key=models.get)

    voting_model = VotingClassifier(
        estimators=[
            (best_model.__class__.__name__, best_model),
            (second_best_model.__class__.__name__, second_best_model)
        ],
        voting='soft'
    )
    voting_model.fit(train_X, train_y)

    voting_y_pred = voting_model.predict(val_X)

    return voting_model

"""
    Prepare test dataset

    Lipophilicity is the value we're trying to predict.
    Which is why it doesnt exists in this dataset.
"""
def test_data(model, scaler):
    test_data_path = 'data/test.csv'
    test_data = pd.read_csv(test_data_path)

    test_features = test_data.drop(columns=['Id'])
    test_X_scaled = scaler.transform(test_features)

    """
        Begin prediction on test set
    """
    y_pred = model.predict(test_X_scaled)

    """
        Save predictions to csv
    """
    best_model_submission = pd.DataFrame({
        'Id': test_data['Id'],
        'lipophilicity': y_pred
    })
    best_model_submission.to_csv(f'out/{model.__class__.__name__}_submission.csv', index=False, sep=',')

if __name__ == '__main__':
    train_X, val_X, train_y, val_y, scaler = preprocess_data()
    best_model = train_data(train_X, val_X, train_y, val_y)
    test_data(best_model, scaler)