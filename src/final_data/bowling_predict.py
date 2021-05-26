import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

RF = RandomForestClassifier(n_estimators=100)
gnb = GaussianNB()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1)

predictor = RF

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\bowling_encoded.csv")


def bowling_predict():
    input_data = pd.read_csv(dataset_source)
    X = input_data[[
        "player_consistency",
        "player_form",
        "temp",
        "wind",
        "rain",
        "humidity",
        "cloud",
        "pressure",
        "viscosity",
        "inning",
        "bowling_session",
        "toss",
        "venue",
        "opposition",
        "season",
    ]]  # Features
    y = input_data["performance"]  # Labels

    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)

    # for classifiers
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))


bowling_predict()
