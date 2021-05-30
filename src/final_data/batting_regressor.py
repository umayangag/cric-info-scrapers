import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import preprocessing

RF = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=False, max_depth=100,
                            class_weight={0: 4, 1: 1, 2: 2})
gnb = GaussianNB()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1)
SVM = svm.SVC(kernel='linear', C=1)
regr = RandomForestRegressor(max_depth=4, random_state=0)
reg = LinearRegression()
mltreg = MultiOutputRegressor(regr)
predictor = mltreg

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\batting_encoded.csv")


def batting_predict():
    input_data = pd.read_csv(dataset_source)

    # scaler = preprocessing.StandardScaler().fit(input_data)
    # data_scaled = scaler.transform(input_data)
    # final_df = pd.DataFrame(data=data_scaled, columns=input_data.columns)
    # final_df["runs"] = input_data["runs"]
    # input_data = final_df

    X = input_data[[
        "batting_position",
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
        "batting_session",
        "toss",
        "venue",
        "opposition",
        "season",
    ]]
    y = input_data[["runs", "balls"]]  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)

    comparison = {}
    comparison["actual"] = y_test.to_numpy()
    comparison["predicted"] = y_pred

    for i in range(0, len(y_pred)):
        print(comparison["actual"][i], " ", comparison["predicted"][i], "")
    print("Accuracy:", metrics.mean_absolute_error(y_test, y_pred))
    # print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=10).max())


batting_predict()
