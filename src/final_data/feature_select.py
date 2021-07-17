import statsmodels.api as sm
import pandas as pd


def select_features(X_train,y_train):
    X1 = sm.add_constant(X_train)
    ols = sm.OLS(y_train, X1)
    lr = ols.fit()

    selected_features = list(X_train.columns)
    pmax = 1
    while len(selected_features) > 0:
        p = []
        X_1 = X_train[selected_features]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y_train, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=selected_features)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if pmax > 0.05:
            selected_features.remove(feature_with_p_max)
        else:
            break

    print(len(selected_features),selected_features)
