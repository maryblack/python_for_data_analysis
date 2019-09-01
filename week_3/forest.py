from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def accuracy_cv(X, y):
    n_trees = [1, 5, 10, 20]
    cv_max = 0
    for trees in n_trees:
        rfc = RandomForestClassifier(random_state=42, n_estimators=trees)
        cv_rfc = cross_val_score(rfc, X, y, cv=StratifiedKFold(4), scoring='accuracy').mean()
        if cv_rfc > cv_max:
            cv_max = cv_rfc

    return round(cv_max, 3)

def compare_gbc_logreg(X, y):
    logerg = LogisticRegression(random_state=42)
    cv_log = cross_val_score(logerg, X, y, cv=StratifiedKFold(4), scoring='accuracy').mean()

    gbc = GradientBoostingClassifier(random_state=42)
    cv_gbc = cross_val_score(gbc, X, y, cv=StratifiedKFold(4), scoring='accuracy').mean()

    return round(max(cv_gbc, cv_log), 3)

def choose_model(data):
    mse_min = 10000
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=54, test_size=0.33)
    models = [
        RandomForestRegressor(random_state=42),
        Ridge(random_state=42),
        Lasso(random_state=42),
        GradientBoostingRegressor(random_state=42)
    ]
    for i in range(len(models)):
        m = models[i]
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        if mse < mse_min:
            mse_min = mse

    return round(mse_min, 2)


def main():
    X, y = load_wine(return_X_y=True)
    print(f'answer 1: {accuracy_cv(X, y)}')
    print(f'answer 2: {compare_gbc_logreg(X, y)}')

    boston = load_boston()
    print(f'answer 3: {choose_model(boston)}')

if __name__ == '__main__':
    main()