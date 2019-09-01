from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso



def choose_alpha(data):
    mse_min = 10000
    alpha_min = 10
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=54, test_size=0.33)
    alpha_range = np.linspace(10, 90, num=9)
    for av in alpha_range:
        model = Ridge(alpha=av, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        if mse < mse_min:
            mse_min = mse
            alpha_min = av

    return alpha_min, round(mse_min, 3)

def coef_selection(data):
    X = data.data
    y = data.target
    all_coef = len(X[0])
    # X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=54, test_size=0.33)
    model = Lasso(random_state=42)
    model.fit(X,y)
    selected_coef = 0
    for c in model.coef_:
        if c != 0:
            selected_coef += 1

    return selected_coef/all_coef

def lasso_mse(data):
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=54, test_size=0.33)
    model = Lasso(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return round(mse, 3)


def main():
    boston = load_boston()
    print(f'alpha and mse: {choose_alpha(boston)}')
    print(f'answer 4: {lasso_mse(boston)}')

    diabetes = load_diabetes()
    print(f'answer 3: {coef_selection(diabetes)}')

if __name__ == '__main__':
    main()