import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def split_xy(data):
    y_train = data['target']
    x_train = data.drop(['target'], axis=1)
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    return x_train, y_train

def scaler(X, y):
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    return scaler1.fit_transform(X), scaler2.fit_transform(y)

def fit_mse(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = Ridge(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse


def main():
    df = pd.read_csv('data.csv')
    print(df.head())
    X, y = split_xy(df)
    print(f'MSE without scaling: {fit_mse(X, y)}')
    scaler= StandardScaler()
    X_s = scaler.fit_transform(X)
    # y_s = scaler.transform(y)
    print(f'MSE with scaling: {fit_mse(X_s, y)}')


if __name__ == '__main__':
    main()