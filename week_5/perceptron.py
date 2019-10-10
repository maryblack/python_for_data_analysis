import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random


class Perceptron:
    def __init__(self, features, learning_rate, t_max):
        self.lr = learning_rate
        self.t_max = t_max
        self.w = np.zeros(features)
        self.b = 0

    def fit(self, X_train, y_train):
        size = len(y_train)
        t = 0
        while t < self.t_max:
            ind = random.randint(0, size)
            obj = X_train[ind]
            f_activation = y_train[ind]*(np.dot(self.w, obj) + self.b)
            if f_activation <= 0:
                self.b = self.b + self.lr*y_train[ind]
                self.w = np.add(self.w, np.multiply(self.lr, np.dot(y_train[ind], X_train[ind])))
            t += 1


    def predict(self, X_test):
        y_pred = []
        for el in X_test:
            if (np.dot(self.w, el) + self.b) >= 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)

        return y_pred





def main():
    X, y = load_iris(return_X_y=True)
    X, y = X[:100], y[:100]
    num_features = X.shape[1]
    y = np.array([1 if y_i == 1 else -1 for y_i in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    model = Perceptron(num_features, 0.1, 40)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = round(accuracy_score(y_test, y_pred), 2)
    print("score {0:.2f}".format(score))


if __name__ == '__main__':
    main()
