import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


class Metrics():
    def __init__(self, df):
        self.true = df['target']
        self.pred = df['prediction']
        self.score_1 = df['scores_1']
        self.score_2 = df['scores_2']

    def error_table(self):
        true = self.true
        pred = self.pred
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(0, len(true)):
            if true[i] == 1:
                if true[i] == pred[i]:
                    TP += 1
                else:
                    FN += 1
            elif true[i] == 0:
                if true[i] == pred[i]:
                    TN += 1
                else:
                    FP += 1
        return TN, FN, TP, FP

    def prf(self):
        TN, FN, TP, FP = self.error_table()
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F_score = 2 * P * R / (P + R)

        return P, R, F_score


def processing_csv(filename):
    df = Metrics(pd.read_csv(filename))
    print(f'answer 1 - TN, FN, TP, FP: {df.error_table()}')
    print(confusion_matrix(df.true, df.pred).ravel())
    print(f'answer 2 - P, R, F_score: {df.prf()}')
    print(f'P, R, F_score: {precision_score(df.true, df.pred)}, {recall_score(df.true, df.pred)}, {f1_score(df.true, df.pred)}')
    print(f'answer 3 - ROC-AUC for first model: {roc_auc_score(df.true, df.score_1)}')
    print(f'ROC-AUC for second model: {roc_auc_score(df.true, df.score_2)}')


def compare_recall(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.33, stratify=y)

    logerg = LogisticRegression(random_state=42)
    logerg.fit(X_train, y_train)
    y_pred_log = logerg.predict(X_test)
    r_log = recall_score(y_test, y_pred_log)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    r_knn = recall_score(y_test, y_pred_knn)

    return round(max(r_log, r_knn), 3)

def compute_cv(X, y):
    kf = StratifiedKFold(n_splits=3)
    kf.get_n_splits(X, y)

    logerg = LogisticRegression(random_state=42)
    cv_log = cross_val_score(logerg, X, y, cv=kf, scoring='recall').mean()

    knn = KNeighborsClassifier()
    cv_knn = cross_val_score(knn, X, y, cv=kf, scoring='recall').mean()

    return round(max(cv_log, cv_knn), 3)





def processing_built_in_data(X, y):
    # counts = pd.value_counts(y)
    # print(counts)
    print(f'answer 4: {compare_recall(X, y)}')
    print(f'answer 5: {compute_cv(X, y)}')



def main():
    filename = 'data2.csv'
    processing_csv(filename)

    X, y = load_breast_cancer(return_X_y=True)
    processing_built_in_data(X, y)




if __name__ == '__main__':
    main()