import numpy as np
from scipy import linalg
import pandas as pd
from geopy.distance import vincenty


def test1():
    b = np.array([[-1, 33, 4, 1], [0, 1, 1, 0]])

    A = np.array([[0, 9, 19, 13], [1, 20, 5, 13], [12, 11, 3, 4]])
    B = np.array([[2, 0, 0, 0], [1, 2, 2, 0], [2, 1, 1, 0], [0, 0, 1, 1]])
    print(A)
    print(B)
    print(np.dot(A,B))
    print(np.mean(b))

    basis = np.array([[6, 0, 3], [0, -1, 2], [12, 3, 0]])
    a = np.array([[1, -1, -1, 0], [-1, 2, -1,-1], [-1,-1,2,-1], [0, -1,-1, 1]])
    print(linalg.det(basis))
    print(linalg.eigh(a))

    I = np.array([[2,4,0,4,1], [2,4,1,1,0],[1,1,1,2,2], [0,1,3,2,4],[2,2,2,0,2]])
    I_ = linalg.inv(I)
    print(np.trace(I_))

def dist(p1,p2):
    return vincenty(p1,p2).km
def test2():
    df = pd.read_csv('citibike-tripdata.csv')
    print(f'Question 1: {len(df.index)}, {len(df.columns)}')
    print(f"Question 2: {round(df['tripduration'].mean()/60,2)}")
    print(f"Question 3: {len(df[df['start station id']==df['end station id']].index)}")
    print(f"Question 4: {df['bikeid'].value_counts().head()}")
    print(f"Question 5: ")
    print(f"Question 6: {df['start station id'].isnull().sum()}")
    df_customer = df[df['usertype']=='Customer']
    df_subscriber = df[df['usertype']=='Subscriber']
    print(f"Question 7: Customer: {round(df_customer['tripduration'].mean()/60,2)},Subscriber: {round(df_subscriber['tripduration'].mean()/60,2)}")
    df_track = df[df['start station id']!=df['end station id']]
    distance = [dist((df_track['start station latitude'][i],df_track['start station longitude'][i]), (df_track['end station latitude'][i],df_track['end station longitude'][i])) for i in df_track.index]
    df_track.insert(loc = len(df.columns), column='distance',value=distance)
    print(f"Question 8:{round(np.mean(distance),2)}")
    print(f"Question 9:{df[df['']]}")


def main():
    # test1()
    test2()

if __name__ == '__main__':
    main()