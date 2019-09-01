import numpy as np
from scipy import linalg
import pandas as pd
from geopy.distance import vincenty
import datetime
import time



def parse_data_hour(data_str: str) -> int:
    return datetime.datetime.strptime(data_str, "%Y-%m-%d %H:%M:%S.%f").time().hour

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
    # print(f'Question 1: {len(df.index)}, {len(df.columns)}')
    # print(f"Question 2: {round(df['tripduration'].mean()/60,2)}")
    # print(f"Question 3: {len(df[df['start station id']==df['end station id']].index)}")
    # print(f"Question 4: {df['bikeid'].value_counts().head()}")
    # print(f"Question 5: ")
    # print(f"Question 6: {df['start station id'].isnull().sum()}")
    # df_customer = df[df['usertype']=='Customer']
    # df_subscriber = df[df['usertype']=='Subscriber']
    # print(f"Question 7: Customer: {round(df_customer['tripduration'].mean()/60,2)},Subscriber: {round(df_subscriber['tripduration'].mean()/60,2)}")
    # df_track = df[df['start station id']!=df['end station id']]
    # distance = [dist((df_track['start station latitude'][i],df_track['start station longitude'][i]), (df_track['end station latitude'][i],df_track['end station longitude'][i])) for i in df_track.index]
    # df_track.insert(loc = len(df.columns), column='distance',value=distance)
    # print(f"Question 8:{round(np.mean(distance),2)}")
    # df_station = df[parse_data_hour(df['starttime'])>=18]
    # df_2 = df_station[parse_data_hour(df_station['starttime']) <= 20]
    # print(len(df_2['starttime']))
    # print(df_2['starttime'][0])
    # print(type(parse_data_hour(df_2['starttime'][0])))
    df['end_hour'] = df['stoptime'].apply(lambda x: datetime.datetime.fromtimestamp(time.mktime(datetime.datetime.strptime(x.strip(), "%Y-%m-%d %H:%M:%S.%f").timetuple())).hour)
    df['start_hour'] = df['starttime'].apply(lambda x: datetime.datetime.fromtimestamp(
        time.mktime(datetime.datetime.strptime(x.strip(), "%Y-%m-%d %H:%M:%S.%f").timetuple())).hour)
    print(df[(df.start_hour.isin([18, 19, 20]))]['start station id'].value_counts().head())

    # start_stations = [df['start station id'][i] for i in df.index if parse_data_hour(df['starttime'][i])>=18 and parse_data_hour(df['starttime'][i]) <= 20]
    # counts = [(el, start_stations.count(el)) for el in start_stations]
    # print(f"Question 9:{sorted(counts, key = lambda element : element[1],reverse=True)[0]}")



def main():
    # test1()
    test2()

if __name__ == '__main__':
    main()

