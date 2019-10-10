import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
import math

def nnz_ind(u, v):
    v_rated = (v != 0)
    u_rated = (u != 0)
    set_uv = u_rated.multiply(v_rated)

    return set_uv



def cosine_similarity_pair_users(u, v):
    s_ob = float(u.dot(v.T)[0, 0])
    nnz = nnz_ind(u, v)
    num = nnz.nnz
    u_new = u.multiply(nnz)
    v_new = v.multiply(nnz)
    s_u = float(u_new.dot(u_new.T)[0, 0])
    s_v = float(v_new.dot(v_new.T)[0, 0])
    if num > 2 and s_u != 0 and s_v != 0:
        return s_ob/math.sqrt(s_u*s_v)
    return 0

def similar_users(u, R, n_neigbours):
    user = R[u]
    res = []
    sim = 0
    row, _ = R.shape
    for i in range(row):
        sim = cosine_similarity_pair_users(user, R[i])
        res.append(sim)

    res = np.array(res)
    res_sort = np.argsort(res)[::-1]

    return res_sort[0:(n_neigbours)]


def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[0], x[2]))


def rate_items_user(u, R, n_neigbours):
    nn = similar_users(u, R, n_neigbours+1)
    R_ = []
    user = R[u]
    row, cols = R.shape
    s = []
    for i in nn:
        suv = cosine_similarity_pair_users(user, R[i])
        s.append(suv)
        R_.append(np.multiply(suv, R[i]))

    ind = (user != 0).indices
    notseen = []

    res = np.multiply(1/np.sum(s), np.sum(R_))
    nres = np.argsort(res.toarray())[0][::-1]
    i = 0
    j = 0
    while i < 5:
        if nres[j] in ind:
            pass
        else:
            notseen.append(nres[j])
            i+=1
        j += 1


    return notseen

def preprocessing():
    filepath = './data/user_ratedmovies.dat'
    df_rates = pd.read_csv(filepath, sep='\t')
    filepath = './data/movies.dat'
    df_movies = pd.read_csv(filepath, sep='\t', encoding='iso-8859-1')
    enc_user = LabelEncoder()
    enc_mov = LabelEncoder()
    enc_user = enc_user.fit(df_rates.userID.values)
    enc_mov = enc_mov.fit(df_rates.movieID.values)
    idx = df_movies.loc[:, 'id'].isin(df_rates.movieID)
    df_movies = df_movies.loc[idx]
    df_rates.loc[:, 'userID'] = enc_user.transform(df_rates.loc[:, 'userID'].values)
    df_rates.loc[:, 'movieID'] = enc_mov.transform(df_rates.loc[:, 'movieID'].values)
    df_movies.loc[:, 'id'] = enc_mov.transform(df_movies.loc[:, 'id'].values)
    R = coo_matrix((df_rates.rating.values, (df_rates.userID.values, df_rates.movieID.values)))
    R = R.tocsr()

    return R


def main():
    R = preprocessing()
    answer1 = round(cosine_similarity_pair_users(R[146], R[239]), 3)
    print(f'answer1: {answer1}')
    answer2 = np.array2string(similar_users(42, R, 10)).replace(' ', '').replace('[', '').replace(']', '')
    print(f'answer2: {answer2}')
    # print(len(answer2))
    # print(similar_users(42, R, 10))
    R_hat = rate_items_user(20, R, n_neigbours=30)
    # print(R_hat)
    print(f'answer3: {R_hat}')

if __name__ == '__main__':
    main()
