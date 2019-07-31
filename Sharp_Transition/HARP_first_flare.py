import numpy as np
import os
import h5py
import pandas as pd
import datetime as dt
from datetime import datetime
import multiprocessing as mp
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from scipy.interpolate import UnivariateSpline
from keras.models import load_model
import sys

def get_class_label(intsy):
    flare_class = intsy[0]
    return flare_class

def create_first_flare_data(f,GOES,feature_list):
    '''This function aims at constructing a first flare dataset'''
    dataset = []
    Y_label = []
    AR = []
    Time = []

    for video in f.keys():
        vobj = f[video]
        active = int(vobj.attrs['NOAA_AR']) ## take record of the active region number
        if active in AR:
            continue
        framelist = list(vobj.keys())
        framelist = sorted(framelist, key=lambda x: int(x[5:]), reverse=False)
        timelist = [vobj[frame].attrs['T_REC'][0:-4] for frame in framelist]

        df = GOES
        df_AR = df[df['NOAA_ar_num']==active]

        # extract the flare type of each recorded flare
        df_intsy = df_AR['class'].values
        df_class = [get_class_label(p) for p in df_intsy]
        df_AR['flare'] = df_class

        if not 'B' in df_class:
            if not 'M' in df_class:
                if not 'X' in df_class:
                    continue

        if 'B' in df_class:
            video_time = []
            df_B = df_AR[df_AR['flare']=='B']
            B_endtime = pd.to_datetime(df_B['end_time']).values
            B_firstflare = min(B_endtime)
            ts = (B_firstflare - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            B_firstflare = datetime.utcfromtimestamp(ts)
            B_frames = [datetime.strptime(timelist[i],"%Y.%m.%d_%H:%M:%S")<=B_firstflare for i in range(len(timelist))]
            B_frames = [framelist[c] for c in range(len(B_frames)) if B_frames[c]==True]

            B_data = []
            for frame in B_frames:
                entry = [vobj[frame].attrs[feature] for feature in feature_list]
                thetime = vobj[frame].attrs['T_REC']
                if not np.isnan(entry).any():
                    B_data.append(entry)
                    video_time.append(thetime)
            B_data = np.array(B_data).reshape((-1,len(feature_list)))
            dataset.append(B_data)
            Y_label.append(0)
            AR.append(active)
            Time.append(np.array(video_time))


        if 'M' in df_class or 'X' in df_class:
            video_time = []
            df_MX = df_AR[(df_AR['flare']=='M') | (df_AR['flare']=='X')]
            MX_endtime = pd.to_datetime(df_MX['end_time']).values
            MX_firstflare = min(MX_endtime)
            ts = (MX_firstflare - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            MX_firstflare = datetime.utcfromtimestamp(ts)
            MX_frames = [datetime.strptime(timelist[i], "%Y.%m.%d_%H:%M:%S") <= MX_firstflare for i in
                        range(len(timelist))]
            MX_frames = [framelist[c] for c in range(len(MX_frames)) if MX_frames[c] == True]

            MX_data = []
            for frame in MX_frames:
                entry = [vobj[frame].attrs[feature] for feature in feature_list]
                thetime = vobj[frame].attrs['T_REC']
                if not np.isnan(entry).any():
                    MX_data.append(entry)
                    video_time.append(thetime)
            MX_data = np.array(MX_data).reshape((-1, len(feature_list)))
            dataset.append(MX_data)
            Y_label.append(1)
            AR.append(active)
            Time.append(video_time)
    dataset = np.array(dataset)
    Y_label = np.array(Y_label)
    AR = np.array(AR)
    Time = np.array(Time)

    np.save('HARP_FFdataset',dataset)
    np.save('HARP_FFlabel',Y_label)
    np.save('HARP_FFAR',AR)
    np.save('HARP_FFTime',Time)
    return dataset, Y_label, AR, Time


def retrieve_data(dataset,Y,derivative=False,maxlen=96):
    obs = dataset.shape[0]
    num_feature = 20
    newdata = []
    newY = []
    t = [0.2*i for i in range(maxlen)]

    for i in range(obs):
        data = dataset[i]
        datalen = data.shape[0]
        if datalen>=maxlen:
            D = data[(datalen-maxlen):(datalen),]
            if derivative==True:
                #D_deri = np.gradient(D,axis=0)
                #D = np.concatenate((D,D_deri),axis=1)
                D_deri = []
                for j in range(num_feature):
                    sdata = D[:,j]
                    localvar = np.std(sdata)
                    f_spline = UnivariateSpline(t,sdata,s=2*len(t)*localvar)
                    f_deri = f_spline.derivative()
                    D_deri.append(f_deri(t))

                D_deri = np.array(D_deri).transpose()
                D = np.concatenate((D, D_deri), axis=1)


            newdata.append(D)
            newY.append(Y[i])

    return np.array(newdata),np.array(newY)

def new_normalization(X_train, X_test):
    num_feature = X_train.shape[2]
    mean_train = []
    std_train = []

    for k in range(num_feature):
        feature = X_train[:, :, k]
        feature = np.reshape(feature, (-1, 1))
        mean = np.mean(feature)
        std = np.std(feature)

        mean_train.append(mean)
        std_train.append(std)

    mean_train, std_train = np.array(mean_train), np.array(std_train)

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            tmp = X_train[i, j, :]
            X_train[i, j, :] = (tmp - mean_train) / std_train

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            tmp = X_test[i, j, :]
            X_test[i, j, :] = (tmp - mean_train) / std_train
    np.save('FFmean',mean_train)
    np.save('FFstd',std_train)
    return X_train, X_test

def fitspline(data,smooth_correction=False):
    '''This function could fit cubic spline on every observation in data'''
    obs = data.shape[0]
    num_feature = 20

    if smooth_correction==True:
        # we are going to use the average of the absolute value of
        # between-frame feature percentage change as a gauge for time series volatility
        resmooth_portion = 0.2
        logdata = np.log(data)
        datadiff = np.absolute(np.diff(logdata,axis=1))

        percentage_ave = np.mean(datadiff,axis=1).reshape(obs,num_feature) # should be of shape obs*20
        resmooth = np.zeros_like(percentage_ave) # change an entry to 1 if it needs to be resmoothed.
        for i in range(20):
            feature = percentage_ave[:,i]
            feature = np.reshape(feature,(-1))

            sort_index = np.argsort(feature)
            sort_index = sort_index[round((1-resmooth_portion)*obs):] # most volatile ones
            resmooth[sort_index,i] = 1

    timerange = [0.2*p for p in range(data.shape[1])]
    data_deri = []
    # refit_list = [0 for i in range(num_feature)]
    for i in range(obs):
        obs_deri = []
        for j in range(num_feature):
            L = data[i,:,j]
            Lvar = np.std(L)
            fspline = UnivariateSpline(timerange, L, s=Lvar ** 2)
            if smooth_correction==True:
                refit = resmooth[i,j]
                if refit==1:
                    fspline = UnivariateSpline(timerange,L,s=4*Lvar**2)
            fspline_deri = fspline.derivative()
            feature_deri = fspline_deri(timerange)
            obs_deri.append(feature_deri)
        obs_deri = np.array(obs_deri).transpose()
        data_deri.append(obs_deri)
    data_deri = np.array(data_deri)
    data = np.concatenate((data,data_deri),axis=2)

    return data

def LSTM_fit(dataset,Y,num_hour,num_period,derivative=False):
    data, label = retrieve_data(dataset, Y, derivative=False, maxlen=240) # get the long enough data
    # data, mean, std = normalize(data)
    # now we gonna retrieve the specific time period from each video
    start = (num_hour+num_period)*5
    end = num_hour*5
    totallen = data.shape[1]

    data = data[:,(totallen-start+1):(totallen-end)+1,:]
    if derivative==True:
        data = fitspline(data,smooth_correction=False)

    ## train test split
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.33,random_state=42)
    X_train, X_test = new_normalization(X_train,X_test)

    ## construct LSTM model
    period = data.shape[1]
    num_features = data.shape[2]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(period, num_features)))
    model.add(Dropout(0.5))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=10)
    if derivative==True:
        model.save("FF_deri.h5")
    else:
        model.save("FF_noderi.h5")
    #y_pred = model.predict_classes(X_test).ravel()
    #TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

def model_train_parallel(dataset,Y,iteration=1,steps=2,derivative=False,smooth_correction=False):
    num_iterations_start = 1 + steps * (iteration - 1)
    num_iterations_end = steps * iteration  # run 2 iterations on a single core

    num_hour_list = [1, 6, 12, 18, 24]
    num_period_list = [1, 3, 6, 12, 24]
    num_features = 20

    # num_iterations = 1
    # num_hour_list = [1]
    # num_period_list = [1,6]

    metrics = ['TN', 'FP', 'FN', 'TP']

    c1 = np.ravel([[x] * len(num_period_list) * len(metrics) for x in num_hour_list])
    c2 = np.ravel([[x] * len(metrics) for x in num_period_list] * len(num_hour_list))
    c3 = np.array(['TN', 'FP', 'FN', 'TP'] * len(num_period_list) * len(num_hour_list))
    df = pd.DataFrame(index=range(num_iterations_start, num_iterations_end + 1),
                      columns=pd.MultiIndex.from_tuples(zip(c1, c2, c3)))

    data, label = retrieve_data(dataset, Y, derivative=False, maxlen=240)  # get the long enough data
    # data, mean, std = normalize(data)

    for i in range(num_iterations_start, num_iterations_end + 1):
        for num_hour in num_hour_list:
            for num_period in num_period_list:
                # print(i, num_hour, num_period)

                # now we gonna retrieve the specific time period from each video
                start = (num_hour + num_period) * 5
                end = num_hour * 5
                totallen = data.shape[1]

                data_retrieve = data[:, (totallen - start + 1):(totallen - end+1), :]
                if derivative==True:
                    data_retrieve = fitspline(data_retrieve,smooth_correction=smooth_correction)
                # data_retrieve, mean, std = normalize(data_retrieve)

                ## train test split
                X_train, X_test, y_train, y_test = train_test_split(data_retrieve, label, test_size=0.33)
                X_train, X_test = new_normalization(X_train,X_test)

                ## construct LSTM model
                period = data_retrieve.shape[1]
                num_features = 20
                if derivative == True:
                    num_features = 40

                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(period, num_features)))
                model.add(Dropout(0.5))
                model.add(LSTM(50))
                model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=10)
                # model.save("flare.h5")

                y_pred = model.predict_classes(X_test).ravel()
                TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
                df.loc[i][num_hour][num_period]['TN'] = TN
                df.loc[i][num_hour][num_period]['FP'] = FP
                df.loc[i][num_hour][num_period]['FN'] = FN
                df.loc[i][num_hour][num_period]['TP'] = TP

    return df

def main(indicator):
    if indicator==0:
        derivative = False
        smooth_correction = False
    elif indicator==1:
        derivative = True
        smooth_correction = False
    elif indicator==2:
        derivative = True
        smooth_correction=True

    f = h5py.File("HARP_with_flare_j.hdf5", "r")
    GOES = pd.read_csv("GOES_dataset.csv")
    feature_list = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                    'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                    'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX']
    dataset, Y, AR = create_first_flare_data(f,GOES,feature_list)
    #dataset, Y, AR = np.load("./Code/HARP_dataset.npy"), np.load("./Code/HARP_label.npy"), np.load("./Code/HARP_AR.npy")


    pool = mp.Pool(processes=8)  # run, at maximum, 8 processes simultaneously
    process_input = []
    steps = 1
    iteration = 20
    for i in range(1, iteration + 1):
        parallel_input = (dataset,Y,i,steps,derivative,smooth_correction)
        process_input.append(parallel_input)
    results = [pool.apply_async(model_train_parallel, t) for t in process_input]
    # the above results shall be 10 separate datasets with each containing 2 iterations results
    output = [p.get() for p in results]
    metrics = pd.concat(output)

    filename = 'timeseries.csv'
    if derivative==True:
        filename = 'timederivative.csv'
    if smooth_correction==True:
        filename = 'timederivative_resmooth.csv'

    metrics.to_csv(filename)





if __name__=='__main__':
    f = h5py.File("../Data/HARP_with_flare_j.hdf5","r") # this is run on flux.
    GOES = pd.read_csv("../Data/GOES_dataset.csv")
    feature_list = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                  'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                 'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX']

    # the refit_list result in fitspline function shows that MEANGAM, MEANJZD,MEANALP,MEANJZH and MEANSHR are the most volatile features among all features
    # namely they have their cubic spline functions being tightly constrained by the constraint while being fitted
    dataset, Y, AR, Time = create_first_flare_data(f,GOES,feature_list)
    dataset, Y, AR = np.load("HARP_FFdataset.npy"), np.load("HARP_FFlabel.npy"), np.load("HARP_FFAR.npy")
    data, label = retrieve_data(dataset,Y,derivative=False,maxlen=240)
    #LSTM_fit(dataset,Y,num_hour=12,num_period=6,derivative=True)
    #post_dataset, post_GOES = create_postfirst_flare_data(f,GOES,feature_list)

    #main(int(sys.argv[1]))
