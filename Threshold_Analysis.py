'''This file contains the code for doing sharp transition analysis. Basically, we want to know, at the
feature level, what drives the sharp transitions of the prediction score for the MX flare prediction

'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import UnivariateSpline
from keras.layers import Input, Dense, LSTM
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
# for each flare we could use retrieve_data() to obtain an arbitrary length of SHARP parameter history from the dataset
def retrieve_data(dataset,Y,AR,time,derivative=False,maxlen=96,extra_feature=False):
    obs = dataset.shape[0]
    num_feature = 20
    newdata = []
    newY = []
    newAR = []
    newtime = []
    t = [0.2*i for i in range(maxlen)]

    if extra_feature==True:
        # currently, we only consider TOTUSJH/SIZE_ACR
        num_feature = 21
        new_feature = []
        for i in range(obs):
            D = dataset[i]
            obs_new_feature = []
            for j in range(D.shape[0]):
                obs_new_feature.append(D[j,9]/(D[j,17]+0.01)) # = TOTUSJH/SIZE_ACR
            obs_new_feature = np.array(obs_new_feature).reshape((D.shape[0],1))
            new_feature.append(obs_new_feature)

        new_feature = np.array(new_feature)
        new_dataset = []
        for i in range(obs):
            D = np.concatenate([dataset[i],new_feature[i]],axis=1)
            new_dataset.append(D)
        dataset = np.array(new_dataset)

    for i in range(obs):
        data = dataset[i]
        timerange = time[i]
        datalen = data.shape[0]
        if datalen>=maxlen:
            D = data[(datalen-maxlen):(datalen),]
            thetime = timerange[(datalen-maxlen):(datalen)]
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
            newAR.append(AR[i])
            newtime.append(thetime)

    return np.array(newdata),np.array(newY), np.array(newAR), np.array(newtime)

def fitspline(data):
    '''This function could fit cubic spline on every observation in data'''
    obs = data.shape[0]
    num_feature = 20
    timerange = [0.2*p for p in range(data.shape[1])] # since every two adjacent data points are separated by 12 mins

    data_deri = []
    for i in range(obs):
        obs_deri = []
        for j in range(num_feature):
            L = data[i,:,j]
            Lvar = np.std(L)
            fspline = UnivariateSpline(timerange,L,s=Lvar**2)
            fspline_deri = fspline.derivative()
            feature_deri = fspline_deri(timerange)
            obs_deri.append(feature_deri)
        obs_deri = np.array(obs_deri).transpose()
        data_deri.append(obs_deri)
    data_deri = np.array(data_deri)
    data = np.concatenate((data,data_deri),axis=2)

    return data

# namely we find the ex-post time point at which the prediction score stays high at a
# certain threshold and also persist for a while, and the ex-ante point where the prediction score remains low
# and also has a few preceding time periods where the score remains low as well

# The model considered here is the First-Flare prediction model, and all flares that we currently
# consider is the M/X first flares. We consider the simple binary classification model

def Feature_Analysis(modelpath,meanpath,stdpath,dataset,Y,AR,num_hour=12,num_period=6,threshold=0.7,low_threshold=0.3):
    dataset, Y, AR, newtime = retrieve_data(dataset,Y,AR,derivative=False,maxlen=240)
    # LON_FWT feature: the feature that determines if the data is of good quality.

    # dataset_deri, Y_deri = retrieve_data(dataset,Y,derivative=True,maxlen=240)

    # load LSTM-relevant amounts
    model = load_model(modelpath)  # load the LSTM pre-trained model, 12-hour, 6-hour model
    mean = np.load(meanpath)
    std = np.load(stdpath)

    mean = mean[0:40]
    std = std[0:40]

    obs = dataset.shape[0] # number of observation
    time = dict()
    AR_list = []
    all_score = []

    for i in range(obs):
        if Y[i]==1: # we only investigate the case of M/X first flares
            scorepath = []
            data = dataset[i]  # still use numpy array for LSTM prediction
            data = np.reshape(data,(1,data.shape[0],data.shape[1]))

            # fit cubic spline for each data window
            last_frame = dataset.shape[1]-(num_period+num_hour)*5
            for j in range(last_frame):
                train_data = data[0,j:(j+num_period*5),:]
                train_data = np.reshape(train_data,(1,train_data.shape[0],train_data.shape[1]))
                train_data = fitspline(train_data)
                train_data = np.reshape(train_data,(num_period*5,40))

                # normalize the data
                train_data_norm = (train_data-mean)/std
                train_data_norm = np.reshape(train_data_norm,(1,train_data_norm.shape[0],train_data_norm.shape[1]))
                score = model.predict(train_data_norm)
                s = score[0,0]
                scorepath.append(s)


            # then we are going to identify when a certain score threshold is being hit
            scorepass = [p>=threshold for p in scorepath]

            if sum(scorepass)>0:

                # now we are going to find the time point where the threshold is being hit and persist
                # for at least 3 consecutive periods (typically a flare would not vanish within 24 mins,
                # hopefully:))
                findit = False # an indicator for whether the timepoint is found
                timepoint = 0 # the variable storing the value of the timepoint getting found

                while findit==False and timepoint<=(len(scorepass)-3):
                    thetime = min([k for k, x in enumerate(scorepass) if x])
                    if np.sum(scorepass[thetime:(thetime+3)])==3:
                        findit = True
                        timepoint = timepoint+thetime
                        break
                    else:
                        scorepass = scorepass[(thetime+1):]
                        timepoint = thetime+1

                if findit==True:
                    scorelow = [p<=low_threshold for p in scorepath]
                    find_exante = False
                    for l in range(1,timepoint-1):
                        p = scorelow[(timepoint-l-2):(timepoint-l+1)]
                        if np.sum(p)==3:
                            exante = timepoint-l
                            find_exante = True
                            break

                if findit==True and find_exante==True:
                    time[i] = [exante,timepoint]
                    AR_list.append(AR[i])
                    all_score.append(np.array(scorepath))

            else:
                continue # the threshold, in this case, is never hit
    return dataset, time, AR_list, np.array(all_score)

# the function transition_time() is used to generate a csv file recording the before/after transition time and the peak time of the flare.
def transition_time(transtime,timedata,ARdata):

    exantetime = []
    exposttime = []
    AR = []
    flaretime = []

    for region in list(transtime.keys()):
        AR.append(ARdata[region])
        exante = transtime[region][0]
        expost = transtime[region][1]

        exante = timedata[region,exante+30][0:-4]
        expost = timedata[region,expost+30][0:-4]

        exantetime.append(exante)
        exposttime.append(expost)
        flaretime.append(timedata[region,-1][0:-4])

    exantetime,exposttime,AR = np.array(exantetime).reshape((-1,1)),np.array(exposttime).reshape((-1,1)),np.array(AR).reshape((-1,1))
    flaretime = np.array(flaretime).reshape((-1,1))
    framedata = np.concatenate((AR, exantetime, exposttime,flaretime), axis=1)

    column = ['AR','before_transition','after_transition','flare_time']
    output = pd.DataFrame(framedata,columns=column)
    output.to_csv('Time.csv')

if __name__=="__main__":
    # Do the sharp transition detection
    modelpath = 'FF_deri.h5'
    meanpath = 'FFmean.npy'
    stdpath = "FFstd.npy"
    dataset, Y, AR = np.load('HARP_FFdataset.npy'), np.load('HARP_FFlabel.npy'), np.load('HARP_FFAR.npy')

    data, time, AR_list, all_score = Feature_Analysis(modelpath,meanpath,stdpath,dataset,Y,AR)
    np.save('wholeseries',data)
    np.save('Sharp_Trans_AR',np.array(AR_list))
    np.save('Sharp_Trans_Scores',np.array(all_score))
    save_obj(time,'time')
    #data, AR_list, all_score = np.load('wholeseries.npy'), np.load('Sharp_Trans_AR.npy'), np.load('Sharp_Trans_Scores.npy')
    #time = load_obj('time')
    
    
    # Generating the Time.csv file
    time = np.load('HARP_FFTime.npy')

    dataset, label, AR, time = retrieve_data(dataset,Y,AR,time,derivative=False,maxlen=240,extra_feature=False)
    trans_time = load_obj('time')
    transition_time(trans_time,time,AR)
