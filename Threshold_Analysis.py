'''This file contains the code for doing threshold analysis. Basically, we want to know, at the
feature level, what drives the sharp transitions of the prediction score for the MX flare prediction

In detail, let us focus on these analyses:
i. For all MX first flares, what drives the sharp transitions in prediction score, can we specify any
numerical threshold for any HMI header features that can explain the sharp transitions?

ii. For other MX flares that have experienced some sharp transitions(not restricted to FF), can we do
the same analysis?

iii. Is our result robust to other sharp transition detection method other than the naive one that I
have specified previously?
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

# The first function here uses the trivial method for finding transition point:
# namely we find the ex-post time point at which the prediction score stays high at a
# certain threshold and also persist for a while, and the ex-ante point where the prediction score remains low
# and also has a few preceding time periods where the score remains low as well

# The model considered here is the First-Flare prediction model, and all flares that we currently
# consider is the M/X first flares. We consider the simpler binary classification model

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

# The next function is a visual examination of the selected ARs filtered by function Feature_Analysis

def Visual_Threshold_A(data,time,AR_list,all_score,show="score",wholeseries=False):
    # in the graphics, I am considering drawing a few example graphs to show the sharp transitions of the
    # prediction score for some selected active regions
    fig, ax = plt.subplots(nrows=3,ncols=4,sharey=True,sharex=True)

    if show=="score":
        for i in range(11):
            t1 = list(range(all_score.shape[1]))
            t2 = [t1[p] + 150 for p in range(len(t1))]  # this makes sure that every pair of lines does not overlap too much
            y1 = all_score[2*i]
            y2 = all_score[2*i+1]

            plt.subplot(3,4,i+1)
            plt.plot(t1,y1,color='tab:red',linewidth=0.5)
            plt.plot(t2,y2,color='tab:orange',linewidth=0.5)

            # and we also label the ex-ante/ex-post sharp transition points
            index = list(time.keys())
            y1_ante = time[index[2*i]][0]
            y1_post = time[index[2*i]][1]

            plt.plot(t1[(y1_ante-3):y1_ante],y1[(y1_ante-3):y1_ante],color='tab:blue',linewidth=3)
            plt.plot(t1[(y1_post):y1_post+3], y1[(y1_post):y1_post+3], color='tab:blue',linewidth=3)

            y2_ante = time[index[2 * i+1]][0]
            y2_post = time[index[2 * i+1]][1]

            plt.plot(t2[(y2_ante - 3):y2_ante], y2[(y2_ante - 3):y2_ante], color='tab:blue',linewidth=3)
            plt.plot(t2[(y2_post):y2_post + 3], y2[(y2_post):y2_post + 3], color='tab:blue',linewidth=3)

            plt.title(str(AR_list[2*i])+'/'+str(AR_list[2*i+1]),loc='left', fontsize=7, fontweight=0)
            plt.ylim((-0.2,1.2))

            if i not in [0, 4, 8]:
                plt.tick_params(labelleft='off')
            if i in list(range(12)):
                plt.tick_params(labelbottom='off')
    else:
        for i in range(11):
            feature_list = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                            'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                            'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX']

            index = list(time.keys())
            t1 = list(range(data.shape[1]))
            t2 = [t1[p] + 300 for p in range(len(t1))]  # this makes sure that every pair of lines does not overlap too much


            if show==20:
                # here we could construct a new feature called TOTUSJH/SIZE_ACR
                y1 = data[index[2 * i], :, 9]/data[index[2 * i], :, 16]
                y2 = data[index[2 * i + 1], :, 9]/data[index[2 * i + 1], :, 16]  # we are plotting the number "show" feature
            else:
                y1 = data[index[2 * i], :, show]
                y2 = data[index[2 * i + 1], :, show]  # we are plotting the number "show" feature
                if show==11:
                    y1 = np.log10(y1)
                    y2 = np.log10(y2)

            plt.subplot(3,4,i+1)
            #plt.plot(t1,y1,color='tab:red',linewidth=0.5)
            #plt.plot(t2,y2,color='tab:orange',linewidth=0.5)

            # and we also label the ex-ante/ex-post sharp transition points
            y1_ante = time[index[2*i]][0]
            y1_post = time[index[2*i]][1]
            if wholeseries==False:
                plt.plot(t1[y1_ante:y1_ante+30],y1[y1_ante:y1_ante+30],color='tab:red',linewidth=1)
                plt.plot(t1[y1_post+30:y1_post+60], y1[y1_post:y1_post+30], color='tab:orange',linewidth=1)
            else:
                plt.plot(t1,y1,color='tab:red',linewidth=1,alpha=0.3)
                shift = 0.2*max(y1)
                #plt.plot(t1[y1_ante:y1_ante + 30], y1[y1_ante:y1_ante + 30]-shift, alpha=0.5, linewidth=1,color='tab:red')
                plt.plot(t1[y1_post:y1_post + 30], y1[y1_post:y1_post + 30],alpha=1, linewidth=1,color='tab:red')

            y2_ante = time[index[2 * i+1]][0]
            y2_post = time[index[2 * i+1]][1]

            if wholeseries==False:
                plt.plot(t2[y2_ante:y2_ante + 30], y2[y2_ante:y2_ante + 30], color='tab:red', linewidth=1)
                plt.plot(t2[y2_post + 30:y2_post + 60], y2[y2_post:y2_post + 30], color='tab:orange', linewidth=1)
            else:
                plt.plot(t2, y2, color='tab:orange', linewidth=1,alpha=0.3)
                shift = 0.2 * max(y2)
                #plt.plot(t2[y2_ante:y2_ante + 30], y2[y2_ante:y2_ante + 30] - shift, alpha=0.5, linewidth=1,color='tab:orange')
                plt.plot(t2[y2_post:y2_post + 30], y2[y2_post:y2_post + 30], alpha=1, linewidth=1,color='tab:orange')

            plt.title(str(AR_list[2*i])+'/'+str(AR_list[2*i+1]),loc='left', fontsize=7, fontweight=0)
            if show==9:
                plt.ylim((0,5000))
            if show==11:
                plt.ylim((11.5,14))

            if i not in [0, 4, 8]:
                plt.tick_params(labelleft='off')
            if i in list(range(12)):
                plt.tick_params(labelbottom='off')

    # title and save section
    if show=="score":
        fig.suptitle("Prediction Score and Sharp Transition", fontsize=11, fontweight=0, color='black',
                     style='italic', y=0.95)
        plt.savefig('Prediction Score and Sharp Transition.pdf')
    else:
        if show==20:
            feature = 'TOTUSJH_SIZE'
        else:
            feature = feature_list[show]
        fig.suptitle(feature+" and Sharp Transition", fontsize=11, fontweight=0, color='black',
                     style='italic', y=0.95)
        plt.savefig(feature+' and Sharp Transition.pdf')

# The next function is a visual examination of where to draw a threshold:
def Visual_Threshold_B(data,time,AR_list,all_score,show=9):
    # in the graphics, I am considering drawing a few example graphs to show the sharp transitions of the
    # prediction score for some selected active regions

    fig = plt.figure()
    for i in range(23):
        feature_list = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                        'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                        'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX']

        index = list(time.keys())
        if show==20:
            y1 = data[index[i], :, 9]/data[index[i], :, 17]
        else:
            y1 = data[index[i], :, show]


        y1_ante = time[index[i]][0]
        y1_post = time[index[i]][1]
        if i==0:
            plt.scatter(y1[y1_ante],i,color='tab:red',marker='o',label='before')
            plt.scatter(y1[y1_post],i,color='tab:orange',marker='^',label='after')
        else:
            plt.scatter(y1[y1_ante], i, color='tab:red', marker='o')
            plt.scatter(y1[y1_post], i, color='tab:orange', marker='^')
        plt.plot([y1[y1_ante],y1[y1_post]],[i,i],linewidth=0.5,linestyle='dashed')

        #if i not in [0, 4, 8]:
        #    plt.tick_params(labelleft='off')

    if show==20:
        feature = 'TOTUSJH_SIZE_ACR'
    else:
        feature = feature_list[show]
    plt.xlabel(feature)
    plt.legend(loc='upper right')
    fig.suptitle(feature+" and Sharp Transition", fontsize=11, fontweight=0, color='black',
                     style='italic', y=0.95)
    plt.savefig(feature + ' horizontal visualization.pdf')

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
    modelpath = 'FF_deri.h5'
    meanpath = 'FFmean.npy'
    stdpath = "FFstd.npy"
    dataset, Y, AR = np.load('HARP_FFdataset.npy'), np.load('HARP_FFlabel.npy'), np.load('HARP_FFAR.npy')

    #data, time, AR_list, all_score = Feature_Analysis(modelpath,meanpath,stdpath,dataset,Y,AR)
    #np.save('wholeseries',data)
    #np.save('Sharp_Trans_AR',np.array(AR_list))
    #np.save('Sharp_Trans_Scores',np.array(all_score))
    #save_obj(time,'time')
    data, AR_list, all_score = np.load('wholeseries.npy'), np.load('Sharp_Trans_AR.npy'), np.load('Sharp_Trans_Scores.npy')
    time = load_obj('time')
    Visual_Threshold_A(data,time,AR_list,all_score,show=11,wholeseries=True)
    Visual_Threshold_B(data, time, AR_list, all_score, show=9)

    time = np.load('HARP_FFTime.npy')

    dataset, label, AR, time = retrieve_data(dataset,Y,AR,time,derivative=False,maxlen=240,extra_feature=False)
    trans_time = load_obj('time')
    transition_time(trans_time,time,AR)
