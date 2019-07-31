import numpy as np
import os
import h5py
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import datetime
from datetime import datetime
from keras.models import load_model
import pandas as pd
from adjustText import adjust_text
import matplotlib
import keras.losses
from keras import backend as K
from scipy.interpolate import UnivariateSpline
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import sys

def retrieve_HARP(filename,GOES):
    # GOES is the path to GOES dataset, which could label flares,
    ## filename is the HARP hdf5 file name
    f = h5py.File(filename,'r')
    video = f['video0'] # This is uniform across all HARP-based data file

    data = [] # collect the Feature Time Series Data
    timeline = [] # store the timeline for the time series
    framelist = list(video.keys())
    # resort the order of frames chronologically
    frame_list = sorted(framelist,key=lambda x: int(x[5:]),reverse=False)
    feature_list = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                    'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                    'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX']
    num_features = len(feature_list)

    for frame in frame_list:
        framedata = [video[frame].attrs[feature] for feature in feature_list]
        if not np.isnan(framedata).any():
            data.append(framedata)
            t = video[frame].attrs['T_REC']
            t = t.split("_")
            t = t[0]+' '+t[1]
            t = datetime.strptime(t,"%Y.%m.%d %H:%M:%S")
            timeline.append(t)

    data = np.array(data)
    data = np.reshape(data,(-1,num_features))

    AR = video[frame_list[0]].attrs['NOAA_AR']
    df = pd.read_csv(GOES) # GOES is simply a path of the GOES dataset
    df1 = df[df['NOAA_ar_num']==AR] # AR-relevant event time line
    df1['peak_time'] = pd.to_datetime(df1['peak_time'],format="%Y-%m-%d %H:%M:%S")

    data = pd.DataFrame(data,index=timeline,columns=feature_list)
    #data.reset_index(level=0,inplace=True)
    #data.columns = ["time"] + feature_list

    return data, AR, df1

def ARplot(modelpath,HARP,GOES,meanpath,stdpath,derivative=False,num_hour=12,num_period=6):
    filename = '../Data/HARP'+str(HARP)+'.hdf5'
    ARdata, AR, AR_GOES = retrieve_HARP(filename,GOES) # get the AR data



    # load LSTM-relevant amounts
    model = load_model(modelpath) # load the LSTM pre-trained model, 12-hour, 24-period
    mean = np.load(meanpath)
    std = np.load(stdpath)

    if derivative==False:
        mean = mean[0:20]
        std = std[0:20]

        # standardize data
        data = ARdata.values # still use numpy array for LSTM prediction
        data = data[:,1:21]
        L = data.shape[0]
        D = []

        for i in range(L):
            obs = data[i,]
            obs = (obs-mean)/std
            D.append(obs)
        data = np.array(D)
    elif derivative==True:
        mean = mean[0:40]
        std = std[0:40]

        # standardize data
        data = ARdata.values  # still use numpy array for LSTM prediction
        data = data[:, 1:21]
        timerange = [0.2*i for i in range(data.shape[0])]
        # fit cubic spline

        for j in range(20):
            timeseries = data[:,j]
            f_var = np.std(timeseries)
            tspline = UnivariateSpline(timerange,timeseries,s=len(timerange)*2*f_var)
            tspline_deri = tspline.derivative()
            f_deri = tspline_deri(timerange)
            f_deri = np.array(f_deri).reshape((-1,1))
            data = np.concatenate((data,f_deri),axis=1)

        L = data.shape[0]
        D = []

        for i in range(L):
            obs = data[i,]
            obs = (obs - mean) / std
            D.append(obs)
        data = np.array(D)

    num_hour = num_hour
    num_period = num_period
    first_frame = (num_hour+num_period)*5
    score = []
    TOTUSJH = []
    SAVNCPP = []
    data1 = ARdata.values

    for i in range(first_frame,data.shape[0]):
        start = i-(num_period+num_hour)*5
        end = start + num_period*5
        train_data = data[start:end,]
        train_data = np.reshape(train_data,(1,train_data.shape[0],train_data.shape[1]))
        s = model.predict(train_data)
        score.append(s[0])
        #intensity.append(s[0,1])
        TOTUSJH.append(data1[i,10])
        SAVNCPP.append(data1[i,12])

    fig, ax = plt.subplots(nrows=2,ncols=1,sharey=False,sharex=True)
    ax1 = ax[0]
    bx1 = ax[1]
    time = data1[first_frame:,0]
    ax1.plot(time,score,color='tab:red',linewidth=0.5)
    ax1.set_ylabel('prediction score',color='tab:red' )
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()
    ax2.plot(time,TOTUSJH,color='tab:blue',linewidth=0.5)
    ax2.set_ylabel('TOTUSJH',color='tab:blue' )
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    matplotlib.axes.Axes.ticklabel_format(self=ax2,axis='y',style='sci',scilimits=(0,0))

    # Now we change the time axis label
    days = mdates.DayLocator() # every day
    hour = mdates.HourLocator() # Every Hour
    Fmt = mdates.DateFormatter('%b %d')

    ax[1].xaxis.set_major_locator(days)
    ax[1].xaxis.set_major_formatter(Fmt)
    ax[1].xaxis.set_minor_locator(hour)
    datemin = np.datetime64(time[1], 'D')
    datemax = np.datetime64(time[-1],'D')+np.timedelta64(1, 'D')

    #ax[1].set_xlim(datemin,datemax)
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(0,1)


    bx1.plot(time,score,color='tab:red',linewidth=0.5)
    bx1.set_ylabel('prediction score',color='tab:red' )
    bx1.tick_params(axis='y', labelcolor='tab:red')
    bx2 = bx1.twinx()
    bx2.plot(time, SAVNCPP, color='tab:blue',linewidth=0.5)
    bx2.set_ylabel('SAVNCPP', color='tab:blue')
    bx2.tick_params(axis='y', labelcolor='tab:blue')

    # now we are going to plot the M/X/B/C flare events on the graph
    palette = {'B':'blue','C':'green','M':'red','X':'red'}
    position = {'B':0.25,'C':0.5,'M':0.75,'X':0.75}
    for row in AR_GOES.iterrows():
        event = row[1]['class'][0]
        this_color = palette[event]
        t = row[1]['peak_time']
        this_position = position[event]
        ax[0].axvline(t, color=this_color, linewidth=2, ymax=this_position - 0.01,
                      ymin=this_position - 0.06)
        ax[1].axvline(t, color=this_color, linewidth=2, ymax=this_position - 0.01,
                      ymin=this_position - 0.06)
    fig.autofmt_xdate()
    plt.suptitle('AR:'+str(AR))

    #plt.show()
    if derivative==False:
        figname = 'HARP'+str(HARP)+'_FF.pdf'
    else:
        figname = 'HARP'+str(HARP)+'_FFderi.pdf'
    plt.savefig(figname,transparent=True)