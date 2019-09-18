import numpy as np
import h5py
import seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import multiprocessing as mp
import glob
import sys
from functools import partial

def KSVM_dataprep(image,threshold=95):
    # this function processed the image data into a few selected pixels whose absolute Bz magnitude
    # is among the threshold-percentile of all pixels with the same polarity label
    height = image.shape[0]
    width = image.shape[1]

    new_data = []
    data_weight = []
    data_label = []
    pos_weight = []
    neg_weight = []
    valid_image = 1 # return a valid image flag with value=1 meaning that both pos/neg polars exist

    for i in range(height):
        for j in range(width):
            # Firstly, make the polarity of each pixel right

            pixel = image[i,j]

            if pixel>0:
                data_label.append(1)
                pos_weight.append(np.abs(pixel))
            else:
                data_label.append(-1)
                neg_weight.append(np.abs(pixel))

            # secondly, make the weights right
            data_weight.append(np.abs(pixel))

            # Finally, append the coordinates
            new_data.append(np.array([i,j]))

    if len(pos_weight)>0 and len(neg_weight)>0:
        pos_weight, neg_weight = np.array(pos_weight), np.array(neg_weight)
        pos_threshold = np.percentile(pos_weight, q=threshold)
        neg_threshold = np.percentile(neg_weight, q=threshold)
    else:
        valid_image = 0

    if valid_image==1:
        final_data = list(zip(new_data,data_weight,data_label))
        selected_data = []

        for item in final_data:
            if item[2]==1:
                if item[1]>=pos_threshold:
                    selected_data.append(item)
            if item[2]==-1:
                if item[1]>=neg_threshold:
                    selected_data.append(item)

        new_data, data_weight, data_label = zip(*selected_data)
        return np.array(new_data), np.array(data_weight), np.array(data_label), valid_image
    else:
        return [], [], [], valid_image

def KSVM_fit(pixel_coor, pixel_weight, pixel_polarity,valid_image,penalty_C=0.4):
    #pixel_coor, pixel_weight, pixel_polarity = KSVM_dataprep(image)
    if valid_image==1:
        classifier = SVC(C=penalty_C,kernel='rbf',probability=True,gamma='auto',random_state=40)
        classifier.fit(pixel_coor,pixel_polarity,sample_weight=pixel_weight)

        return classifier
    else:
        return valid_image

def KSVM_final(image,penalty_C=1.0,threshold=90,show=False,imagename='Nil'):
    # this function is the final version of Kernel SVM model, and the parameters are:
    # image: the image input
    # penalty_C: the margin parameter (the larger it is, the smaller margin the SVM model would use)
    # threshold: the percentile at which pixels are selected for doing the SVM

    # output: the PIL pixels for the image
    # print(imagename)
    pixel_coor, pixel_weight, pixel_polarity, valid_image = KSVM_dataprep(image, threshold=threshold)
    model = KSVM_fit(pixel_coor, pixel_weight, pixel_polarity, valid_image, penalty_C=penalty_C)

    if valid_image==0:
        return (imagename,np.array([]))
    else:
        # firstly, we should select a binding box in which we draw the PIL
        row_top = np.min(pixel_coor[:,0])
        row_bottom = np.max(pixel_coor[:,0])
        column_top = np.min(pixel_coor[:,1])
        column_bottom = np.max(pixel_coor[:,1])

        # secondly, we draw a subgrid inside
        x = np.linspace(row_top, row_bottom, num=500)
        y = np.linspace(column_top, column_bottom, num=500)

        xv, yv = np.meshgrid(x,y,indexing='xy')
        xv,yv = np.ndarray.flatten(xv), np.ndarray.flatten(yv)
        thegrid = np.array([xv,yv]).transpose()

        proba = model.predict_proba(thegrid)[:,0]
        boundary_point = [(int(thegrid[i,0]), int(thegrid[i,1])) for i in range(len(proba)) if proba[i]>=0.49 and proba[i]<=0.51]
        boundary_point = list(set(boundary_point))
        boundary_point = np.array(boundary_point)

        if show==True:
            fig = plt.subplots(nrows=1, ncols=2)

            plt.subplot(1, 2, 1)
            seaborn.heatmap(image, center=0, cbar=False)

            plt.subplot(1, 2, 2)
            # plt.contourf(x,y,Z,cmap=plt.cm.coolwarm)
            plt.scatter(boundary_point[:, 1], boundary_point[:, 0], color='red', marker='o', s=1)
            plt.xlim(0, image.shape[1])
            plt.ylim(0, image.shape[0])
            plt.gca().invert_yaxis()
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')

            plt.show()
        else:
            return (imagename,boundary_point)

def PIL(file,threshold=95,penalty_C=1.0):
    thefile = h5py.File(file,'r')
    video = thefile['video0']
    framelist = list(video.keys())
    frames = sorted(framelist, key=lambda x: int(x[5:]), reverse=False)
    images = [video[f]['channels'][:,:,2] for f in frames]
    process_input = []

    for i in range(len(frames)):
        process_input.append((images[i],penalty_C,threshold,False,frames[i]))

    pool = mp.Pool(processes=8)
    #funcs = partial(KSVM_final,penalty_C=1.0,threshold=90,show=False)
    results = [pool.apply_async(KSVM_final,t) for t in process_input]
    output = [p.get() for p in results]

    filename = './new_result/'+file[3:-5]+'_PIL'+'.hdf5'
    PILfile = h5py.File(filename,mode='w')
    PILfile.create_group(name='video0')
    newvideo = PILfile['video0']

    for i in range(len(output)):
        framename = output[i][0]
        newvideo.create_group(name=framename)
        newframe = newvideo[framename]
        newframe.create_dataset(name='PIL',data=output[i][1])

    PILfile.close()


if __name__=="__main__":
    filestart = str(sys.argv[1])
    total_files = glob.glob('../*.hdf5')
    for filename in total_files:
        if filename[7]==filestart:
            print(filename[3:-5])
            PIL(filename)

