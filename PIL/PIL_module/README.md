This folder contains the code for the PIL python module.

Some references:

1. Stanford lab's coding on feature calculation based on magnetic field data:
https://github.com/mbobra/calculating-spaceweather-keywords/blob/master/calculate_swx_fits.py

2. Online tutorial drms module in python:
https://drms.readthedocs.io/en/stable/tutorial.html

3. Ipython notebook tutorial on retrieving header and image data from HMI and AIA source using drms, including d3 plot tutorial on mpld3:
https://nbviewer.jupyter.org/github/mbobra/calculating-spaceweather-keywords/blob/master/plot_swx_d3.ipynb#plotting-the-image-data

4. Stanford group's data on more localized features(yliu's group):
http://sun.stanford.edu/~yliu/Michigan/SHARP_parameters/

5. Bobra's paper on SHARP data:
https://link.springer.com/article/10.1007%2Fs11207-014-0529-3

6. Connected-Components Labeling (can be implemented in openCV), method used to connect a single polarity inversion line
https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
https://en.wikipedia.org/wiki/Connected-component_labeling

7. non-maxima suppression, method used to "thin" the polarity inversion line
https://en.wikipedia.org/wiki/Canny_edge_detector
and a simple python implementation looks like https://github.com/sebasvega95/Canny-edge-detector/blob/master/nonmax_suppression.py

8. A more detailed tutorial on canny edge detection
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

9. Paper on defining GWILL and polarity inversion line algorithm discussion based on Bokenkamp(2007) senior thesis at Stanford
https://iopscience.iop.org/article/10.1088/0004-637X/723/1/634/pdf

10. Important! There is a problem about data_preparation method. The output of the method would push some pixels to 0, and this enables some pixels which do not have both a local maxima and a local minima to have a large gradient, if its neighbor happens to have both local maxima and local minima. See the problem with the help of the first image of the test data, pixel(293,440), which should not be considered as a PIL candidate. (09/17,make change soon!)
