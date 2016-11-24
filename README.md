# Semi-Automatic 3D Hand Pose Annotation

Author: Markus Oberweger <oberweger@icg.tugraz.at>

## Requirements:
  * OS:
    - Ubuntu 14.04
  * via Ubuntu package manager:
    * python2.7
    * python-matplotlib
    * python-scipy
    * python-pil
    * python-numpy
    * python-vtk6
    * python-opencv2
    * python-qt4
    * python-pip
  * via pip install:
    * progressbar
    * psutil
    * theano (0.8)

For a description of our method see:

M. Oberweger, G. Riegler, P. Wohlhart, and V. Lepetit.  Efficiently Creating 3D Training Data for Fine Hand Pose Estimation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

## Setup:
  * Read the included manual on how to use our method
  * Put dataset files into ./data
  * Download [SIFTFlow](people.csail.mit.edu/celiu/SIFTflow/) and put it into ./src/etc/sift_flow
  * Goto ./src and see the main file main_blender_semiauto.py how to handle the API
  * Goto ./src and see the main files main_labeling_pose.py and main_labeling_detect.py how to do the annotation for your own dataset
