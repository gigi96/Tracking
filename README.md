# About the project
This project is born as a possible solution to the project *Detection and Tracking* of the course *Fundamentals of Computer Vision & Image Processing in Python* of **OpenCV University**.
It tries to implement a *detection+tracking pipeline* with the **OpenCV** library; here is the [link](https://www.youtube.com/watch?v=qoEG_wnp8AE) of the original description.

In order to make the project more *realistic* and *professional*, I used a GUI library to make the entire layout of the application

# Built With
- python 3.12.1
- numpy 1.26.3
- opencv-python 4.9.0.80
- pysimplegui 4.60.5

# Description
The application implements a *detection+tracking pipeline*: the target is a soccer ball in a video. 

The pipeline starts with the detection of the target; after that, the tracking algorithm starts to run, trying to follow the previously detected object.
If the tracking algorithm failed, the pipeline automatically restarts with the detection and so on.
If the user recognizes that the tracking algorithm gives false positives, he can pause the video to restart the pipeline from the detection phase.

Detection is performed with **YOLOv3**: the deep-neural-network is provided by the official YOLO repository of Ultralytics (not provided in this repository). 
The application limits only to execute the inference of the model on every frame of the video to the class of object of interest (i.e. the soccer ball).

Tracking algorithm can be chosen by the user from a list exposed in the main window.

# Future development
As can be seen after running the application, the entire tracking performs not very well. The reason is due to the follwing drawbacks: detection inference is slow (it takes about 0.5 seconds),
inhibiting a smoothly execution of the pipeline; the provided tracking algorithms have low performances.

This obviously is not the state of the art of computer vision tracking: therefore, a future development can be to improve the performances using new algorithms.
