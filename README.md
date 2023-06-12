# Eye_Tracking_with_Deep_CNN
A deep convolutional neural network implementation for tracking eye movements in videos

A tensorflow/keras model is developed here to track eye motions in videos. 
The model is trained, validated and tested with Labeled Pupils in the Wild (LPW) dataset:
https://perceptualui.org/research/datasets/LPW/

![image](https://user-images.githubusercontent.com/40482921/234497211-f76d938e-45b3-4c86-a4e5-c013bbd53c55.png)

The repository consists of 4 Python scripts

1) selectTrainTestValData.py randomly selects videos for training, validation and, test sets

2) getMeanIntensity.py scans the training data and finds the mean intensity for all the frames in all of the videos

3) vid2Frames.py extracts frames from videos, resizes, converts them to grayscale, and saves them in train, validation and test folders

4) trainModel.py forms a CNN model, forms train, test and, validation sets and trains the model with selected parameters. Finally, it measures the accuracy of the model using the test set.

To use the scripts, download the LPW dataset to your local and adjust folder locations in the scripts. 

Utilizing GPU is recommended. A single epoch with CPU may take over an hour to complete.  

Recommended configuration is: Python 3.8, Tensorflow 2.10.0, CUDA 11.2, CUDNN 8.8.1 and Zlib.

![eye_tracker_2](https://user-images.githubusercontent.com/40482921/235327664-1a9024f0-cfaa-4776-a058-c03385fd02c8.PNG)

![eye_tracker](https://user-images.githubusercontent.com/40482921/235327674-7dff69a1-2a11-44fb-86b2-851c6124a26e.PNG)
