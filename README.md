# Predicting Celebrity Faces

### Table of Contents:
1. EDA_Preprocessing 
    - Notebook containing my EDA and preprocessing code
2. Model
    - Two notebooks containing the models that were run on AWS
3. Evaluation
    - Notebook containing my model's predictions
4. Python_scripts
    - Script to find faulty images that are in dataset and writes file path to a text file
    - Script to find images where face detector can’t find a face to crop and writes file path to a text file
    - Script to take multiple pictures through the webcam
    - Script to make real time predictions of faces through the webcam
    - Text file for AWS command lines to upload and download folders and files
5. Cascades
    - OpenCV library of face cascades that I used to detect the faces in my images
6. Visuals
    - Images for README

### Prerequisites
Libraries/packages:
- TensorFlow 
- Keras 
- Scikit-learn
- RegEx - https://pypi.org/project/regex/
- OpenCV - https://pypi.org/project/opencv-python/
    
All of these external libraries and packages are required in order to run my code. OpenCV will mainly be used for their face cascade, face detector and video capture.

My computer is not computationally efficient and can't run GPU supported tensorflow, so I had to set up an Amazon EC2 instance to run my models. For this I followed this guide: https://towardsdatascience.com/boost-your-machine-learning-with-amazon-ec2-keras-and-gpu-acceleration-a43aed049a50

### Problem Statement
There is a lot of visual data out in the world and it is important that we are able to utilize and interpret this data. This project is a baseline direction towards computer vision with deep learning techniques. My goal is to accurately predict the correct name of the celebrity in a given set of images. Another academic research has conducted a similar experiment and they were able to accurately recognize 44% of the images. I would consider this project to be successful if I am able to achieve close to five percent of their score or exceed it.

In this case, I used the faces of celebrities because the data is more attainable and easier to demonstrate. In the real world, if the data is accessible, everyone will be able to use this technology and we could provide efficiency and convenience. An example would be the recent iPhone models that recognize your face to unlock the phone. But ideally speaking we should be able to substitute a lot of our old methods with facial recognition. For instance, using our face to buy train tickets, unlock the door to our building, or enhancing security systems for law enforcement. The numerous opportunities for implementation are endless and worth exploring.

### Data Collection
There are a lot of different websites out there that provide image datasets of celebrities. Most of these websites don't give a large enough sample size to work with, usually between 10 to 200 images per celebrity. Thankfully I found a dataset from a github repo where the contributor scrapped the images from Google and had around 700 to 800 images for each of the 1100 celebrities. 

Dataset collected from: https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset

### EDA and Preprocessing
Originially I decided to include ten celebrities in my model, however, I decided to work with a smaller dataset first with five celebrities plus myself and a friend. Once I have a working model, then more people will be added. 

I labeled my image folders a certain way so that I can use the class name and class label for my predictions. This is how I set up my data folders:
![Data Folder Outline](https://github.com/BenjaminRCho/Celebrity-Face-Recognition/tree/master/6_visuals/img7.png)

First of all, my required dataset has to have as many frontal face images of the five celebrities that I am trying to predict. The initial approach after gathering the data is to make sure that there is only one face in each image, this way we can assure that OpenCV’s face detector will capture the face we want. This means that the images have to be clean with the correct celebrity face and with no other noise (i.e. any other face) in them.

Here are some examples of images that are considered messy data:
![Messy Data](https://github.com/BenjaminRCho/Celebrity-Face-Recognition/tree/master/6_visuals/img1.png)

After cleaning all the images in my dataset, I plotted to view them in a single color channel and to look at just linear features in the images. 
![Green Color Channel Only](https://github.com/BenjaminRCho/Celebrity-Face-Recognition/tree/master/6_visuals/img2.png)
![Linear Features Only](https://github.com/BenjaminRCho/Celebrity-Face-Recognition/tree/master/6_visuals/img3.png)

The next step was to use OpenCV because it has a prebuilt face detector that will help in cropping each of the faces. I would like to use the Dlib library to capture the facial landmarks myself in the future, however, for the sake of time I used OpenCV’s face cascades. The face detector required some fine tuning to find the optimal setting because sometimes it would capture locations on the image that didn't have a face. Now, I had to resize the cropped images to the width and height of my choice (175 by 175) so that all the data dimensions will be the same because all my images were different sizes. This is important for feeding the data into my model otherwise it won't train. Once the faces are cropped, resized and saved, my dataset was complete at this point and ready to be separated into my X (i.e. image arrays) and y (i.e. image labels). 

### Modeling
The model I built was a convolutional neural network (CNN) with eight layers followed by a fully connected layer. The architecture of the CNN consists of two pairs of convolutional layers with two subsampling layers after each pair. I built two convolutional layers after each other in order to capture more feature out of each image before max pooling. In the end, the model had about 60 million parameters to train on, which was relatively fast computationally. 

Just like any other model, there are a few parameters to tweak around with so that the model will perform better. One important parameter that I included in my CNN layers is having padding equal the same so that the edge features are better captured and to have the output size the same dimensions as the input image. Another customization that helped in improving the accuracy was decreasing the adam optimizer’s learning rate so that the model will learn slower. Also, my dataset size is on the smaller end of the scale so using Keras’ image data generator augments the data by randomly distorting the images to provide more sampling data. 

### Evaluation
All in all, after training the model I was able to obtain an accuracy score of 98%. Although the dataset only includes images of seven people, the model should run successfully for other people so long as there is enough data to be trained on. 

Here are the loss and accuracy curves of my model:
![Loss Curve](https://github.com/BenjaminRCho/Celebrity-Face-Recognition/tree/master/6_visuals/img4.png)
![Accuracy Curve](https://github.com/BenjaminRCho/Celebrity-Face-Recognition/tree/master/6_visuals/img5.png)

Although my model had such high accuracy, it still makes a few false predictions with the test images that weren't included in training set. When I run the model through a webcam script to make real time predictions, it accurately predicts the faces most of the time but it can still be more precise. I have tried fine tuning my model parameters multiple times but it still seems to be overfitting. One way to solve this would be to gather at least double the amount of data I currently have, if not more, to make it more accurate.

Predictions on test images:
![Predictions](https://github.com/BenjaminRCho/Celebrity-Face-Recognition/tree/master/6_visuals/img6.png)

When I added more people to the model, the performance dropped dramatically. This furthers my belief that there needs to be a substantial amount of face images so that the model can better learn the unique features of each individual. I really learned that the more clean data you have the better a face recognition model will perform. Thus, in order to implement such model in the real world, we would need to continuously feed the model with more and more images for better classification.

### Future Exploration
Practically speaking, no one would use an application that makes bad predictions all the time. One limitation of training a neural network is that it tends to overfit when there is not enough data. To further improve this project and the precision of the model, I would love to collect more data on each person. A possible method would be to go through interview videos of celebrities and collect frame by frame pictures of their faces. Perhaps, structure my model a different way or make it more complex. Furthermore, I would like to use the library dlib as one way to increase the accuracy of detecting faces properly by grabbing the feature points on each face. 


