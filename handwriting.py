import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import pickle
import boto3
import math
import os
from random import randint

import environ
env = environ.Env()
environ.Env.read_env()

from django.conf import settings

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_storages import MediaStorage

s3 = boto3.resource('s3', aws_access_key_id=env('AWS_ACCESS_KEY_ID'),
         aws_secret_access_key= env('AWS_SECRET_ACCESS_KEY'))
bucket = s3.Bucket('**********')

media_storage = MediaStorage()

def load_keras_model(modeldir):
    """Load in the pre-trained model"""
    global model
    model = load_model(modeldir)
    # Required for model to work
    global graph
    graph = tf.compat.v1.get_default_graph()
    return model

def get_image_dir(image_name):
    ext = image_name.split('.')[-1]
    key = 'media/user_uploaded/' + str(image_name)
    imagepath = 'predict_image' + str(randint(1,1000)) + f'.{ext}' #cloudfront
    imagedir = os.path.join(settings.BASE_DIR, imagepath)
    print("image directory is:", imagedir)
    if not os.path.exists(imagedir):
        with open(imagepath, 'wb') as file:
            bucket.download_fileobj(key, file)
    return imagedir, imagepath

def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2

	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta

	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize

			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

def get_handwritten_words(image_name):
    modeldir = os.path.join(settings.BASE_DIR, 'handwriting_model.h5')
    if not os.path.exists(modeldir):
        with open('handwriting_model.h5', 'wb') as file:
            bucket.download_fileobj('/handwrite_take2.h5', file)
    model = load_keras_model(modeldir)

    #loading lables from pickle
    with open('label_classes', 'wb') as data:
        bucket.download_fileobj('label_classes', data)

    with open('label_classes', 'rb') as data:
        labels = pickle.load(data)

    imagedir, image_path = get_image_dir(image_name)

    #image processing
    #read image
    image = cv2.imread(imagedir)

    #prepare image and convert height
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = createKernel(91, 27, 19)

    #apply threshold and filter
    imgFiltered = cv2.filter2D(imgray, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    ret3,thresh = cv2.threshold(imgFiltered,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = 255 - thresh

    # find contours
    (components, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < 350:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = image[y:y+h, x:x+w]
        res.append((currBox, currImg))
        res = sorted(res, key=lambda entry:entry[0][0])

    for box,img in res:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_array = cv2.resize(imgray, (50, 20))
        array = new_array.reshape(-1, 50, 20, 1)
        pred = model.predict(array)
        y = np.argmax(pred)
        pred_word = labels[y]
        (x, y, w, h) = box
        cv2.rectangle(image,(x,y),(x+w,y+h),0,1)
        cv2.putText(image,pred_word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0),2,cv2.LINE_AA)
    cv2.imwrite(imagedir, image)

    filepath = 'predicted_images/' + image_path
    with open(image_path, 'rb') as f:
        media_storage.save(filepath, f)
    image_url = media_storage.url(filepath)

    return labels, image_url
