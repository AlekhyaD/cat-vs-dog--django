from django.shortcuts import render

# Create your views here.

from tensorflow import keras
from keras.models import load_model
import numpy as np
#import cv2
from keras.preprocessing import image
from PIL import Image

from django.shortcuts import render
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
global graph,model
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
import io

model = load_model('catvsdog_classification_app/dogvscat.model')
model.summary()

class_dict = {'cat': 0, 'dog': 1}

class_names = list(class_dict.keys())

def prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        post = request.method == 'POST'
        myfile = request.FILES['myfile']

        # dimensions of our images.
        #img = Image.open(myfile)
        #img_width, img_height = 150, 150
        #img = image.load_img(myfile, target_size = (img_width, img_height))
        #img = image.img_to_array(img)
        #img = np.expand_dims(img, axis = 0)
        img = Image.open(myfile)
        print("**************************************************************")
        print(type(img))
        #img = img.convert('RGB')
        img = img.resize((150, 150))
        img = image.img_to_array(img)
        #img = image.load_img(myfile, target_size=(224, 224))
        #img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        #img = img/255

        predictions = model.predict(img)

        print(predictions)


        if predictions == 0:
            predictions = 'Cat'
        elif predictions == 1:
            predictions = 'Dog'
        print( "Prediction completed: this is a", predictions)
        
        return render(request, "catvsdog_classification_app/prediction.html", {
            'result': predictions})
    else:
        return render(request, "catvsdog_classification_app/prediction.html")

'''
        img = cv2.imread('myfile')
        img = cv2.resize(img, (150, 150))
        img = np.reshape(img, [1, 150, 150, 3])


        predictions = model.predict_classes(img)
        preds = preds.flatten()
        m = max(preds)
        for index, item in enumerate(preds):
            if item == m:
                result = class_names[index]
        return render(request, "catvsdog_classification_app/prediction.html", {
            'result': result})
    else:
        return render(request, "catvsdog_classification_app/prediction.html")
'''

'''
        if predictions == 0:
            predictions = 'Cat'
        elif predictions == 1:
            predictions = 'Dog'
        print( "Prediction completed: this is a", predictions)

        return render(request, "catvsdog_classification_app/prediction.html", {
            'result': predictions})
    else:
        return render(request, "catvsdog_classification_app/prediction.html")
'''