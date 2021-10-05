import cv2
import numpy as np
from scipy.spatial.distance import cosine
from keras.models import load_model
from PIL import Image
database = {}
model = load_model('C:\\Users\\SuryaPrakash\\Downloads\\Facemodel.h5')
model.load_weights('C:\\Users\\SuryaPrakash\\Downloads\\Weights.h5')

def extract_face(img, required_size=(224, 224)):
    wholeimg = img.copy()
    facecascade = cv2.CascadeClassifier('C:\\Users\\SuryaPrakash\\Desktop\\Drone project\\Computer-Vision-with-Python\\DATA\\haarcascades\\haarcascade_frontalface_default.xml')
    rect_cd = facecascade.detectMultiScale(img)
    for (x,y,w,h) in rect_cd:
        face = wholeimg[y:y+h, x:x+w]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def img_to_encoding(img, model):
    if type(img) == str:
        img = cv2.imread(img)
    else:
        pass
    try:
        img = extract_face(img)
    except:
        img = cv2.resize(img, (224, 224))

    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def verify(img, identity, database, model):
    encoding = img_to_encoding(img, model)
    dist = cosine(encoding, database[identity])
    return dist

