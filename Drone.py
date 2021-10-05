import cv2
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def extract_face(filename, required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def img_to_encoding(image_path, model):
    img = extract_face(image_path)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)
    dist = cosine(encoding, database[identity])
    if dist < 0.5:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    return dist, door_open

database = {}
database['Surya'] = img_to_encoding('C:\\Users\\SuryaPrakash\\Desktop\\Drone project\\WhatsApp Image 2021-10-02 at 10.43.43 AM.jpeg',model)
print(database)