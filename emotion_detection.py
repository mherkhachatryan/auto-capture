import numpy as np
from keras.models import load_model
import cv2
from os import listdir
from os.path import isfile, join


predicted_classes = {}
filenames = [f for f in listdir('AutoCaps/') if isfile(join("AutoCaps/", f))]

emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
model = load_model("model/model_v6_23.hdf5")

for face in filenames:
    face_image = cv2.imread(f"AutoCaps/{face}")

    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    predicted_class = np.argmax(model.predict(face_image))

    label_map = dict((v, k) for k, v in emotion_dict.items())
    predicted_label = label_map[predicted_class]

    predicted_classes[face] = predicted_label


print(predicted_classes)
print("*"*15)
# print(predicted_classes["manual_capture_9.png"])
# for key, value in predicted_classes.items():
#     if value == "Happy":
#         print(key, " is happy picture")