import numpy as np
from keras.models import load_model
import cv2
import face_recognition
import argparse

# get a pretrained model
model = load_model("model/model_v6_23.hdf5")
emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}


def detect_emotion(path):
    """
    Predict face emotion using pretrained model.
    Works on a single face.

    :param path: str: Image path
    :return: str:     Emotion on face
    """

    # load and detect a face in a image
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]

    # resize and change a colors for better prediction
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v, k) for k, v in emotion_dict.items())
    predicted_label = label_map[predicted_class]

    return predicted_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect face emotion, from a given image.")
    parser.add_argument("path", help="Image path", type=str)
    args = parser.parse_args()

    print("*"*15)
    print(f"\nI'm a bad predictor, but my guess is that on this image is captured a {detect_emotion(args.path)}")
