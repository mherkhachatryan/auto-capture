from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2




image1 = Image.open("data/image.jpg")
image_array1 = np.array(image1)
