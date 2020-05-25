from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np


def smile(mouth: tuple):
    """
    Compute Mouth Aspect Ration (MAR)

    :param mouth: (in, int) : (x, y) coordinates of the facial landmark for a mouth
    :return: float: MAR
    """
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    MAR = avg / D
    return MAR


def blink(eye: tuple):
    """
    Compute Eye Aspect Ration (EAR), dividing average euclidean distances of 2 points
    on upper and lower parts of eye on euclidean distance of rightmost and leftmost points of the eye

    :param eye: (in, int) : (x, y) coordinates of the facial landmark for a given eye
    :return: float:  EAR
    """

    # vertical landmarks of the eye
    vertical1 = dist.euclidean(eye[1], eye[5])
    vertical2 = dist.euclidean(eye[2], eye[4])

    # horizontal landmark of the eye
    horizontal = dist.euclidean(eye[0], eye[3])

    # calculate eye aspect ration
    EAR = (vertical1 + vertical2) / (2 * horizontal)

    return EAR


COUNTER = 0
TOTAL = 0

shape_predictor = "model/shape_predictor_68_face_landmarks.dat"  # dace_landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# get facial landmarks' coordinates
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

# print("[INFO] starting video stream thread...")

video_stream = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

# fps = FPS().start()
cv2.namedWindow("auto-capture")

while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=840)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mouth_start:mouth_end]
        MAR = smile(mouth)

        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        left_EAR = blink(left_eye)
        right_EAR = blink(right_eye)
        EAR = (left_EAR + right_EAR) / 2
        mouth_hull = cv2.convexHull(mouth)
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)

        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [left_eye], -1, (214, 16, 232), 1)
        cv2.drawContours(frame, [right_eye], -1, (214, 16, 232), 1)

        if (MAR <= .3 or MAR > .38) and (EAR >= 0.28):
            COUNTER += 1
        else:
            if COUNTER >= 1:
                TOTAL += 1
                frame = video_stream.read()
                time.sleep(.004)
                img_name = f"auto_capture_frame_{TOTAL}.png"
                cv2.imwrite(img_name, frame)
                print(f"{img_name} written!")
            COUNTER = 0

        cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"LEFT EAR: {left_EAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (214, 16, 232), 2)
        cv2.putText(frame, f"RIGHT EAR: {right_EAR:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (214, 16, 232), 2)

    cv2.imshow("Frame", frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q') or key_pressed == 27:
        break
    elif key_pressed == 13 or key_pressed == 32:
        frame = video_stream.read()
        time.sleep(.004)
        img_name = f"manual_capture_{TOTAL}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        TOTAL += 1
        continue


cv2.destroyAllWindows()
video_stream.stop()
