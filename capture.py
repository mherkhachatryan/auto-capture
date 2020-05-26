from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import argparse

shape_predictor = "model/shape_predictor_68_face_landmarks.dat"  # face_landmark

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# get facial landmarks' coordinates
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]


def smile(mouth):
    """
    Compute Mouth Aspect Ration (MAR), by dividing mean distance of vertical
    points on mouth over horizontal distance on rightmost and leftmost points

    :param mouth: (in, int) : (x, y) coordinates of the facial landmark for a mouth
    :return: float:           MAR
    """
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    MAR = avg / D
    return MAR


def blink(eye):
    """
    Compute Eye Aspect Ration (EAR), dividing average euclidean distances of 2 points
    on upper and lower parts of eye on euclidean distance of rightmost and leftmost points of the eye

    :param eye: (in, int) : (x, y) coordinates of the facial landmark for a given eye
    :return: float:         EAR
    """

    # vertical landmarks of the eye
    vertical1 = dist.euclidean(eye[1], eye[5])
    vertical2 = dist.euclidean(eye[2], eye[4])

    # horizontal landmark of the eye
    horizontal = dist.euclidean(eye[0], eye[3])

    # calculate eye aspect ration
    EAR = (vertical1 + vertical2) / (2 * horizontal)

    return EAR


def auto_capturing(auto_path, man_path, mar_threshold, ear_threshold, frame_waiter, verbose):
    """
    Creates a cv2 window which opens a web camera and depending on conditions (eyes are open and smile)
    take a snapshot, also available manual shots.

    :param auto_path: str :                 Directory name where automatically captured photos will be stored
    :param man_path: str :                  Directory name where manual captured photos will be stored
    :param mar_threshold:  (float, flaot) : Mouth Aspect Ratio threshold, (min, max):
                                            to control smiling threshold, default numbers are computed
                                            from trial and error, and may differ on
                                            different mouth shapes. min controls smiling
                                            without teeth, max controls smiling with teeth
    :param ear_threshold: float :           Eye Aspect Ratio threshold, if EAR is great or
                                            great or equal this number, it means eyes are open,
                                            and ready to a shot
    :param frame_waiter: float :            Number of frames that camera should wait before taking a snapshot
    :param verbose: bool:                   Show contours around facial features and MAR and EAR values
    :return:                                None
    """
    COUNTER = 0  # frame counter
    TOTAL = 0  # picture counter to update filename after each shot
    video_stream = VideoStream(src=0).start()
    fileStream = False
    time.sleep(1.0)
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

            if (MAR <= mar_threshold[0] or MAR > mar_threshold[1]) and (EAR >= ear_threshold):
                COUNTER += 1
            else:
                if COUNTER >= frame_waiter:
                    TOTAL += 1
                    frame = video_stream.read()
                    # time.sleep(.05)
                    img_name = f"auto_capture_frame_{TOTAL}.png"
                    cv2.imwrite(auto_path + img_name, frame)
                    print(f"{img_name} written!")
                COUNTER = 0

            if verbose:
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [left_eye_hull], -1, (214, 16, 232), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (214, 16, 232), 1)

                cv2.putText(frame, f"MAR: {MAR:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, f"LEFT EAR: {left_EAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (214, 16, 232),
                            2)
                cv2.putText(frame, f"RIGHT EAR: {right_EAR:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (214, 16, 232), 2)

        cv2.imshow("Frame", frame)

        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed == ord('q') or key_pressed == 27:
            break
        elif key_pressed == 13 or key_pressed == 32:
            frame = video_stream.read()
            time.sleep(.004)
            img_name = f"manual_capture_{TOTAL}.png"
            cv2.imwrite(man_path + img_name, frame)
            print(f"{img_name} written!")
            TOTAL += 1
            continue

    cv2.destroyAllWindows()
    video_stream.stop()


parser = argparse.ArgumentParser(description="Apps which captures automatic photos when user is smiling")
parser.add_argument("-a", "--auto_path", default='AutoCaps/', help="Directory name where \
                                                        automatically captured photos will be stored", type=str)
parser.add_argument("-m", "--man_path", default="ManCaps/", help="Directory name where \
                                                           manual captured photos will be stored", type=str)
parser.add_argument("--eye", default=0.21, help="Eye Aspect Ratio threshold, if EAR is great or  \
                                                great or equal this number, it means eyes are open, \
                                                and ready to a shot", type=float)
parser.add_argument("--mouth", help="Mouth Aspect Ratio threshold, (min, max):  \
                                   to control smiling threshold, default numbers are computed \
                                   from trial and error, and may differ on \
                                   different mouth shapes. min controls smiling \
                                   without teeth, max controls smiling with teeth", default=(0.3, 0.35), type=tuple)
parser.add_argument("--frame_waiter", help="Number of frames that camera should wait \
                                            before taking a snapshot", default=5, type=int)
parser.add_argument("-v", "--verbose", help="Show contours around facial features \
                                             and MAR and EAR values", default=False, action="store_true")
args = parser.parse_args()

auto_capturing(auto_path=args.auto_path, man_path=args.man_path, mar_threshold=args.mouth, ear_threshold=args.eye,
               frame_waiter=args.frame_waiter, verbose=args.verbose)
