import os

os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "DEBUG"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|vcodec;h264"

import cv2
import tensorflow as tf


def detect_masks():
    #  camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture("#", cv2.CAP_FFMPEG)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    mask_classifier = tf.keras.models.load_model("wooly_mask_detector/models/maskclassifier.model", compile=False)

    while True:
        is_valid, frame = camera.read()

        if not is_valid:
            camera.release()
            camera = cv2.VideoCapture("#", cv2.CAP_FFMPEG)
            continue

        ratio = 1000.0 / frame.shape[1]
        dimension = (1000, int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for x, y, h, w in faces:
            face_roi = frame[y : y + h, x : x + w, :]
            face_roi = cv2.resize(face_roi, (160, 160))
            face_roi = tf.keras.preprocessing.image.img_to_array(face_roi)
            face_roi = face_roi.reshape(1, 160, 160, 3)

            prediction = mask_classifier(face_roi)
            (without_mask, with_mask) = prediction[0].numpy()

            (label, color, prob) = ("Mask", (0, 255, 0), with_mask * 100.0) if with_mask > without_mask else ("No mask", (0, 0, 255), without_mask * 100.0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x + 15, y + 2), (x + w - 15, y + 20), (0, 0, 0), -1)  # lower text box
            cv2.rectangle(frame, (x + 15, y - 2), (x + w - 15, y - 20), (0, 0, 0), -1)  # upper text box

            cv2.putText(frame, str(prob) + " %", (x + 20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.putText(frame, label, (x + 20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.imshow("LIVE", frame)

        key = cv2.waitKey(10)
        if key == 27:  # The Escape Key
            break

    camera.release()
    cv2.destroyAllWindows()
