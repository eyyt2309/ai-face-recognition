import cv2
import get_features as gf
import create_models as cm
import time
from dotenv import load_dotenv

load_dotenv()

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error starting webcam")

# create detectors
fp_detector = cm.createFPDetector()
h_detector = cm.createHDetector()


while True:
    ret, frame = capture.read()

    timestamp_ms = int(time.time() * 1000)

    features = gf.getFeatures(frame, timestamp_ms, fp_detector, h_detector)
    
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()