import cv2
import get_features as gf
import draw_features as df
from ultralytics import YOLO
import mediapipe as mp
from dotenv import load_dotenv

load_dotenv()

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error starting webcam")

# YOLO detectors
detModel   = YOLO("/models/yolo/yolo26n.pt")

# BlazeFace detector
modelPath = "./models/landmarker/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=modelPath),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)

frameCount = 0
fps = 30

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = capture.read()

        timeStamp = int(frameCount * 1000 / fps)
        frameCount += 1

        features = gf.getFeatures(frame=frame, timeStamp=timeStamp, detModel=detModel, landmarker=landmarker)
        df.drawFeatures(frame, features)
        
        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()