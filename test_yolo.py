
import cv2
from ultralytics import YOLO

model = YOLO("runs/attention/hands_phones_v3/weights/best.pt")

print("Model classes:", model.names)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = model.predict(
        frame,
        conf=0.4,
        imgsz=640,
        max_det=100,
        verbose=False
    )[0]

    hand_count = 0
    phone_count = 0
    hand_scores = []
    phone_scores = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        name = result.names[cls_id]
        confidence = float(box.conf.item())

        if name == "hand":
            hand_count += 1
            hand_scores.append(confidence)

        elif name == "cell phone":
            phone_count += 1
            phone_scores.append(confidence)

    annotated = result.plot()

    cv2.putText(
        annotated,
        f"Hands: {hand_count} | Phones: {phone_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    best_hand = max(hand_scores, default=0.0)
    best_phone = max(phone_scores, default=0.0)

    cv2.putText(
        annotated,
        f"Hand conf: {best_hand:.3f} | Phone conf: {best_phone:.3f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )

    cv2.imshow("YOLO Diagnostic", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
