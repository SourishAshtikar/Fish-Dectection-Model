import cv2
from ultralytics import YOLO

# Paths for videos and model
video_path = r"D:\YOLO Fish Project\code\videos\task1vid2.mp4"
model_path = r"D:\YOLO Fish Project\2000 image model\train3\weights\best.pt"  # Here used a custom trained YOLO model
video_out = f"{video_path[:-4]}_out.mp4"

# Load model and video
model = YOLO(model_path)  
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()
H, W = frame.shape[:2]
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

confidence = 0.3
print("Started Processing!")
# Dectection starts
while ret:
    resized = cv2.resize(frame, (640, 640))
    results = model(resized, verbose=False)[0]

    for x1, y1, x2, y2, score, class_id in results.boxes.data.tolist():
        if score > confidence:
            # Scale to original size
            x_scale, y_scale = W / 640, H / 640
            x1, x2 = x1 * x_scale, x2 * x_scale
            y1, y2 = y1 * y_scale, y2 * y_scale

            # Draw box and label
            label = f"{model.names[int(class_id)].upper()}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame,label, (int(x1), max(int(y1) - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done")