from ultralytics import YOLO

# YOLOv8 Modell laden
model = YOLO("./best.pt")

def run_inference(image_path):
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "x": int(box.xyxy[0][0]),
                "y": int(box.xyxy[0][1]),
                "w": int(box.xyxy[0][2] - box.xyxy[0][0]),
                "h": int(box.xyxy[0][3] - box.xyxy[0][1]),
                "confidence": float(box.conf[0])
            })
    return {"boxes": detections}
