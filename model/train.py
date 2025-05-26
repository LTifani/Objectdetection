
from ultralytics import YOLO
import os

# === KONFIGURATION ===
YAML_PATH = "data.yaml"
EXPERIMENT_NAME = "oblique-yolo-model"
PROJECT_NAME = "yolo_training"
EPOCHS = 100
BEST_MODEL_PATH = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, "weights", "best.pt")
TEST_IMAGE_PATH = "yolo_dataset\\test\\Am-Bachberg2_60435Frankfurt-am-Main_S_png.rf.c50f2cf981e05bb37edd60562db8374f.jpg"  

# === TRAINING MIT EARLY STOPPING & LOGGING ===
model = YOLO("yolov8n.pt")
results = model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=640,
    batch=8,
    dropout=0.1,
    name=EXPERIMENT_NAME,
    project=PROJECT_NAME,
    augment=True,
    patience=20,             # Early Stopping
    verbose=True
)

# === EVALUATION AUF VALIDIERUNGSDATEN ===
print("\nEvaluierung des Modells auf dem Validierungsset:")
metrics = model.val(data=YAML_PATH)
print(metrics)

# === INFERENZ AUF EINEM TESTBILD ===
if os.path.exists(TEST_IMAGE_PATH):
    print("\nTestbild wird verarbeitet...")
    trained_model = YOLO(BEST_MODEL_PATH)
    results = trained_model(TEST_IMAGE_PATH)

    # Anzeige des Bildes mit Bounding Boxes
    results[0].show()

    # Speichern des Bildes mit Bounding Boxes
    results[0].save("output_image.jpg")
    print("Ergebnisbild gespeichert als output_image.jpg")

    # Details zu erkannten Objekten ausgeben
    for result in results:
        for box in result.boxes:
            print(f"Klasse: {box.cls}, Konfidenz: {box.conf:.2f}, Bounding Box: {box.xyxy}")
else:
    print("Kein Testbild gefunden! Bitte TEST_IMAGE_PATH anpassen.")
