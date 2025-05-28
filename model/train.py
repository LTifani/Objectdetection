
from ultralytics import YOLO
import os

# === KONFIGURATION ===
YAML_PATH = "data.yaml"  # <- Pfad zu deinem Dataset YAML
HYP_PATH = "custom_hyp_segmentation.yaml"  # <- Pfad zu deiner Hyp-Datei
EXPERIMENT_NAME = "seg_model_augmented"
PROJECT_NAME = "yolo_seg_training_mapped"
EPOCHS = 50
BEST_MODEL_PATH = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, "weights", "best.pt")
TEST_IMAGE_PATH = "yolo_dataset_mapped/test/Am-Bachberg2_60435Frankfurt-am-Main_N_png.rf.248861a909c883de787f6ee65cab9fe2.jpg"

# === TRAINING ===
print("🔧 Starte Training...")
model = YOLO("yolov8n-seg.pt")
results = model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=640,
    batch=8,
    lr0=0.01,
    weight_decay=0.0005,
    degrees=5.0,
    scale=0.3,
    shear=1.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.2,
    patience=20,
    name=EXPERIMENT_NAME,
    project=PROJECT_NAME
)

# === EVALUATION ===
print("\nEvaluierung des besten Modells auf dem Validierungsset:")
best_model = YOLO(BEST_MODEL_PATH)
metrics = best_model.val(data=YAML_PATH)
print(metrics)

# === TESTBILD-INFERENZ ===
if os.path.exists(TEST_IMAGE_PATH):
    print("\n🧪 Testbild wird verarbeitet...")
    results = best_model(TEST_IMAGE_PATH, save=True, save_txt=True, save_conf=True)

    # Anzeige der Maske
    if results[0].masks is not None:
        print(f"🎯 {len(results[0].masks.data)} Masken erkannt.")
    else:
        print("Keine Maske erkannt.")

    # Bild mit Maske wird automatisch als 'predict' gespeichert
    print("Ergebnisbild mit Maske gespeichert im Ordner 'runs/segment/predict/'")

else:
    print("Testbild nicht gefunden! Bitte TEST_IMAGE_PATH anpassen.")
