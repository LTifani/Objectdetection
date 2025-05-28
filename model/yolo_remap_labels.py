
import os

# Mapping ALT-ID → NEU-ID (aus vollständiger Klasse zu reduziertem Mapping)
class_map = {
    0: 6, 1: 6, 2: 6, 3: 0, 4: 6, 5: 6, 6: 2, 7: 3, 8: 6, 9: 1, 10: 6, 11: 6,
    12: 6, 13: 6, 14: 6, 15: 6, 16: 6, 17: 6, 18: 4, 19: 6, 20: 6, 21: 5,
    22: 6, 23: 6, 24: 6, 25: 6, 26: 6
}

# Eingabe-Ordner der YOLO-Labels
LABEL_DIR = "C:\\MASTER\\Semester 2\\PROKEKT\\Oblique-instance-segmentation.v1i.yolov11\\train\\labels"  # <- ANPASSEN!
SAVE_BACKUP = True  # Setze auf False, wenn du keine _old.txt behalten willst

for filename in os.listdir(LABEL_DIR):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, filename)
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_id = int(parts[0])
        new_id = class_map.get(old_id, -1)
        if new_id == -1:
            continue
        new_line = " ".join([str(new_id)] + parts[1:])
        new_lines.append(new_line)

    if SAVE_BACKUP:
        os.rename(path, path.replace(".txt", "_old.txt"))

    with open(path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print("Alle YOLO-Labeldateien wurden erfolgreich konvertiert.")
