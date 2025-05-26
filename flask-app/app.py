from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import requests
import base64
from model import run_inference  # deine eigene Modellfunktion

# Optional: Geolocation-Import (später implementierbar)
# from geolocation import georeference_bboxes

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.form
    file = request.files.get('image')
    
    # ======= 1. BILDQUELLE (Upload oder Vexcel) =======

    # Option 1: Bild wurde hochgeladen
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        image_path = filepath

    # Option 2: Bild von Vexcel API abrufen
    elif 'lat' in data and 'lon' in data:
        lat = data['lat']
        lon = data['lon']
        direction = data.get('direction', 'north')  # z. B. north, south, east, west

        # Beispielhafter Vexcel API Call (Dummy URL)
        vexcel_url = f"https://vexcel.api/get_image?lat={lat}&lon={lon}&direction={direction}"
        response = requests.get(vexcel_url)

        if response.status_code != 200:
            return jsonify({"error": "Fehler beim Abrufen des Bildes von Vexcel"}), 500

        image_path = os.path.join(UPLOAD_FOLDER, f"{lat}_{lon}_{direction}.jpg")
        with open(image_path, 'wb') as f:
            f.write(response.content)

    else:
        return jsonify({"error": "Bitte entweder ein Bild hochladen oder Koordinaten angeben"}), 400

    # ======= 2. MODELL-INFERENZ =======

    try:
        result = run_inference(image_path)  # enthält Bounding Boxes, Klassen, Konfidenz

       # ======= 3. GEOREFERENZIERUNG =======
       

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
