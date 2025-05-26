Projektstruktur anlegen
projekt/
├── app.py                  # Haupt-Flask-API
├── model.py               # Modellinferenz (YOLOv8)
├──         
├── uploads/               # Ordner für gespeicherte Bilder-.Möflichkeit vllt Bilder per Upload zu machen 
├── requirements.txt       # Abhängigkeiten
└── best.pt                # Dein trainiertes YOLOv8-Modell

1.Abruf über Vexcel API im App machen
    lat, lon, direction 
    
2.Modell einbinden
3.Model Output verarbeiten --> App output
       Ausgabe ist ein JSON mit allen erkannten Objekten (BBox + Label + Score)

4. API-Antwort formatieren