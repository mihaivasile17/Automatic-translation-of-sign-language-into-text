# ✅ Importăm librăriile necesare
import os  # Gestionarea fișierelor și căilor
import cv2  # OpenCV pentru procesare imagine
import numpy as np  # Operații numerice și matriciale
import tensorflow as tf  # Biblioteca pentru rețele neuronale și deep learning
import json  # Manipulare fișiere JSON
from flask import Flask, request, jsonify  # Flask pentru crearea API-ului web
from flask_cors import CORS  # Permite accesul API-ului dintr-o aplicație frontend (Cross-Origin Resource Sharing)
from PIL import Image  # Procesare imagine folosind Pillow (PIL)
import mediapipe as mp  # Biblioteca MediaPipe pentru detectarea landmark-urilor mâinii

# ✅ Inițializăm aplicația Flask
app = Flask(__name__)  # Creăm instanța aplicației web
CORS(app)  # Permitem cereri de tip cross-origin (CORS) pentru frontend

# ✅ Definim căile către fișierele modelului și ale claselor
MODEL_CNN_PATH = "gesture_model.keras"  # Calea către modelul CNN antrenat
CLASS_INDICES_PATH_CNN = "class_indices.json"  # Calea către fișierul cu clasele gesturilor

# ✅ Încărcăm modelul CNN, dacă există
if os.path.exists(MODEL_CNN_PATH):  # Verificăm dacă fișierul modelului există
    model_cnn = tf.keras.models.load_model(MODEL_CNN_PATH)  # Încărcăm modelul CNN din fișier
    print(f"[INFO] Loaded MobileNet-based CNN from {MODEL_CNN_PATH}")  # Afișăm un mesaj de succes
else:
    model_cnn = None  # Dacă modelul nu este găsit, setăm variabila ca None
    print("[WARNING] CNN model not found. Please train it first.")  # Mesaj de avertizare

# ✅ Încărcăm fișierul cu numele claselor și creăm un dicționar inversat
if os.path.exists(CLASS_INDICES_PATH_CNN):  # Verificăm dacă fișierul JSON există
    with open(CLASS_INDICES_PATH_CNN, "r") as f:
        original_class_names = json.load(f)  # Încărcăm mapping-ul claselor (ex: {"hello": 0, "no": 1})

    # Inversăm mapping-ul pentru a putea converti predicția numerică în etichetă
    class_names_cnn = {v: k for k, v in original_class_names.items()}  # ex: {0: "hello", 1: "no"}
    print(f"[DEBUG] Reversed CNN Class Names Mapping: {class_names_cnn}")  # Afișăm mapping-ul inversat pentru debugging
else:
    class_names_cnn = {}  # Dacă fișierul nu există, setăm dicționarul ca gol
    print("[WARNING] CNN class indices not found. Train the model first.")  # Mesaj de avertizare

# ✅ Inițializăm MediaPipe Hands pentru detectarea landmark-urilor mâinii
mp_hands = mp.solutions.hands  # Importăm componenta "hands" din MediaPipe
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)  
# - `static_image_mode=True` -> Procesăm imagini statice
# - `max_num_hands=2` -> Putem detecta până la două mâini în imagine
# - `min_detection_confidence=0.7` -> Pragul minim de încredere pentru detectarea unei mâini

# ✅ Funcție de preprocesare a imaginilor pentru modelul CNN
def preprocess_image(image):
    """
    Preprocesează imaginea înainte de a fi trimisă către modelul CNN.
    """
    image = image.resize((128, 128))  # Redimensionăm imaginea la dimensiunea de antrenare (128x128)
    image = np.array(image).astype("float32") / 255.0  # Normalizăm pixelii între 0 și 1
    image = np.expand_dims(image, axis=0)  # Adăugăm o dimensiune suplimentară pentru a se potrivi cu modelul CNN
    return image  # Returnăm imaginea preprocesată

# ✅ Ruta principală API (Test)
@app.route("/", methods=["GET"])  # Creăm o rută pentru pagina principală (doar GET)
def home():
    return jsonify({"message": "Gesture Recognition API is running!"})  
    # Returnăm un mesaj JSON pentru a indica faptul că API-ul rulează

# ✅ Ruta pentru clasificarea unei imagini încărcate
@app.route("/predict_image", methods=["POST"])  # Rută pentru primirea unei imagini și returnarea predicției
def predict_image():
    try:
        # ✅ Verificăm dacă a fost trimis un fișier imagine în cerere
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400  # Dacă nu există imagine, returnăm o eroare
        
        file = request.files["image"]  # Extragem imaginea încărcată
        image = Image.open(file.stream).convert("RGB")  # Deschidem imaginea și o convertim în format RGB
        processed_image = preprocess_image(image)  # Apelăm funcția de preprocesare a imaginii

        # ✅ Verificăm dacă modelul CNN a fost încărcat
        if model_cnn is None:
            return jsonify({"error": "CNN model not loaded"}), 500  # Dacă modelul nu a fost găsit, returnăm o eroare
        
        # ✅ Realizăm predicția folosind modelul CNN
        predictions = model_cnn.predict(processed_image, verbose=0)  # Obținem predicțiile
        confidence = np.max(predictions) * 100  # Extragem cea mai mare probabilitate și o convertim în procentaj
        predicted_label_index = int(np.argmax(predictions))  # Obținem indexul clasei prezise

        # ✅ Mapăm indexul prezis în eticheta corespunzătoare
        label = class_names_cnn.get(predicted_label_index, "Unknown")  # Obținem numele clasei sau "Unknown" dacă nu există

        # ✅ Loguri pentru debugging
        print(f"[DEBUG] Prediction Raw Output: {predictions}")  # Afișăm vectorul brut de predicție
        print(f"[DEBUG] Predicted Label Index: {predicted_label_index}, Confidence: {confidence:.2f}%")  # Afișăm indexul prezis și încrederea
        print(f"[DEBUG] Mapped Label: {label}")  # Afișăm numele clasei prezise

        return jsonify({"label": label, "confidence": round(confidence, 2)})  # Returnăm eticheta prezisă și încrederea în JSON
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Dacă apare o eroare, o returnăm în JSON

# ✅ Lansăm aplicația Flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Pornim serverul Flask pe portul 5000, cu debug activat
