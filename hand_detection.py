# Importarea bibliotecilor necesare
import cv2  # OpenCV - pentru procesarea imaginilor și utilizarea camerei
import mediapipe as mp  # MediaPipe - pentru detectarea mâinilor și landmark-urilor
import numpy as np  # NumPy - pentru operații matematice și lucrul cu array-uri
import os  # OS - pentru gestionarea fișierelor și directoarelor
import csv  # CSV - pentru salvarea și încărcarea datelor în format tabelar
import json  # JSON - pentru stocarea și manipularea metadatelor
import pandas as pd  # Pandas - pentru gestionarea și analiza datelor tabulare
from collections import defaultdict  # Pentru organizarea datelor într-un dicționar implicit
from sklearn.model_selection import train_test_split  # Pentru împărțirea datelor de antrenament și testare
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Pentru generarea de date augmentate pentru antrenare
from tensorflow.keras.applications import MobileNetV2  # Model de rețea neuronală pre-antrenată MobileNetV2
import tensorflow as tf  # TensorFlow - pentru învățarea automată și rularea modelului de rețea neuronală

# ---------------------------------------------
# A) Inițializarea sistemului de detectare a mâinilor cu MediaPipe
# ---------------------------------------------

mp_hands = mp.solutions.hands  # Inițializarea detectorului de mâini MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Utilitar pentru desenarea landmark-urilor detectate

# Crearea unui obiect Hands pentru detectarea mâinilor
hands = mp_hands.Hands(
    static_image_mode=False,  # Modul live (nu static)
    max_num_hands=2,  # Se pot detecta maximum 2 mâini
    min_detection_confidence=0.7,  # Pragul minim de încredere pentru detectare
    min_tracking_confidence=0.7  # Pragul minim de încredere pentru urmărirea mâinii
)

# ---------------------------------------------
# B) Configurare directoare și căi pentru salvarea datelor
# ---------------------------------------------

data_dir = "gesture_data"  # Directorul unde se vor salva imaginile gesturilor
model_path_cnn = "gesture_model.keras"  # Calea către modelul CNN MobileNetV2
model_path_landmark = "landmark_model.keras"  # Calea către modelul bazat pe landmark-uri
class_indices_path_cnn = "class_indices.json"  # Fișier JSON pentru clasele modelului CNN
class_indices_path_landmark = "landmark_class_names.json"  # Fișier JSON pentru clasele modelului bazat pe landmark-uri
csv_path = "hand_landmarks.csv"  # Fișier CSV pentru salvarea landmark-urilor detectate
old_csv_path = "hand_landmarks_old.csv"  # Backup pentru fișierul CSV anterior

# Crearea directorului pentru datele de antrenament (dacă nu există deja)
os.makedirs(data_dir, exist_ok=True)

# Inițializarea camerei video
cap = cv2.VideoCapture(0)  # Deschiderea camerei (index 0 = prima cameră detectată)

# Variabile de control
collect_data = False  # Indică dacă se colectează imagini pentru antrenare
gesture_label = "gesture_name"  # Eticheta implicită pentru gesturi (se poate modifica)
frame_index = 0  # Contor pentru salvarea imaginilor

# ---------------------------------------------
# C) Încărcarea modelului CNN pre-antrenat MobileNetV2
# ---------------------------------------------

model_cnn = None  # Inițializare model CNN
class_names_cnn = {}  # Dicționar pentru etichetele claselor CNN

n = tf.keras.models.load_model(model_path_cnn)
print(f"[INFO] Loaded MobileNet-based CNN from {model_path_cnn}")

# Verificare dacă modelul MobileNetV2 există pe disc
if os.path.exists(model_path_cnn):
    # Dacă modelul a fost antrenat anterior și salvat, îl încărcăm
    model_cnn = tf.keras.models.load_model(model_path_cnn)
    print(f"[INFO] Loaded MobileNet-based CNN from {model_path_cnn}")

    # Verificăm dacă există și fișierul JSON cu clasele modelului CNN
    if os.path.exists(class_indices_path_cnn):
        with open(class_indices_path_cnn, "r") as f:
            indices_dict = json.load(f)  # Încărcarea dicționarului de clase

            # Exemplu de conversie a dicționarului {'hello': 0, 'no': 1} -> {0: 'hello', 1: 'no'}
            class_names_cnn = {v: k for k, v in indices_dict.items()}  # Inversăm cheile cu valorile
        print(f"[INFO] Loaded CNN class names: {class_names_cnn}")  # Afișăm clasele încărcate
    else:
        # Dacă fișierul JSON cu clasele nu există, emitem o avertizare
        print("[WARNING] CNN class indices not found. Train the CNN model first.")
else:
    # Dacă modelul CNN nu există pe disc, informăm utilizatorul că trebuie antrenat
    print("[INFO] CNN model not found. Press 't' to train a new MobileNet-based CNN.")

# ---------------------------------------------
# D) Încărcarea modelului bazat pe landmark-uri
# ---------------------------------------------

# Inițializare model și dicționar de clase pentru modelul bazat pe landmark-uri
model_landmark = None
class_names_landmark = {}

# Verificare dacă modelul Landmark MLP există
if os.path.exists(model_path_landmark):
    # Dacă modelul există, îl încărcăm
    model_landmark = tf.keras.models.load_model(model_path_landmark)
    print(f"[INFO] Loaded Landmark MLP from {model_path_landmark}")

    # Verificare dacă există fișierul JSON cu clasele pentru modelul landmark
    if os.path.exists(class_indices_path_landmark):
        with open(class_indices_path_landmark, "r") as f:
            class_names_landmark = json.load(f)  # Încărcarea claselor din fișierul JSON
        print(f"[INFO] Loaded landmark class names: {class_names_landmark}")  # Afișăm clasele încărcate
    else:
        # Dacă nu există fișierul cu clasele, avertizăm utilizatorul
        print("[WARNING] Landmark class names not found. Train the Landmark model first.")
else:
    # Dacă modelul nu există, anunțăm utilizatorul că trebuie antrenat
    print("[INFO] Landmark model not found. Press 't' to train the Landmark model.")

# ---------------------------------------------
# E) Funcție ajutătoare: calcularea unui bounding box în jurul mâinii
# ---------------------------------------------

def get_hand_bbox(hand_landmarks, frame_shape):
    """Calculează un bounding box în jurul landmark-urilor detectate."""
    img_w, img_h = frame_shape[1], frame_shape[0]  # Extragem lățimea și înălțimea imaginii

    # Determinăm coordonatele minime și maxime pentru bounding box
    x_min = min(lm.x for lm in hand_landmarks.landmark) * img_w
    x_max = max(lm.x for lm in hand_landmarks.landmark) * img_w
    y_min = min(lm.y for lm in hand_landmarks.landmark) * img_h
    y_max = max(lm.y for lm in hand_landmarks.landmark) * img_h

    # Returnăm coordonatele bounding box-ului cu o margine de 10 pixeli
    return int(x_min) - 10, int(y_min) - 10, int(x_max) + 10, int(y_max) + 10

# ---------------------------------------------
# F) Backup pentru fișierul CSV existent (dacă există)
# ---------------------------------------------

# Dacă există un fișier CSV vechi, îl ștergem
if os.path.exists(old_csv_path):
    os.remove(old_csv_path)  # Ștergem fișierul CSV vechi

# Dacă există fișierul CSV curent, îl redenumim ca backup
if os.path.exists(csv_path):
    os.rename(csv_path, old_csv_path)  # Redenumim fișierul curent în backup

# ---------------------------------------------
# G) Crearea și configurarea fișierului CSV pentru logarea landmark-urilor
# ---------------------------------------------

# Deschidem fișierul CSV în modul "write" (suprascriere)
csv_file = open(csv_path, 'w', newline='')

# Creăm un writer pentru scrierea datelor în CSV
writer = csv.writer(csv_file)

# Scriem header-ul fișierului CSV (numele coloanelor)
# Coloanele sunt: Frame (indicele frame-ului), Hand (ID-ul mâinii), Joint (indicele punctului),
# X, Y, Z (coordonatele landmark-urilor), Gesture (eticheta gestului)
writer.writerow(['Frame', 'Hand', 'Joint', 'X', 'Y', 'Z', 'Gesture'])

# ---------------------------------------------
# H) Bucla principală
# Aceasta este bucla principală care captează cadrele video, detectează mâinile,
# colectează date, salvează imagini și efectuează inferența folosind modelele CNN.
# ---------------------------------------------

while cap.isOpened():  # Verifică dacă fluxul video de la cameră este deschis
    success, frame = cap.read()  # Citește un cadru video din fluxul camerei
    if not success:  # Dacă nu se poate citi cadrul (de exemplu, camera nu funcționează)
        print("[ERROR] Failed to access the camera.")  # Afișează un mesaj de eroare
        break  # Iese din buclă dacă citirea eșuează

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertim cadrul din BGR (OpenCV) în RGB (necesar pentru MediaPipe)
    results = hands.process(frame_rgb)  # Aplicăm modelul MediaPipe pentru a detecta mâinile în imagine

    if results.multi_hand_landmarks:  # Verificăm dacă s-au detectat mâini în imagine
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):  # Iterăm prin fiecare mână detectată
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  
            # Desenăm landmark-urile mâinii detectate pe imagine

            # -- Logare landmark-uri în CSV --
            h, w, _ = frame.shape  # Obținem dimensiunile imaginii
            for i, lm in enumerate(hand_landmarks.landmark):  # Iterăm prin toate punctele landmark detectate pe mână
                x = lm.x * w  # Convertim coordonata X în pixeli
                y = lm.y * h  # Convertim coordonata Y în pixeli
                z = lm.z  # Coordonata Z (adâncimea)
                writer.writerow([frame_index, hand_idx, i, x, y, z, gesture_label])  
                # Salvăm coordonatele landmark-urilor în fișierul CSV împreună cu eticheta gestului

            # -- Colectare de date (imagini decupate cu mâinile) --
            if collect_data:  # Dacă colectarea de date este activată
                x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame.shape)  
                # Obținem coordonatele bounding box-ului pentru mână
                hand_crop = frame[max(0, y_min):min(frame.shape[0], y_max),
                                  max(0, x_min):min(frame.shape[1], x_max)]
                # Decupăm regiunea care conține mâna
                if hand_crop.size > 0:  # Verificăm dacă decuparea a fost realizată corect
                    save_dir = os.path.join(data_dir, gesture_label)  # Creăm calea directorului unde se vor salva imaginile
                    os.makedirs(save_dir, exist_ok=True)  # Creăm directorul dacă nu există
                    save_path = os.path.join(save_dir, f"{frame_index}.jpg")  # Definim calea fișierului pentru salvare
                    cv2.imwrite(save_path, hand_crop)  # Salvăm imaginea decupată
                    frame_index += 1  # Incrementăm indexul cadrului

            # -- Inferență folosind modelul CNN MobileNet --
            gesture_text = "Gesture: Unknown - 0%"  # Inițializăm textul pentru recunoașterea gestului
            if model_cnn is not None:  # Verificăm dacă modelul CNN este încărcat
                x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame.shape)  
                # Obținem bounding box-ul mâinii
                hand_crop_inference = frame[max(0, y_min):min(frame.shape[0], y_max),
                                            max(0, x_min):min(frame.shape[1], x_max)]
                # Decupăm mâna pentru inferență
                if hand_crop_inference.size > 0:  # Verificăm dacă decuparea a fost realizată corect
                    frame_resized = cv2.resize(hand_crop_inference, (128, 128)) / 255.0  
                    # Redimensionăm imaginea la 128x128 și normalizăm valorile pixelilor
                    frame_input = np.expand_dims(frame_resized, axis=0)  # Adăugăm o dimensiune suplimentară pentru rețea
                    prediction = model_cnn.predict(frame_input, verbose=0)  # Realizăm inferența cu modelul CNN
                    confidence = np.max(prediction) * 100  # Obținem încrederea în predicție
                    pred_label_idx = np.argmax(prediction)  # Obținem indexul clasei prezise

                    if confidence >= 80:  # Dacă încrederea este mai mare de 80%, afișăm rezultatul
                        label_str = class_names_cnn.get(pred_label_idx, "Unknown")  
                        gesture_text = f"Gesture: {label_str} - {int(confidence)}%"  
                    else:
                        gesture_text = f"Gesture: Unknown - {int(confidence)}%"

            # Afișăm textul recunoscut pe imagine
            cv2.putText(frame, gesture_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afișăm imaginea procesată într-o fereastră
    cv2.imshow("Hand Gesture Recognition", frame)

    # --- Gestionarea tastelor ---
    key = cv2.waitKey(1) & 0xFF  # Așteptăm input de la tastatură
    if key == ord('q') or cv2.getWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
        # Dacă utilizatorul apasă 'q' sau fereastra este închisă, ieșim din buclă
        break
    elif key == ord('n'):  # Dacă utilizatorul apasă 'n'
        new_label = input("Enter new gesture label: ")  # Solicită un nou nume pentru gest
        gesture_label = new_label.strip()  # Salvează eticheta gestului
        print(f"[INFO] Gesture label changed to: {gesture_label}")
    elif key == ord('c'):  # Dacă utilizatorul apasă 'c'
        collect_data = not collect_data  # Activează sau dezactivează colectarea de date
        print(f"[INFO] Data collection {'enabled' if collect_data else 'disabled'}")
    elif key == ord('t'):
        print("[INFO] Stopping camera for training...")
        cap.release()  # Oprește capturarea video
        cv2.destroyAllWindows()  # Închide fereastra OpenCV
        print("[INFO] Training BOTH MobileNet-based CNN and Landmark-based models...")
        # Aici va începe antrenarea
        # După antrenare, trebuie să redeschizi camera dacă vrei să continui capturarea

         
# *************************************
# 1) Antrenarea modelului CNN MobileNetV2 pe baza imaginilor decupate cu mâinile
# *************************************
print("[INFO] Training MobileNet-based CNN on cropped images...")  
# Afișăm un mesaj informativ că antrenarea CNN începe

# Creăm un generator de imagini pentru augmentare și normalizare
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalizăm valorile pixelilor între 0 și 1
    validation_split=0.2,  # Separația între datele de antrenament (80%) și de validare (20%)
    rotation_range=25,  # Rotim imaginile aleator cu maxim 25 de grade
    width_shift_range=0.2,  # Deplasăm orizontal imaginea cu 20% aleator
    height_shift_range=0.2,  # Deplasăm vertical imaginea cu 20% aleator
    zoom_range=0.2,  # Aplicăm un zoom aleator de maxim 20%
    horizontal_flip=True  # Permitem inversarea orizontală a imaginilor (flipping)
)

# Creăm generatorul pentru setul de antrenament
train_gen = datagen.flow_from_directory(
    data_dir,  # Directorul unde sunt salvate imaginile pentru antrenare
    target_size=(128, 128),  # Redimensionăm imaginile la 128x128 pixeli
    batch_size=32,  # Setăm batch-ul de 32 de imagini
    class_mode='categorical',  # Clasificare multi-clasă
    subset='training'  # Folosim acest generator pentru antrenament
)

# Creăm generatorul pentru setul de validare
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Folosim acest generator pentru validare
)

num_classes = len(train_gen.class_indices)  # Numărul de clase detectate în datele de antrenament

# Salvăm indexul claselor pentru a-l putea folosi la inferență
with open(class_indices_path_cnn, "w") as f:
    json.dump(train_gen.class_indices, f)

class_names_cnn = {v: k for k, v in train_gen.class_indices.items()}  
# Inversăm cheile și valorile pentru a putea face maparea de la indice la nume de clasă
print(f"[INFO] CNN class indices saved: {class_names_cnn}")  
# Afișăm un mesaj informativ cu clasele identificate

# Definim modelul MobileNetV2 ca bază a rețelei neurale convoluționale
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # Setăm dimensiunea intrării ca fiind 128x128 pixeli și 3 canale (RGB)
    include_top=False,  # Excludem straturile finale ale modelului MobileNetV2 (vom adăuga propriile noastre straturi)
    weights='imagenet'  # Folosim greutăți pre-antrenate pe ImageNet pentru transfer learning
)

# Congelăm straturile de bază (nu le antrenăm inițial) pentru a păstra caracteristicile pre-antrenate
base_model.trainable = False

# Construim modelul final adăugând un cap de clasificare peste MobileNetV2
model_cnn = tf.keras.Sequential([
    base_model,  # Folosim MobileNetV2 ca extractor de caracteristici
    tf.keras.layers.GlobalAveragePooling2D(),  # Facem o reducere globală pe caracteristicile extrase
    tf.keras.layers.Dense(128, activation='relu'),  # Adăugăm un strat dens cu 128 de neuroni și funcție de activare ReLU
    tf.keras.layers.Dropout(0.3),  # Adăugăm un strat de dropout (30%) pentru a evita supraînvățarea
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Stratul final pentru clasificare cu `num_classes` ieșiri
])

# Compilăm modelul CNN
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Antrenăm modelul inițial (cu straturile de bază înghețate)
print("[INFO] Initial training (frozen base)...")
model_cnn.fit(train_gen, validation_data=val_gen, epochs=10)  # Antrenăm modelul pentru 10 epoci

# Opțional: Îmbunătățim modelul prin fine-tuning
print("[INFO] Fine-tuning MobileNetV2 top layers...")

fine_tune_at = 100  # Numărul de straturi care rămân înghețate. Straturile de după acest număr vor fi antrenate.

for layer in base_model.layers[fine_tune_at:]:  
    layer.trainable = True  # Facem antrenabile doar straturile superioare ale MobileNetV2

# Compilăm din nou modelul, dar cu un learning rate mai mic
model_cnn.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model_cnn.fit(train_gen, validation_data=val_gen, epochs=10)  # Continuăm antrenarea pentru încă 10 epoci

# Salvăm modelul antrenat
model_cnn.save(model_path_cnn)  
print(f"[INFO] MobileNet-based CNN saved to {model_path_cnn}")  # Afișăm un mesaj informativ

# *************************************
# 2) Antrenarea modelului MLP bazat pe landmark-uri
# *************************************
print("[INFO] Training Landmark-based MLP (Approach B) from CSV...")

df = pd.read_csv(csv_path)  # Citim fișierul CSV cu landmark-urile detectate

# Grupăm datele după frame și gest
grouped = df.groupby(['Frame', 'Gesture'])

samples = []  # Lista în care stocăm datele de intrare
labels = []  # Lista etichetelor pentru antrenare

# Iterăm prin fiecare grup de date format pe baza ID-ului unui cadru (frame_id) și a gestului asociat (label)
for (frame_id, label), group in grouped:
    
    # Putem avea între 0 și 2 mâini detectate în acest grup
    # Vom separa datele pentru fiecare mână folosind coloana "Hand"
    hand_ids = sorted(group['Hand'].unique())  # Obținem lista de ID-uri unice ale mâinilor și o sortăm
    
    if len(hand_ids) > 2:
        # Dacă sunt detectate mai mult de 2 mâini într-un singur cadru, ignorăm acest cadru
        continue  

    # Inițializăm două liste pentru coordonatele landmark-urilor fiecărei mâini detectate
    # Fiecare listă are 63 de elemente (21 de puncte × 3 coordonate: X, Y, Z)
    coords_hand_0 = [0.0] * 63
    coords_hand_1 = [0.0] * 63

    # Definim o funcție care umple o listă de coordonate pe baza unui subset de date (subdf)
    def fill_coords(coords_list, subdf):
        # Verificăm dacă avem exact 21 de landmark-uri pentru o mână completă
        if len(subdf) != 21:
            return False  # Dacă sunt mai puține sau mai multe puncte, returnăm False
        
        # Sortăm punctele pe baza indexului "Joint" pentru a ne asigura că sunt în ordinea corectă
        subdf_sorted = subdf.sort_values('Joint')  
        
        idx = 0  # Inițializăm indexul pentru listă
        for _, row in subdf_sorted.iterrows():
            # Stocăm coordonatele X, Y și Z în listă
            coords_list[idx] = row['X']
            coords_list[idx + 1] = row['Y']
            coords_list[idx + 2] = row['Z']
            idx += 3  # Incrementăm indexul cu 3 pentru a trece la următorul set de coordonate

        return True  # Returnăm True dacă landmark-urile au fost completate corect

    filled_count = 0  # Contor pentru numărul de mâini procesate corect

    # Iterăm prin fiecare mână detectată în acest cadru
    for i, h_id in enumerate(hand_ids):
        subdf = group[group['Hand'] == h_id]  # Selectăm datele corespunzătoare acelei mâini
        
        if i == 0:  # Procesăm prima mână
            ok = fill_coords(coords_hand_0, subdf)  # Completăm coordonatele primei mâini
            if not ok:
                break  # Dacă landmark-urile sunt incomplete, ieșim din buclă
        
        elif i == 1:  # Procesăm a doua mână (dacă există)
            ok = fill_coords(coords_hand_1, subdf)  # Completăm coordonatele celei de-a doua mâini
            if not ok:
                break  # Dacă landmark-urile sunt incomplete, ieșim din buclă
        
        filled_count += 1  # Incrementăm numărul de mâini procesate corect

    if filled_count == 0:
        # Dacă nicio mână nu a fost procesată corect, ignorăm acest cadru
        continue

    # Combinăm coordonatele celor două mâini într-o singură listă de 126 de elemente (63 + 63)
    combined = coords_hand_0 + coords_hand_1  
    samples.append(combined)  # Adăugăm această secvență de landmark-uri în setul de antrenament
    labels.append(label)  # Adăugăm eticheta (numele gestului)

# Convertim lista de mostre într-un array NumPy pentru procesare eficientă
X = np.array(samples, dtype=np.float32)  
y_labels = np.array(labels)  # Convertim lista de etichete într-un array NumPy

# Convertim etichetele de text în indici numerici
unique_labels = np.unique(y_labels)  # Obținem lista de etichete unice (numele gesturilor)
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}  # Mapăm fiecare etichetă la un număr
idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}  # Creăm un dicționar invers pentru conversie inversă

y_int = np.array([label_to_idx[lbl] for lbl in y_labels])  # Convertim etichetele în numere întregi

if len(y_int) < 2:
    # Dacă avem mai puțin de 2 clase de gesturi, nu putem antrena modelul
    print("[WARNING] Not enough data to train Landmark MLP. Skipping.")
else:
    # Împărțim datele în seturi de antrenament și validare (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_int, test_size=0.2, shuffle=True, stratify=y_int
    )

    # Definim arhitectura unui model Multi-Layer Perceptron (MLP) pentru clasificarea gesturilor
    model_landmark = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(126,)),  # Primul strat dens (fully connected)
        tf.keras.layers.Dropout(0.3),  # Dropout pentru prevenirea supraînvățării
        tf.keras.layers.Dense(128, activation='relu'),  # Al doilea strat dens
        tf.keras.layers.Dropout(0.3),  
        tf.keras.layers.Dense(len(unique_labels), activation='softmax')  # Strat de ieșire cu softmax pentru clasificare
    ])

    # Compilăm modelul cu optimizatorul Adam și funcția de pierdere corespunzătoare clasificării
    model_landmark.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  

    # Antrenăm modelul folosind setul de antrenament și validare
    model_landmark.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

    # Salvăm modelul antrenat
    model_landmark.save(model_path_landmark)
    print(f"[INFO] Landmark MLP saved to {model_path_landmark}")

    # Salvăm etichetele claselor într-un fișier JSON pentru utilizare ulterioară
    with open(class_indices_path_landmark, "w") as f:
        json.dump({str(i): idx_to_label[i] for i in idx_to_label}, f)  # Salvăm maparea claselor

    print("[INFO] Landmark model training finished.")

# Incrementăm indexul cadrului pentru următoarea iterație
frame_index += 1

# Curățăm resursele alocate pentru fișiere și camera video
csv_file.close()  # Închidem fișierul CSV
cap.release()  # Oprim capturarea video
cv2.destroyAllWindows()  # Închidem toate ferestrele OpenCV
