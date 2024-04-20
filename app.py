import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO
from subprocess import run

app = Flask(__name__)

# Charger le modèle YOLO
model = YOLO("besst.pt")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Effacer le contenu du dossier 'cropped_images'
    cropped_images_dir = os.path.join('static', 'cropped_images')
    for filename in os.listdir(cropped_images_dir):
        file_path = os.path.join(cropped_images_dir, filename)
        os.remove(file_path)

    if request.method == 'POST':
        # Récupérer l'image envoyée
        file = request.files['image']
        
        # Enregistrer l'image dans un fichier temporaire
        file_path = os.path.join('static', 'temp.jpg')
        file.save(file_path)
        
        # Détecter les objets avec YOLO
        results = model(file_path)
        
        # Charger l'image originale
        img = cv2.imread(file_path)
        
        # Créer le dossier 'cropped_images' s'il n'existe pas
        cropped_images_dir = os.path.join('static', 'cropped_images')
        os.makedirs(cropped_images_dir, exist_ok=True)
        
        # Parcourir les détections et dessiner les bounding boxes
        box_coordinates = []
        recognized_texts = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = [coord for sublist in box.xyxy.tolist() for coord in sublist]
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Enregistrer les coordonnées des bounding boxes
                box_coordinates.append(f"{file_path} {x1} {y1} {x2} {y2}")
                
                # Enregistrer les textes reconnus
                recognized_texts.append(f"{file_path} placeholder_text 0.5")
                
                # Cropper l'image à l'intérieur de la bounding box
                cropped_img = img[y1:y2, x1:x2]
                cropped_img_path = os.path.join(cropped_images_dir, f"cropped_{len(box_coordinates)}.jpg")
                cv2.imwrite(cropped_img_path, cropped_img)
        
        # Enregistrer l'image avec les bounding boxes
        results_path = os.path.join('static', 'results.jpg')
        cv2.imwrite(results_path, img)
        
        # Écrire les coordonnées des bounding boxes et les textes reconnus dans les fichiers .txt
        with open('box_coordinates.txt', 'w') as boxes_file, open('recognized_texts.txt', 'w') as texts_file:
            boxes_file.write('\n'.join(box_coordinates))
            texts_file.write('\n'.join(recognized_texts))
        
        # Appeler la fonction run_demo et attendre qu'elle termine
        run_demo_output = run_demo("TPS", "ResNet", "BiLSTM", "Attn", cropped_images_dir, "TPS-ResNet-BiLSTM-Attn.pth")
        
        # Récupérer les mots détectés
        with open("recognized_texts.txt", "r") as file:
            detected_words = [line.strip().split()[1] for line in file.readlines()]
        
        # Récupérer le contenu du fichier log_demo.txt
        with open("log_demo.txt", "r") as log_file:
            log_demo_content = log_file.read()
        
        # Renvoyer le rendu HTML avec les images et les mots détectés
        return render_template('index.html', original_image='temp.jpg', processed_image='results.jpg', cropped_images=os.listdir(cropped_images_dir), detected_words=detected_words, log_demo=log_demo_content)
    
    return render_template('index.html')

def run_demo(transformation, feature_extraction, sequence_modeling, prediction, image_folder, saved_model):
    command = f"python demo.py --Transformation {transformation} --FeatureExtraction {feature_extraction} --SequenceModeling {sequence_modeling} --Prediction {prediction} --image_folder {image_folder} --saved_model {saved_model}"
    result = run(command, shell=True, capture_output=True, text=True)
    
    # Écrire la sortie de la commande dans le fichier log_demo.txt
    with open("log_demo.txt", "w") as log_file:
        log_file.write(result.stdout)
    
    return result

if __name__ == '__main__':
    app.run(debug=True)