#1. Zuerst importieren wir die benötigten Module:
import cv2
import numpy as np
import os
import re

#2. Dann definieren wir die Funktion `compare_images()`, die zwei Bilder mit ORB vergleicht und die Anzahl der Übereinstimmungen zurückgibt:
def compare_images(image1, image2):
    """
    Vergleicht die Übereinstimmungen zwischen zwei Bildern mithilfe von ORB (Oriented FAST and Rotated BRIEF)
    und gibt die Anzahl der Übereinstimmungen zurück.
    """
    #1. Bilder skalieren, falls unterschiedliche Größe
    image1 = cv2.resize(image1, (500, 500))
    image2 = cv2.resize(image2, (500, 500))

    #2. Extrahieren der Keypoints und Descriptoren mithilfe von ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    #3. Verwenden Sie die BFMatcher-Klasse, um Übereinstimmungen zwischen den Keypoints zu finden
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    #4. Vergleich von Bildern
    return len(matches)

#3. Jetzt rufen wir das Verzeichnis der Hauptgruppe und das Verzeichnis der Sammelgruppe ab:
# Pfad zur Hauptgruppe
hauptgruppe_path = "C:/Users/Admin/Pictures/memecontest/hauptgruppe/photos"
# Pfad zur Sammelgruppe
sammelgruppe_path = "C:/Users/Admin/Pictures/memecontest/sammelgruppe/photos"

#4. Wir löschen zuerst alle "thumb"-Dateien im Hauptgruppen-Verzeichnis:
# Löschen aller "thumb"-Dateien im Hauptgruppen-Verzeichnis
files = os.listdir(hauptgruppe_path)
for filename in files:
    if "thumb" in filename:
        os.remove(os.path.join(hauptgruppe_path, filename))

#5. Jetzt skalieren wir die Bilder in der Hauptgruppe auf 500x500 Pixel und speichern sie mit dem neuen Namen:
# Skalieren und Umbenennen der Bilder in der Hauptgruppe
files = os.listdir(hauptgruppe_path)
for filename in files:
    if not filename.endswith(".jpg"):
        continue
    
    if "thumb" in filename:
        continue
    
    if "@" not in filename:
        continue

    # Nummer und Datum aus dem Dateinamen extrahieren
    image_number = re.search(r"photo_(\d+)@\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}.jpg", filename).group(1)
    date = re.search(r"\d{2}-\d{2}-\d{4}", filename).group(0)

    # Bild einlesen und skalieren, falls notwendig
    image = cv2.imread(os.path.join(hauptgruppe_path, filename))
    if image.shape != (500, 500, 3):
        image = cv2.resize(image, (500, 500))
        # Skaliertes Bild speichern
        new_name = os.path.join(hauptgruppe_path, f"photo_{image_number}_{date}.jpg")
        cv2.imwrite(new_name, image)
        # Originalbild löschen
        os.remove(os.path.join(hauptgruppe_path, filename))


           
 #6. Jetzt überprüfen wir die Bilder in der Sammelgruppe auf Ähnlichkeit mit den Bildern in der Hauptgruppe:
# Liste aller Bilder in der Hauptgruppe
hauptgruppe_images = [cv2.imread(f"{hauptgruppe_path}/{filename}") for filename in os.listdir(hauptgruppe_path)]
hauptgruppe_filenames = [filename for filename in os.listdir(hauptgruppe_path)]

# Überprüfen Sie alle Bilder in der Sammelgruppe
for filename in os.listdir(sammelgruppe_path):
    image = cv2.imread(f"{sammelgruppe_path}/{filename}")
    
    # Überprüfen Sie jedes Bild in der Hauptgruppe
    for j, hauptgruppe_image in enumerate(hauptgruppe_images):
        match_count = compare_images(image, hauptgruppe_image)
        if match_count > 300:
            print(f"Das Bild {filename} ähnelt dem Bild {hauptgruppe_filenames[j]} in der Hauptgruppe.")
            break
    else:
        # Falls kein passendes Bild gefunden wird, geben Sie aus, dass das Bild in der Hauptgruppe nicht vorhanden ist.
        print(f"Das Bild {filename} ist in der Hauptgruppe nicht vorhanden.")
        
    # Löschen des Originalbildes in der Sammelgruppe
    #os.remove(f"{sammelgruppe_path}/{filename}")

#7. Zum Schluss geben wir eine Erfolgsmeldung aus
print("Alle Bilder in der Sammelgruppe wurden überprüft und die Originalbilder in der Hauptgruppe wurden gelöscht.")
