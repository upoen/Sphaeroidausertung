import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading

# Globale Variablen für den Abbrechen-Mechanismus und Fortschrittsanzeige
process_running = threading.Event()
progress_var = None

def draw_contours_and_centroid(binary_mask, output_path):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_mask, contours, -1, (0, 255, 0), 2)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(color_mask, (cX, cY), 5, (0, 0, 255), -1)
    cv2.imwrite(output_path, color_mask)

def load_and_preprocess_image_pillow(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((256, 192))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def predict_with_model(model, image):
    image = np.expand_dims(image, axis=0)
    predicted_mask = model.predict(image)
    return np.squeeze(predicted_mask)

def calculate_average_contour_size(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return cv2.contourArea(contour) / perimeter

def find_best_threshold(mask):
    best_threshold = 0
    best_score = 0
    for threshold in np.arange(0.1, 0.5, 0.01):
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            avg_size = calculate_average_contour_size(largest_contour)
            score = avg_size
            if score > best_score:
                best_score = score
                best_threshold = threshold
    return best_threshold

def calculate_average_diameter(contour, center, num_lines=180):
    angles = np.linspace(0, 2 * np.pi, num=num_lines, endpoint=False)
    diameters = []
    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        intersections = []
        for i in range(-1000, 1000):
            x = int(center[0] + i * dx)
            y = int(center[1] + i * dy)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                intersections.append((x, y))
        if len(intersections) >= 2:
            d = np.linalg.norm(np.array(intersections[0]) - np.array(intersections[-1]))
            diameters.append(d)
    return np.mean(diameters)

def start_processing():
    global process_running
    input_dir = input_folder_entry.get()
    output_dir = output_folder_entry.get()

    if not (os.path.isdir(input_dir) and os.path.isdir(output_dir)):
        messagebox.showerror("Fehler", "Bitte überprüfen Sie die angegebenen Pfade.")
        return

    process_running.set()  # Signal, dass der Prozess läuft
    progress_var.set(0)
    processing_thread = threading.Thread(target=run_processing, args=(input_dir, output_dir))
    processing_thread.start()
    root.after(100, check_thread, processing_thread)

def run_processing(input_dir, output_dir):
    model_path = 'C:/Users/chris/Documents/Master/Sphaeroidauswertung/Modelle/spheroid_segmentation_200_trainingsbilder_noch_komplexer_early_stop_and_droput_3_3.h5'
    model = tf.keras.models.load_model(model_path)

    predicted_diameters = []
    manual_diameters = []

    scale_factor_width = 1296 / 256
    scale_factor_height = 966 / 192

    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    total_files = len(tif_files)
    for i, filename in enumerate(tif_files):
        if not process_running.is_set():
            break

        img_path = os.path.join(input_dir, filename)
        try:
            image = load_and_preprocess_image_pillow(img_path)
            predicted_mask = predict_with_model(model, image)
            if predicted_mask is None:
                print(f"Fehler bei der Vorhersage für {filename}.")
                continue

            best_threshold = find_best_threshold(predicted_mask)
            binary_mask = (predicted_mask > best_threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                print(f"Keine Konturen im Bild {filename} gefunden.")
                continue

            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.5:
                    filtered_contours.append(contour)

            if not filtered_contours:
                print(f"Keine geeigneten Konturen im Bild {filename} gefunden.")
                continue

            largest_contour = max(filtered_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                print(f"Schwerpunkt konnte im Bild {filename} nicht berechnet werden.")
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            avg_diameter_pixels = calculate_average_diameter(largest_contour, center)
            avg_diameter_micrometers = avg_diameter_pixels * (0.3745 * scale_factor_width) * 0.994

            predicted_diameters.append(avg_diameter_micrometers)

            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_segmentiert.tif")
            draw_contours_and_centroid(binary_mask, output_path)
            print(f"Segmentiertes Bild für {filename} wurde gespeichert.")
            print(f"Bild: {filename}, Durchmesser: {avg_diameter_micrometers:.2f} µm")

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {filename}: {e}")

        progress = (i + 1) / total_files * 100
        progress_var.set(progress)
        root.update_idletasks()

    # Excel-Ausgabe nach Verarbeitung aller Bilder
    save_excel_file(output_dir, predicted_diameters)

    # Öffnen des Ausgabeordners
    open_output_folder(output_dir)

    # Signalisieren, dass der Prozess abgeschlossen ist
    process_running.clear()

def save_excel_file(output_dir, diameters):
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel files", "*.xlsx")],
                                             initialdir=output_dir,
                                             title="Speichern Sie die Excel-Datei")
    if not save_path:
        return

    df = pd.DataFrame({'Durchmesser (µm)': diameters})
    df.to_excel(save_path, index=False)
    print(f"Ergebnisse wurden in {save_path} gespeichert.")

def cancel_process():
    global process_running
    process_running.clear()  # Signal, dass der Prozess abgebrochen wurde
    messagebox.showinfo("Abbruch", "Die Verarbeitung wurde abgebrochen.")

def check_thread(thread):
    if thread.is_alive():
        root.after(100, check_thread, thread)
    else:
        progress_var.set(100)  # Fortschritt auf 100% setzen, wenn abgeschlossen
        root.update_idletasks()
        root.after(500, close_window)  # Fenster nach 500 ms schließen

def close_window():
    root.quit()  # Beendet die Tkinter-Hauptschleife
    root.destroy()  # Bereinigt alle Tkinter-Ressourcen und schließt das Fenster

def open_output_folder(output_dir):
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_dir)
        elif os.name == 'posix':  # macOS/Linux
            os.system(f'open "{output_dir}"')  # macOS
            # os.system(f'xdg-open "{output_dir}"')  # Linux (entfernen das Kommentarzeichen wenn auf Linux verwenden)
    except Exception as e:
        print(f"Fehler beim Öffnen des Ausgabeordners: {e}")

def select_folder(title):
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

def main():
    global input_folder_entry, output_folder_entry, progress_var, root

    root = tk.Tk()
    root.title("Sphäroid-Auswertung")

    progress_var = tk.DoubleVar()

    tk.Label(root, text="Eingabeordner:").grid(row=0, column=0, padx=10, pady=5)
    input_folder_entry = tk.Entry(root, width=50)
    input_folder_entry.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="Durchsuchen...", command=lambda: input_folder_entry.insert(0, select_folder("Wählen Sie den Eingabeordner"))).grid(row=0, column=2, padx=10, pady=5)

    tk.Label(root, text="Ausgabeordner:").grid(row=1, column=0, padx=10, pady=5)
    output_folder_entry = tk.Entry(root, width=50)
    output_folder_entry.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(root, text="Durchsuchen...", command=lambda: output_folder_entry.insert(0, select_folder("Wählen Sie den Ausgabeordner"))).grid(row=1, column=2, padx=10, pady=5)

    tk.Button(root, text="Verarbeitung starten", command=start_processing).grid(row=2, column=1, pady=20)

    progress_bar = ttk.Progressbar(root, orient='horizontal', length=400, mode='determinate', variable=progress_var)
    progress_bar.grid(row=3, column=0, columnspan=3, padx=10, pady=5)
    
    tk.Button(root, text="Abbrechen", command=cancel_process).grid(row=4, column=1, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
