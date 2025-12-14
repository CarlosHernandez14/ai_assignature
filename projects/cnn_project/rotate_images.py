import cv2
import os

# --- CONFIGURACIÓN DE RUTAS ---
INPUT_FOLDER = "/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/cnn_project/datasets/ant/new_clean"
OUTPUT_FOLDER = "/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/cnn_project/datasets/ant/cleaned_mirrored" 
MAX_IMAGES_TO_PROCESS = 6000

# --- PROCESAMIENTO ---


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Carpeta de salida para imágenes espejadas: {OUTPUT_FOLDER}")


image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]

print(f"Iniciando procesamiento de {len(image_files)} imágenes de {INPUT_FOLDER}...")

processed_count = 0
for filename in image_files:

    if processed_count >= MAX_IMAGES_TO_PROCESS:
        print(f"Límite de {MAX_IMAGES_TO_PROCESS} imágenes alcanzado. Deteniendo el procesamiento.")
        break

    input_image_path = os.path.join(INPUT_FOLDER, filename)
    
    image = cv2.imread(input_image_path)
    
    if image is None:
        print(f"Advertencia: No se pudo leer la imagen {filename}. Omitiendo.")
        continue
        
   
    # flipCode = 0 para volteo vertical
    # flipCode = 1 para volteo horizontal (espejo)
    # flipCode = -1 para volteo horizontal y vertical
    #mirrored_image = cv2.flip(image, 1) # El '1' indica volteo horizontal
    mirrored_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

  
    base_name, extension = os.path.splitext(filename)

    new_filename = f"mirror_{base_name}_r{extension}"

    output_image_path = os.path.join(OUTPUT_FOLDER, new_filename)

    cv2.imwrite(output_image_path, mirrored_image)
    
    processed_count += 1
    print(f"Procesado: {filename} -> Guardado como {output_image_path}")

print(f"\n✅ Procesamiento terminado. Se espejaron y guardaron {processed_count} imágenes en {OUTPUT_FOLDER}.")