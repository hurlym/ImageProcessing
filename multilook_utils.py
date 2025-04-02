import cv2
import os
import numpy as np

def calculate_window_size(img):
    # Get image dimensions
    height, width = img.shape[:2]

    # Calculate noise level using local standard deviation
    kernel_size = 5
    mean = cv2.blur(img, (kernel_size, kernel_size))
    mean_sq = cv2.blur(img * img, (kernel_size, kernel_size))
    variance = mean_sq - mean * mean
    noise_level = np.mean(np.sqrt(np.abs(variance)))

    # Basic heuristic for window size
    # Higher noise levels require larger windows
    if noise_level > 0.5:
        return 7
    elif noise_level > 0.3:
        return 5
    else:
        return 3

def multilook_process(folder_path, auto_window=True):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    if not image_files:
        print("No .tif files found in the folder")
        return

    for image_file in image_files:
        try:
            image_path = os.path.join(folder_path, image_file)
            #Se lee la imagen tif sin modificar el formato
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error reading image: {image_file}")
                continue
                # Auto calculate window size based on image characteristics
            if auto_window:
                window_size = calculate_window_size(img)
                print(f"Calculated window size: {window_size}")
            #Almacena el tipo de dato original de la imagen
            original_dtype = img.dtype
            #Se calcula el valor minimo y maximo de la imagen
            min_val = np.min(img)
            max_val = np.max(img)

            # Convert to float32 and normalize
            img_normalized = (img.astype(np.float32) - min_val) / (max_val - min_val)

            # Apply multilooking
            kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
            multilooked_img = cv2.filter2D(img_normalized, -1, kernel)

            # Scale back to original range
            multilooked_img = (multilooked_img * (max_val - min_val) + min_val)

            # Convert back to original data type
            multilooked_img = np.clip(multilooked_img, 0, np.iinfo(original_dtype).max)
            multilooked_img = multilooked_img.astype(original_dtype)

            output_path = os.path.join(folder_path, f"multilooked_{image_file}")
            cv2.imwrite(output_path, multilooked_img)
            print(f"Processed {image_file}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

