import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import csv
import os

def save_values_to_csv(ndpi_values, ndbi_values, normalized_ndbi_values, normalized_VV_values, normalized_VH_values,  rbr_ndbi_values, rbr_VV_values, rbr_VH_values, x1, x2, y1, y2, burn_classification):
    # Ensure all vectors have the same length
    if not (len(ndpi_values) == len(ndbi_values) == len(normalized_ndbi_values) == len(normalized_VV_values) == len(normalized_VH_values) == len(rbr_ndbi_values) == len(rbr_VV_values) == len(rbr_VH_values)):
        raise ValueError("All vectors must have the same length")

    output_dir = 'vector_outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Full path for the output file
    name = f"{x1}_{x2}_{y1}_{y2}"
    vectors_output = f"vectors_output_{name}_{burn_classification}.csv"
    output_path = os.path.join(output_dir, vectors_output)

    # Open the CSV file and write
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header with burn classification
        csv_writer.writerow(['NDPI', 'NDBI', 'NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI', 'RBR_VV', 'RBR_VH', 'Burn_Classification'])

        # Write data rows
        for i in range(len(ndpi_values)):
            csv_writer.writerow([
                ndpi_values[i],
                ndbi_values[i],
                normalized_ndbi_values[i],
                normalized_VV_values[i],
                normalized_VH_values[i],
                rbr_ndbi_values[i],
                rbr_VV_values[i],
                rbr_VH_values[i],
                burn_classification
            ])

    print(f"Vectors saved to {output_path}")


class BurnClassificationApp:
    def __init__(self, master):
        self.master = master
        master.title("Burn Classification")
        master.geometry("300x200")

        # Label
        self.label = tk.Label(master, text="Classify the selected region:", font=("Arial", 12))
        self.label.pack(pady=20)

        # Burned Button
        self.burned_button = tk.Button(master, text="Burned", command=lambda: self.classify_region("Burned"),
                                       bg="red", fg="white", font=("Arial", 12))
        self.burned_button.pack(pady=10)

        # Not Burned Button
        self.not_burned_button = tk.Button(master, text="Not Burned",
                                           command=lambda: self.classify_region("Not_Burned"),
                                           bg="green", fg="white", font=("Arial", 12))
        self.not_burned_button.pack(pady=10)

        # Classification result
        self.classification = None

    def classify_region(self, classification):
        self.classification = classification
        self.master.destroy()


def select_zone(image_path, ndpi_matrix_path, ndbi_matrix_path, normalized_ndbi_matrix_path, normalized_vh_matrix_path, normalized_VV_matrix_path, rbr_ndbi_matrix_path, rbr_VH_matrix_path, rbr_VV_matrix_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) > 2:
        img = img[..., 0]  # Take first channel if multi-channel

    # Load matrices
    ndpi_data = np.load(ndpi_matrix_path)
    ndpi_matrix = ndpi_data['ndpi']

    ndbi_data = np.load(ndbi_matrix_path)
    ndbi_matrix = ndbi_data['ndbi']

    normalized_ndbi_data = np.load(normalized_ndbi_matrix_path)
    normalized_ndbi_matrix = normalized_ndbi_data['normalized']

    normalized_vh_data = np.load(normalized_vh_matrix_path)
    normalized_vh_matrix = normalized_vh_data['normalized']

    normalized_VV_data = np.load(normalized_VV_matrix_path)
    normalized_VV_matrix = normalized_VV_data['normalized']

    rbr_ndbi_data = np.load(rbr_ndbi_matrix_path)
    rbr_ndbi_matrix = rbr_ndbi_data['rbr']

    rbr_VH_data = np.load(rbr_VH_matrix_path)
    rbr_VH_matrix = rbr_VH_data['rbr']

    rbr_VV_data = np.load(rbr_VV_matrix_path)
    rbr_VV_matrix = rbr_VV_data['rbr']

    # Convert to 3-channel image for drawing
    display_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    clone = img.copy()
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 3, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.rectangle(display_img, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Image", display_img)

    # Create window and set mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.imshow("Image", display_img)

    # Wait for two points to be selected
    while len(points) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None

    # Get coordinates for the region
    x1, y1 = min(points[0][0], points[1][0]), min(points[0][1], points[1][1])
    x2, y2 = max(points[0][0], points[1][0]), max(points[0][1], points[1][1])

    # Close image selection window
    cv2.destroyAllWindows()

    # Open classification window
    root = tk.Tk()
    app = BurnClassificationApp(root)
    root.mainloop()

    # Check if classification was made
    if app.classification is None:
        print("No classification selected. Aborting.")
        return None

    # Extract the region
    region = clone[y1:y2, x1:x2]

    # Extract region from matrices
    region_ndpi = ndpi_matrix[y1:y2, x1:x2]
    ndpi_values = printRegion(region_ndpi, y1, x1, 'NDPI')

    region_ndbi = ndbi_matrix[y1:y2, x1:x2]
    ndbi_values = printRegion(region_ndbi, y1, x1, 'NDBI')

    region_normalized_ndbi = normalized_ndbi_matrix[y1:y2, x1:x2]
    normalized_ndbi_values = printRegion(region_normalized_ndbi, y1, x1, 'Normalized ndbi')

    region_normalized_vh = normalized_vh_matrix[y1:y2, x1:x2]
    normalized_vh_values = printRegion(region_normalized_vh, y1, x1, 'Normalized VH')

    region_normalized_VV = normalized_VV_matrix[y1:y2, x1:x2]
    normalized_VV_values = printRegion(region_normalized_VV, y1, x1, 'Normalized VV')

    region_rbr_ndbi = rbr_ndbi_matrix[y1:y2, x1:x2]
    rbr_ndbi_values = printRegion(region_rbr_ndbi, y1, x1, 'RBR NDBI')

    region_rbr_VV = rbr_VV_matrix[y1:y2, x1:x2]
    rbr_VV_values = printRegion(region_rbr_VV, y1, x1, 'RBR VV')

    region_rbr_VH = rbr_VH_matrix[y1:y2, x1:x2]
    rbr_VH_values = printRegion(region_rbr_VH, y1, x1, 'RBR VH')

    # Save values with burn classification
    save_values_to_csv(
        ndpi_values,
        ndbi_values,
        normalized_ndbi_values,
        normalized_VV_values,
        normalized_vh_values,
        rbr_ndbi_values,
        rbr_VV_values,
        rbr_VH_values,
        x1, x2, y1, y2,
        app.classification
    )

    # Process each pixel in the region
    pixel_values = []

    return region_rbr_ndbi, pixel_values


def printRegion(region, y1, x1, titulo):
    pixel_values = []
    print(f"\n*************: {titulo}")
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            # Check if the region has multiple channels
            if len(region.shape) > 2:
                pixel_value = region[i, j][0]  # Take first channel
            else:
                pixel_value = region[i, j]  # Single channel
            pixel_values.append(pixel_value)
            print(f"Position ({i + y1}, {j + x1}): {pixel_value}")
    return pixel_values


def selected_zone_image():
    image_path = 'Images/Post1/Band_Sigma0_VH_Post_Incendio.tif'
    ndpi_matrix_path = 'Matrixs/ndpi_Post_matrix.npz'
    ndbi_matrix_path = 'Matrixs/ndbi_Post_matrix.npz'
    normalized_ndbi_matrix_path = 'Matrixs/normalized_ndbi_matrix.npz'
    normalized_VH_matrix_path = 'Matrixs/normalized_VH_matrix.npz'
    normalized_VV_matrix_path = 'Matrixs/normalized_VV_matrix.npz'
    rbr_ndbi_matrix_path = 'Matrixs/rbr_ndbi_matrix.npz'
    rbr_VH_matrix_path = 'Matrixs/rbr_VH_matrix.npz'
    rbr_VV_matrix_path = 'Matrixs/rbr_VV_matrix.npz'

    selected_matrix, values = select_zone(
        image_path,
        ndpi_matrix_path,
        ndbi_matrix_path,
        normalized_ndbi_matrix_path,
        normalized_VH_matrix_path,
        normalized_VV_matrix_path,
        rbr_ndbi_matrix_path,
        rbr_VH_matrix_path,
        rbr_VV_matrix_path
    )

    if selected_matrix is not None:
        print(f"Selected region shape: {selected_matrix.shape}")