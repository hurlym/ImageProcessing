import cv2
import numpy as np
import csv

def save_values_to_csv(ndpi_values, ndbi_values, normalized_values, rbr_values, x1, x2, y1, y2):
    # Ensure all vectors have the same length
    if not (len(ndpi_values) == len(ndbi_values) == len(normalized_values) == len(rbr_values)):
        raise ValueError("All vectors must have the same length")

    import os
    output_dir = 'vector_outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Full path for the output file
    name = str(x1) + "_" + str(x2) + "_" + str(y1) + "_" + str(y2)
    vectors_output = f"vectors_output_{name}.csv"
    output_path = os.path.join(output_dir, vectors_output)

    # Open the CSV file and write
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(['Index', 'NDPI', 'NDBI', 'NORMALIZED', 'RBR'])

        # Write data rows
        for i in range(len(ndpi_values)):
            csv_writer.writerow([i, ndpi_values[i], ndbi_values[i], normalized_values[i], rbr_values[i]])

    print(f"Vectors saved to {output_path}")


def printRegion (region, y1, x1, titulo):
    pixel_values = []
    print(f"\n*************: {titulo}")
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            pixel_value = region[i, j][0]
            pixel_values.append(pixel_value)
            print(f"Position ({i + y1}, {j + x1}): {pixel_value}")
    return pixel_values
def select_zone(image_path, ndpi_matrix_path, ndbi_matrix_path, normalized_matrix_path, rbr_matrix_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) > 2:
        img = img[..., 0]  # Take first channel if multi-channel

    # Load NDPI matrix
    ndpi_data = np.load(ndpi_matrix_path)
    ndpi_matrix = ndpi_data['ndpi']

    # Load NDBI matrix
    ndbi_data = np.load(ndbi_matrix_path)
    ndbi_matrix = ndbi_data['ndbi']

    # Load Normalized matrix
    normalized_data = np.load(normalized_matrix_path)
    normalized_matrix = normalized_data['normalized']

    # Load RBR matrix
    rbr_data = np.load(rbr_matrix_path)
    rbr_matrix = rbr_data['rbr']

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

    # Extract the region
    region = clone[y1:y2, x1:x2]
    #printRegion(region, y1, x1, 'REGION')

    # Extract region from NDPI matrix
    region_ndpi = ndpi_matrix[y1:y2, x1:x2]
    ndpi_values = printRegion(region_ndpi, y1, x1, 'NDPI')

    # Extract region from NDBI matrix
    region_ndbi = ndbi_matrix[y1:y2, x1:x2]
    ndbi_values = printRegion(region_ndbi, y1, x1, 'NDBI')

    # Extract region from Normalized matrix
    region_normalized = normalized_matrix[y1:y2, x1:x2]
    normalized_values = printRegion(region_normalized, y1, x1, 'Normalized')

    # Extract region from RBR matrix
    region_rbr = rbr_matrix[y1:y2, x1:x2]
    rbr_values = printRegion(region_rbr, y1, x1, 'RBR')

    save_values_to_csv(ndpi_values, ndbi_values, normalized_values, rbr_values, x1, x2, y1, y2)
    # Process each pixel in the region
    pixel_values = []



    cv2.destroyAllWindows()
    return region_rbr, pixel_values


def selected_zone_image():
    image_path = 'Images/Post1/Band_Sigma0_VH_Post_Incendio.tif'
    ndpi_matrix_path = 'Matrixs/ndpi_Post_matrix.npz'
    ndbi_matrix_path = 'Matrixs/ndbi_Post_matrix.npz'
    normalized_matrix_path = 'Matrixs/normalized_ndbi_matrix.npz'
    rbr_matrix_path = 'Matrixs/rbr_ndbi_matrix.npz'
    selected_matrix, values = select_zone(image_path, ndpi_matrix_path, ndbi_matrix_path, normalized_matrix_path, rbr_matrix_path)
    if selected_matrix is not None:
        print(f"Selected region shape: {selected_matrix.shape}")
        print(f"Matrix values:\n{selected_matrix}")
