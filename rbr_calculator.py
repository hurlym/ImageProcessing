import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_rbr(pre_path, post_path, output_folder, band_type):
    # Read pre and post images
    pre_img = cv2.imread(pre_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    post_img = cv2.imread(post_path, cv2.IMREAD_UNCHANGED).astype(np.float64)

    # Take first channel if images are multi-channel
    if len(pre_img.shape) > 2:
        pre_img = pre_img[..., 0]
    if len(post_img.shape) > 2:
        post_img = post_img[..., 0]

    # Print shapes for debugging
    print(f"Pre image shape: {pre_img.shape}")
    print(f"Post image shape: {post_img.shape}")

    # Ensure both images have the same dimensions
    if pre_img.shape != post_img.shape:
        post_img = cv2.resize(post_img, (pre_img.shape[1], pre_img.shape[0]))
        print(f"Resized post image shape: {post_img.shape}")

    # Avoid division by zero
    epsilon = 1e-10

    # Calculate RBR (post/pre)
    rbr = np.divide(post_img, pre_img + epsilon)
    rbr = np.clip(rbr, 0, 10)  # Limit extreme values
    output_path = os.path.join('Matrixs', f'rbr_{band_type}_matrix.npz')
    np.savez_compressed(output_path, rbr=rbr)

"""
 # Save RBR image in TIFF format
    base_name = os.path.basename(post_path)
    output_path = os.path.join(output_folder, f"rbr_{band_type}_{base_name}")
    rbr_save = (rbr * 6553.5).astype(np.uint16)  # Scale to 16-bit range
    cv2.imwrite(output_path, rbr_save)

    # Create visualization with normalized data
    plt.figure(figsize=(10, 10))
    rbr_viz = rbr / 2  # Normalize to [0,1] for visualization
    plt.imshow(rbr_viz, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(label=f'RBR {band_type}')
    plt.title(f'RBR {band_type} Visualization')

    # Save visualization
    viz_path = os.path.join(output_folder, f"rbr_{band_type}_viz_{base_name.replace('.tif', '.png')}")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return rbr
"""


def calculate_matrix_division(matrix1_path, matrix2_path, output_path, epsilon=1e-10):
    # Load the matrices from npz files
    matrix1_data = np.load(matrix1_path)
    matrix2_data = np.load(matrix2_path)

    # Get the actual arrays (assuming they're stored with known keys)
    matrix1 = matrix1_data['ndbi']  # or 'rbr' depending on which matrix you're using
    matrix2 = matrix2_data['ndbi']  # adjust key name as needed

    # Calculate division (avoiding division by zero)
    rbr = np.divide(matrix1, matrix2 + epsilon)

    # Clip results to handle extreme values
    rbr = np.clip(rbr, 0, 10)

    # Save the result
    np.savez_compressed(output_path, rbr=rbr)

    return rbr

def process_folder_rbr(pre_folder, post_folder):
    output_folder = os.path.join(os.path.dirname(pre_folder), 'RBR')
    os.makedirs(output_folder, exist_ok=True)

    # Process VV band
    pre_vv = [f for f in os.listdir(pre_folder)if 'VV' in f][0]
    post_vv = [f for f in os.listdir(post_folder)if 'VV' in f][0]

    print("Calculating RBR for VV band...")
    calculate_rbr(
        os.path.join(pre_folder, pre_vv),
        os.path.join(post_folder, post_vv),
        output_folder,
        'VV'
    )

    # Process VH band
    pre_vh = [f for f in os.listdir(pre_folder) if 'VH' in f][0]
    post_vh = [f for f in os.listdir(post_folder) if 'VH' in f][0]

    print("Calculating RBR for VH band...")
    calculate_rbr(
        os.path.join(pre_folder, pre_vh),
        os.path.join(post_folder, post_vh),
        output_folder,
        'VH'
    )

    # Example usage:
    matrix1_path = os.path.join('Matrixs', 'ndbi_Post_matrix.npz')
    matrix2_path = os.path.join('Matrixs', 'ndbi_Pre_matrix.npz')
    output_path = os.path.join('Matrixs', 'rbr_ndbi_matrix.npz')

    result = calculate_matrix_division(matrix1_path, matrix2_path, output_path)


"""
NO SE PUEDE CALCULAR COMO LOS OTROS PORQUE NO ES UNA IMAGEN
# Process NDBI band
    pre_ndbi = [f for f in os.listdir(os.path.join(pre_folder, 'NDBI'))]
    post_ndbi = [f for f in os.listdir(os.path.join(post_folder, 'NDBI'))]

    print("Calculating RBR for NDBI band...")
    calculate_rbr(
        os.path.join(pre_folder, 'NDBI', pre_ndbi),
        os.path.join(post_folder, 'NDBI', post_ndbi),
        output_folder,
        'NDBI'
    )
"""
