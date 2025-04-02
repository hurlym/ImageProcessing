import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_normalized(pre_path, post_path, output_folder, band_type):
    # Read pre and post images
    pre_img = cv2.imread(pre_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    post_img = cv2.imread(post_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

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

    # Calculate Normalized difference
    normalized = np.divide(post_img - pre_img, post_img + pre_img + epsilon)
    normalized = np.clip(normalized, -1, 1)  # Clip to [-1, 1] range
    output_path = os.path.join('Matrixs', f'normalized_{band_type}_matrix.npz')
    np.savez_compressed(output_path, normalized=normalized)
"""
# Save Normalized image in TIFF format
    base_name = os.path.basename(post_path)
    output_path = os.path.join(output_folder, f"normalized_{band_type}_{base_name}")
    normalized_save = ((normalized + 1) * 32767).astype(np.uint16)  # Scale to 16-bit range
    cv2.imwrite(output_path, normalized_save)

    # Create visualization
    plt.figure(figsize=(10, 10))
    normalized_viz = (normalized + 1) / 2  # Normalize to [0,1] range for visualization
    plt.imshow(normalized_viz, cmap='RdYlBu', vmin=0, vmax=1)
    plt.colorbar(label=f'Normalized {band_type}')
    plt.title(f'Normalized {band_type} Visualization')

    # Save visualization
    viz_path = os.path.join(output_folder, f"normalized_{band_type}_viz_{base_name.replace('.tif', '.png')}")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return normalized
"""
def calculate_normalized_ndpi_division(matrix1_path, matrix2_path, output_path, epsilon=1e-10):
    # Load the matrices from npz files
    matrix1_data = np.load(matrix1_path)
    matrix2_data = np.load(matrix2_path)

    # Get the actual arrays (assuming they're stored with known keys)
    matrix1 = matrix1_data['ndbi']  # or 'rbr' depending on which matrix you're using
    matrix2 = matrix2_data['ndbi']  # adjust key name as needed

    # Calculate numerator (post - pre)
    numerator = matrix1 - matrix2

    # Calculate denominator (post + pre + epsilon)
    denominator = matrix1 + matrix2 + epsilon

    # Calculate final normalized result
    result = np.divide(numerator, denominator)
    result = np.clip(result, -1, 1)

    # Save the result
    np.savez_compressed(output_path, normalized=result)

    return result

def process_folder_normalized(pre_folder, post_folder):
    output_folder = os.path.join(os.path.dirname(pre_folder), 'Normalized')
    os.makedirs(output_folder, exist_ok=True)

    # Process VV band
    pre_vv = [f for f in os.listdir(pre_folder)if 'VV' in f][0]
    post_vv = [f for f in os.listdir(post_folder)if 'VV' in f][0]

    print("Calculating Normalized for VV band...")
    calculate_normalized(
        os.path.join(pre_folder, pre_vv),
        os.path.join(post_folder, post_vv),
        output_folder,
        'VV'
    )

    # Process VH band
    pre_vh = [f for f in os.listdir(pre_folder)if 'VH' in f][0]
    post_vh = [f for f in os.listdir(post_folder)if 'VH' in f][0]

    print("Calculating Normalized for VH band...")
    calculate_normalized(
        os.path.join(pre_folder, pre_vh),
        os.path.join(post_folder, post_vh),
        output_folder,
        'VH'
    )
    # Example usage:
    matrix1_path = os.path.join('Matrixs', 'ndbi_Post_matrix.npz')
    matrix2_path = os.path.join('Matrixs', 'ndbi_Pre_matrix.npz')
    output_path = os.path.join('Matrixs', 'normalized_ndbi_matrix.npz')

    result = calculate_normalized_ndpi_division(matrix1_path, matrix2_path, output_path)


"""
# Process NDBI band
    pre_ndbi = [f for f in os.listdir(os.path.join(pre_folder, 'NDBI'))]
    post_ndbi = [f for f in os.listdir(os.path.join(post_folder, 'NDBI'))]

    print("Calculating Normalized for NDBI band...")
    calculate_normalized(
        os.path.join(pre_folder, 'NDBI', pre_ndbi),
        os.path.join(post_folder, 'NDBI', post_ndbi),
        output_folder,
        'NDBI'
    )
"""
