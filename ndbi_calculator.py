import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_ndbi(vv_path, vh_path, output_folder, origin):
    # Read VV and VH images - take only the first channel
    vv_img = cv2.imread(vv_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    vh_img = cv2.imread(vh_path, cv2.IMREAD_UNCHANGED).astype(np.float64)

    # Ensure both images have the same dimensions by resizing VH to match VV
    if vv_img.shape != vh_img.shape:
        vh_img = cv2.resize(vh_img, (vv_img.shape[1], vv_img.shape[0]))
        print(f"Resized VH image shape: {vh_img.shape}")

    # Avoid division by zero
    epsilon = 1e-10

    # Calculate NDBI
    ndbi = np.divide(vv_img - vh_img, vv_img + vh_img + epsilon)
    ndbi = np.clip(ndbi, -1, 1)
    output_path = os.path.join('Matrixs', f'ndbi_{origin}_matrix.npz')
    np.savez_compressed(output_path, ndbi=ndbi)

"""
# Save NDBI image in TIFF format
    base_name = os.path.basename(vv_path)
    output_path = os.path.join(output_folder, f"ndbi_{base_name}")
    ndbi_save = ((ndbi + 1) * 32767).astype(np.uint16)
    cv2.imwrite(output_path, ndbi_save)

    # Create visualization with normalized data
    plt.figure(figsize=(10, 10))
    ndbi_viz = (ndbi + 1) / 2  # Normalize to [0,1] range
    plt.imshow(ndbi_viz, cmap='RdYlBu', vmin=0, vmax=1)
    plt.colorbar(label='NDBI')
    plt.title('NDBI Visualization')

    # Save visualization
    viz_path = os.path.join(output_folder, f"ndbi_viz_{base_name.replace('.tif', '.png')}")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return ndbi
"""



def process_folder_ndbi(folder_path, origin):
    output_folder = os.path.join(folder_path, 'NDBI')
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(folder_path)]
    vv_files = [f for f in files if 'VV' in f]
    vh_files = [f for f in files if 'VH' in f]

    for vv_file in vv_files:
        base_name = vv_file.replace('VV', 'VH')
        if base_name in vh_files:
            vv_path = os.path.join(folder_path, vv_file)
            vh_path = os.path.join(folder_path, base_name)

            print(f"Calculating NDBI for {vv_file}")
            calculate_ndbi(vv_path, vh_path, output_folder, origin)