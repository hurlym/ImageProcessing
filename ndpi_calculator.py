import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_ndpi(vv_path, hv_path, output_folder, origin):
    # Leer imágenes VV y HV
    vv_img = cv2.imread(vv_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    hv_img = cv2.imread(hv_path, cv2.IMREAD_UNCHANGED).astype(np.float64)

    # Evitar división por cero
    epsilon = 1e-10

    # Calcular NDPI
    ndpi = np.divide(vv_img - hv_img, vv_img + hv_img + epsilon)
    ndpi = np.clip(ndpi, -1, 1)
    output_path = os.path.join('Matrixs', f'ndpi_{origin}_matrix.npz')
    np.savez_compressed(output_path, ndpi=ndpi)
"""
 # Guardar imagen NDPI en formato TIFF
    base_name = os.path.basename(vv_path)
    output_path = os.path.join(output_folder, f"ndpi_{base_name}")
    ndpi_save = ((ndpi + 1) * 32767).astype(np.uint16)
    cv2.imwrite(output_path, ndpi_save)

    # Crear visualización
    plt.figure(figsize=(10, 10))
    ndpi_viz = (ndpi + 1) / 2  # Normalize to [0,1] range
    plt.imshow(ndpi_viz, cmap='RdYlBu', vmin=0, vmax=1)
    plt.colorbar(label='NDPI')
    plt.title('NDPI Visualization')

    # Guardar visualización
    viz_path = os.path.join(output_folder, f"ndpi_viz_{base_name.replace('.tif', '.png')}")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return ndpi
"""



def process_folder_ndpi(folder_path, origin):
    output_folder = os.path.join(folder_path, 'NDPI')
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(folder_path)]
    vv_files = [f for f in files if 'VV' in f]
    hv_files = [f for f in files if 'HV' in f]

    for vv_file in vv_files:
        base_name = vv_file.replace('VV', 'HV')
        if base_name in hv_files:
            vv_path = os.path.join(folder_path, vv_file)
            hv_path = os.path.join(folder_path, base_name)

            print(f"Calculating NDPI for {vv_file}")
            calculate_ndpi(vv_path, hv_path, output_folder, origin)