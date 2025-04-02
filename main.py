from multilook_utils import multilook_process
from ndpi_calculator import process_folder_ndpi
from ndbi_calculator import process_folder_ndbi
from rbr_calculator import process_folder_rbr
from normalized_calculator import process_folder_normalized
from selectedZoneImage2 import selected_zone_image

# Usage
folder_path_Pre = 'Images/Pre1'
folder_path_Post = 'Images/Post1'
# Procesa las imagenes de la carpeta Pre
#multilook_process(folder_path_Pre)
# Procesa las imagenes de la carpeta Post
#multilook_process(folder_path_Post)

# Calcular NDPI
process_folder_ndpi(folder_path_Pre, 'Pre')
process_folder_ndpi(folder_path_Post, 'Post')

# Calculate NDBI
process_folder_ndbi(folder_path_Pre, 'Pre')
process_folder_ndbi(folder_path_Post, 'Post')

# Calculate RBR
process_folder_rbr(folder_path_Pre, folder_path_Post)

# Calculate Normalized difference
process_folder_normalized(folder_path_Pre, folder_path_Post)

for i in range(6):
    selected_zone_image()
