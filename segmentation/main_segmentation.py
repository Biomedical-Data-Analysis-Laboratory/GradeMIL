import os
from skimage.measure import regionprops
from skimage import morphology
from scipy import ndimage
from sklearn.cluster import KMeans
import numpy as np
import pyvips
import math
import pickle
import time
import cv2
import pandas as pd
import utils_segmentation

############

# Set this option to True to save tiles as JPEG images in a separate folder.
save_tiles_as_jpeg = True

# Set this option to True to save the binary annotation mask as a JPEG image.
save_binary_annotation_mask = False

# Set this option to True to save region masks as a JPEG image.
save_region_annotation_mask = True

# In the WSI folder, there is a file containing a dict with the 7 binary masks.
# To specify which of these masks to use, list the tissue types in the following variable.
# Available options: ['urothelium', 'stroma', 'muscle', 'blood', 'damaged', 'background']
tissue_classes_to_fit_tiles_on = ['urothelium']

# How large percentage of the tile must cover the region to be consider a valid tile.
# Float between 0-1.
PHI = 0.7

# Which level in the WSI to use when checking for valid tiles (level 0=400x, level 1=100x, and level 2=25x).
# Available options: '25x', '100x', '400x'.
ALPHA = '400x'

# All valid tiles are displayed on the 25x image and saved as JPEG to the folder.
# This option determines which of the three levels to include in the final image.
# Tiles from all three levels in the WSI are saved, this option is only for visualization.
# Available options: ['25x', '100x', '400x'].
TILES_TO_SHOW = ['400x']

# Size of width/height of the tiles to extract. Integer.
TILE_SIZE = 128

# The level the annotation mask is on, and also the level our tiles are on. For our mask it's '25x'.
TAU = '25x'

# The binary masks contain small regions which is not of interest.
# These are removed using the remove_small_objects() function.
# This variable sets the minimum size to remove.
# Available options: Integer values, usually between 500 and 20000.
SMALL_REGION_REMOVE_THRESHOLD = 3000

# Minimum number of tiles a region must contain
# Any regions with less tiles will be discarded
SMALL_REGION_TILE_THRESHOLD = 500

# Maximum number of tiles a region must contain
# Any regions with more tiles will be clustered into int(len(region_tiles)/BIG_REGION_TILE_THRESHOLD)
BIG_REGION_TILE_THRESHOLD = 2000

# Paths
wsi_dataset_file_path = 'WSIs/'
output_folder = 'Tiles_dataset/'
if save_tiles_as_jpeg:
    os.makedirs(output_folder, exist_ok=True)

# Create a dict containing the ratio for each level
ratio_dict = dict()
ratio_dict['400x'] = 1
ratio_dict['100x'] = 4
ratio_dict['25x'] = 16

# Create a dict containin the index of each class
tissue_class_to_index = dict()
tissue_class_to_index['background'] = 0
tissue_class_to_index['blood'] = 1
tissue_class_to_index['damaged'] = 2
tissue_class_to_index['muscle'] = 3
tissue_class_to_index['stroma'] = 4
tissue_class_to_index['urothelium'] = 5
tissue_class_to_index['undefined'] = 6

# Loop through each WSI
for wsi_name in os.listdir(wsi_dataset_file_path):

    # Start timer
    current_wsi_start_time = time.time()
    
    # Variable initialization
    list_of_valid_tiles_from_current_wsi = 0

    # Inter magnification tracker
    df_csv = pd.DataFrame.from_dict({
    'Path_400x': list(),
    'Path_100x': list(),
    'Path_25x': list(),
    'Reg_ID': list()
    })

    # Load annotation mask. For us, this is a pickle file containing the annotation mask for all tissue classes.
    # You should obtain the pickle file after running the tissue segmentation algorithm from:
    # https://github.com/Biomedical-Data-Analysis-Laboratory/multiscale-tissue-segmentation-for-urothelial-carcinoma
    annotation_mask_path = wsi_dataset_file_path + wsi_name + '/Pickle_files/COLORMAP_IMAGES_PICKLE.obj'
    with open(annotation_mask_path, 'rb') as handle:
        annotation_mask_all_classes = pickle.load(handle)

    # Read images
    # Our files used SCN file format
    print("WSI: " + wsi_name)
    full_image_400 = pyvips.Image.new_from_file(wsi_dataset_file_path + '/' + wsi_name + '.scn', level=0).flatten().rot(1)
    full_image_100 = pyvips.Image.new_from_file(wsi_dataset_file_path + '/' + wsi_name + '.scn', level=1).flatten().rot(1)
    full_image_25 = pyvips.Image.new_from_file(wsi_dataset_file_path + '/' + wsi_name + '.scn', level=2).flatten().rot(1)

    # Save overview image
    offset_x_x25, offset_y_x25, width_25x, height_25x = utils_segmentation.remove_white_background_v3(input_img=full_image_25, PADDING=0, folder_path=wsi_dataset_file_path + '/' + wsi_name)
    cropped_image_25 = full_image_25.extract_area(offset_x_x25, offset_y_x25, width_25x, height_25x)
    cropped_image_25.jpegsave(output_folder + wsi_name + '/thumbnail.jpeg', Q=100)
    full_image_100 = full_image_100.extract_area(int(4*offset_x_x25), int(4*offset_y_x25), int(4*width_25x), int(4*height_25x))
    full_image_400 = full_image_400.extract_area(int(16*offset_x_x25), int(16*offset_y_x25), int(16*width_25x), int(16*height_25x))
    
    # Find width/height of 25x image
    scn_width_25x = cropped_image_25.width
    scn_height_25x = cropped_image_25.height

    # Loop through each tissue class to fit tiles on
    for current_class_to_copy in tissue_classes_to_fit_tiles_on:
        print('Now processing {} regions'.format(current_class_to_copy))

        # Extract mask for current class
        current_class_mask = annotation_mask_all_classes[tissue_class_to_index[current_class_to_copy]].copy()

        # Resize colormap to the size of 25x overview image
        current_class_mask = cv2.resize(current_class_mask, dsize=(scn_width_25x, scn_height_25x), interpolation=cv2.INTER_CUBIC)
        print('Loaded segmentation mask with size {} x {}'.format(current_class_mask.shape[1], current_class_mask.shape[0]))

        # Apply closing operation to fill gaps between cells
        current_class_mask = cv2.morphologyEx(current_class_mask, cv2.MORPH_CLOSE, np.ones((3,3)))

        # Save the annotation mask image (If option is set to True)
        if save_binary_annotation_mask:
            annotation_mask_for_saving = current_class_mask * 255
            cv2.imwrite(output_folder + wsi_name + '/Binary_annotation_mask_{}.jpg'.format(current_class_to_copy), annotation_mask_for_saving, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Use a boolean condition to find where pixel values are > 0.75
        blobs = current_class_mask > 0.75

        # label connected regions that satisfy this condition
        labels, regions_found_in_wsi_before_removing_small_obj = ndimage.label(blobs)
        print('\tFound {} regions'.format(regions_found_in_wsi_before_removing_small_obj))

        # Remove all the small regions
        labels = morphology.remove_small_objects(labels, min_size=SMALL_REGION_REMOVE_THRESHOLD)

        # Get region properties
        list_of_regions = regionprops(labels)

        n_regions_after_removing_small_obj = len(list_of_regions)
        print('\tFound {} regions (after removing small objects)'.format(n_regions_after_removing_small_obj))

        if save_region_annotation_mask:
            # Read overview image again using cv2, and add alpha channel to overview image.
            overview_jpeg_file = cv2.imread(output_folder + wsi_name + '/thumbnail.jpeg', cv2.IMREAD_UNCHANGED)
            overview_jpeg_file = np.dstack([overview_jpeg_file, np.ones((overview_jpeg_file.shape[0], overview_jpeg_file.shape[1]), dtype="uint8") * 255])

        # Create a grid of all possible x- and y-coordinates (starting position (0,0))
        all_x_pos, all_y_pos = [], []
        for x_pos in range(0, int(current_class_mask.shape[1] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_x_pos.append(x_pos)
        for y_pos in range(0, int(current_class_mask.shape[0] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_y_pos.append(y_pos)

        # Extract all coordinates (to draw region on overview image)
        region_count = -1
        overview_uro_jpeg_file = overview_jpeg_file.copy()

        for current_region in list_of_regions:
            region_masks = dict()
            region_masks[current_class_to_copy] = np.zeros(shape=(current_class_mask.shape[0], current_class_mask.shape[1]))
            for current_region_coordinate in current_region.coords:
                region_masks[current_class_to_copy][current_region_coordinate[0], current_region_coordinate[1]] = 1
    
            # Create a new list with all xy-positions in current SCN image
            list_of_valid_tiles_from_current_class = []
            for y_pos in all_y_pos:
                for x_pos in all_x_pos:
                    # A tile from a given region has to be PHI percentage of urothelium mask in order to be eligible
                    if int(sum(sum(region_masks[current_class_to_copy][y_pos:y_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])),
                                   x_pos:x_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))]))) >= (pow((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), 2) * PHI):
                        list_of_valid_tiles_from_current_class.append((x_pos, y_pos))

            # If there is not a sufficient number of tiles in the region, discard it
            if len(list_of_valid_tiles_from_current_class) < SMALL_REGION_TILE_THRESHOLD:
                continue
            # If the region is too big, split into several subregions
            elif len(list_of_valid_tiles_from_current_class) > BIG_REGION_TILE_THRESHOLD:
                resulting_number_of_regions = int(math.ceil(len(list_of_valid_tiles_from_current_class)/BIG_REGION_TILE_THRESHOLD))
                kmeans = KMeans(n_clusters=resulting_number_of_regions, random_state=0).fit(list_of_valid_tiles_from_current_class)
                resulting_cluster_labels = kmeans.labels_
            # If region size is within the acceptable range, simply proceed
            else:
                resulting_number_of_regions = 1
                resulting_cluster_labels = [0] * len(list_of_valid_tiles_from_current_class)

            # Convert masks from 0-1 -> 0-255 (can also be used to set the color)
            region_masks[current_class_to_copy] *= 255

            # Resize masks to same size as the overview image
            region_masks[current_class_to_copy] = cv2.resize(region_masks[current_class_to_copy], dsize=(overview_jpeg_file.shape[1], overview_jpeg_file.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Create a empty alpha channel
            alpha_channel = np.zeros(shape=(region_masks[current_class_to_copy].shape[0], region_masks[current_class_to_copy].shape[1]))

            # Each mask is 1-channel, merge them to create a 3-channel image (RGB), the order is used to set the color for each mask. Add a alpha-channel.
            region_masks[current_class_to_copy] = cv2.merge((region_masks[current_class_to_copy], alpha_channel, alpha_channel, alpha_channel))

            # Draw the selected regions on the overview image
            for _, current_tissue_mask in region_masks.items():
                overview_region_jpeg_file = cv2.addWeighted(current_tissue_mask, 1, overview_jpeg_file, 1.0, 0, dtype=cv2.CV_64F)

            for subregion_index in range(0,resulting_number_of_regions):
                region_count += 1
                os.mkdir(output_folder + wsi_name + '/' + str(region_count))
                list_of_valid_tiles_from_current_subregion = [coord_x_y for coord_x_y, label in zip(list_of_valid_tiles_from_current_class,resulting_cluster_labels) if label == subregion_index]

                if save_region_annotation_mask:

                    # To avoid overlapping rectangles
                    overview_subregion_jpeg_file = overview_jpeg_file.copy()

                    # Draw tiles on the overview image
                    for current_xy_pos in list_of_valid_tiles_from_current_subregion:
                        start_x = dict()
                        start_y = dict()
                        end_x = dict()
                        end_y = dict()

                        # Equation 3 in paper.
                        for BETA in ['25x', '100x', '400x']:
                            start_x[BETA] = int(((current_xy_pos[0] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['25x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x'])) / 2))
                            start_y[BETA] = int(((current_xy_pos[1] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['25x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x'])) / 2))
                            end_x[BETA] = int(start_x[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x']))
                            end_y[BETA] = int(start_y[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x']))

                        # Draw tiles (Red tiles indicate which level ALPHA is, and the corresponding levels are shown in green)
                        for draw_level in TILES_TO_SHOW:
                            color = (0, 0, 255) if draw_level == ALPHA else (0, 255, 0)
                            cv2.rectangle(overview_uro_jpeg_file, (start_x[draw_level], start_y[draw_level]), (end_x[draw_level], end_y[draw_level]), color, 3)
                            cv2.rectangle(overview_subregion_jpeg_file, (start_x[draw_level], start_y[draw_level]), (end_x[draw_level], end_y[draw_level]), color, 3)

                    # Save overview image
                    cv2.imwrite(output_folder + wsi_name + '/reg_{}_alpha_{}_phi_{}.jpeg'.format(region_count, ALPHA, PHI), overview_subregion_jpeg_file, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
                # Add the tiles to the list of tiles of current wsi
                list_of_valid_tiles_from_current_wsi += len(list_of_valid_tiles_from_current_subregion) 
                tile_x = dict()
                tile_y = dict()
                
                # Save coordinates for each tiles to a dict to create a dataset
                os.mkdir(output_folder + wsi_name + '/' + str(region_count) + '/400x/')
                os.mkdir(output_folder + wsi_name + '/' + str(region_count) + '/100x/')
                os.mkdir(output_folder + wsi_name + '/' + str(region_count) + '/25x/')

                for current_xy_pos in list_of_valid_tiles_from_current_subregion:
        
                    # Obtain the coordinates of tiles from the same triplet, taking the base magnification and projecting its coordinates
                    for BETA in ['25x', '100x', '400x']:
                        tile_x[BETA] = int(offset_x_x25 * (ratio_dict[TAU] / ratio_dict[BETA]) + (current_xy_pos[0] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2)
                        tile_y[BETA] = int(offset_y_x25 * (ratio_dict[TAU] / ratio_dict[BETA]) + (current_xy_pos[1] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2)
        
                    # Save tile to coordinate dict (coordinate of top-left corner)
                    tile_400x = full_image_400.extract_area(int(tile_x['400x']), int(tile_y['400x']), TILE_SIZE, TILE_SIZE)
                    tile_400x_filename = output_folder + wsi_name + '/' + str(region_count) + '/400x/' + str(tile_x['400x']) + '_' + str(tile_y['400x']) + '.jpeg'
                    tile_400x.jpegsave(tile_400x_filename, Q=100)

                    tile_100x = full_image_100.extract_area(int(tile_x['100x']), int(tile_y['100x']), TILE_SIZE, TILE_SIZE)
                    tile_100x_filename = output_folder + wsi_name + '/' + str(region_count) + '/100x/' + str(tile_x['100x']) + '_' + str(tile_y['100x']) + '.jpeg'
                    tile_100x.jpegsave(tile_100x_filename, Q=100)

                    tile_25x = full_image_25.extract_area(int(tile_x['25x']), int(tile_y['25x']), TILE_SIZE, TILE_SIZE)
                    tile_25x_filename = output_folder + wsi_name + '/' + str(region_count) + '/25x/' + str(tile_x['25x']) + '_' + str(tile_y['25x']) + '.jpeg'
                    tile_25x.jpegsave(tile_25x_filename, Q=100)

                    df_csv.loc[len(df_csv)] = [tile_400x_filename, tile_100x_filename, tile_25x_filename, region_count]

    cv2.imwrite(output_folder + wsi_name + '/uro_mask.jpeg', overview_uro_jpeg_file, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    df_csv.to_csv(output_folder + wsi_name + '/tile_info.csv')
    
    # Calculate elapse time for current run
    elapse_time = time.time() - current_wsi_start_time
    m, s = divmod(elapse_time, 60)
    h, m = divmod(m, 60)
    model_time = '%02d:%02d:%02d' % (h, m, s)

    # Print out results
    print('\tFound {} regions (after discarding small regions)'.format(region_count))
    print('Found {} tiles in image'.format(list_of_valid_tiles_from_current_wsi))
    print('Finished. Duration: {}'.format(model_time))