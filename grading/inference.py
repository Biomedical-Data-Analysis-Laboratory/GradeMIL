import os
import pyvips
import sklearn
import torch
import numpy as np
import datetime
import cv2
import sklearn.metrics
from timeit import default_timer as timer
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import glob
from torchsummary import summary

import skimage
from skimage.filters import rank
from skimage.morphology import disk
from skimage.transform import resize
import torch.nn.functional as F
# from torchsummary import summary
import random
from PIL import Image
from scipy import ndimage
import pandas as pd

import utils_grading

class NMILInference():
    def __init__(self, dir_out, network, dir_inference, id='', ori_patch_size=128):

        self.dir_results = dir_out
        self.dir_inference = dir_inference
        if not os.path.exists(self.dir_results + self.dir_inference):
            os.mkdir(self.dir_results + self.dir_inference)

        # Other
        self.init_time = 0
        self.network = network
        self.metrics = {}
        self.id = id
        self.ori_patch_size = ori_patch_size

    def infer(self, current_wsi_subfolder):
        self.current_wsi_subfolder = current_wsi_subfolder
        self.current_wsi_name = self.current_wsi_subfolder.split('/')[-1]
        self.preds_train = []
        self.refs_train = []
        self.images_400x = []
        self.images_100x = []
        self.images_25x = []
        self.region_id = []
        self.current_tile_info_dataframe = pd.read_csv(os.path.join(self.current_wsi_subfolder, 'tile_info.csv'))

        # Move network to gpu
        self.network.cuda()
        self.network.eval()

        self.init_time = timer()

        if not os.path.exists(self.dir_results + self.dir_inference + self.current_wsi_name):
            os.mkdir(self.dir_results + self.dir_inference + self.current_wsi_name)

        full_wsi_filename = 'WSIs/' + self.current_wsi_name + '.scn'

        if not os.path.exists(self.dir_results + self.dir_inference + self.current_wsi_name):
            os.mkdir(self.dir_results + self.dir_inference + self.current_wsi_name)

            full_image_25 = pyvips.Image.new_from_file(full_wsi_filename, level=2).flatten().rot(1)
            offset_x_x25, offset_y_x25, width_25x, height_25x = utils_grading.remove_white_background_v3(input_img=full_image_25, PADDING=0, folder_path='')
            wsi_thumbnail = full_image_25.extract_area(offset_x_x25, offset_y_x25, width_25x, height_25x)

            if not os.path.exists(self.dir_results + self.dir_inference + self.current_wsi_name + '/patch_classification.npy'):

                # Loop over training dataset
                print('[Inference]: {}'.format(self.current_wsi_subfolder.split('/')[-1]))
                wsi_thumbnail.jpegsave(self.dir_results + self.dir_inference + self.current_wsi_name + '/thumbnail.jpeg', Q=100)
                current_number_of_regions = len([j for j in os.listdir(self.current_wsi_subfolder) if '.' not in j])

                for reg_id in range(current_number_of_regions):

                    # Current region dataframe
                    current_reg_info_dataframe = self.current_tile_info_dataframe.loc[self.current_tile_info_dataframe['Reg_ID'] == reg_id]

                    for current_tile_id in range(len(current_reg_info_dataframe)):

                        current_dataframe_row = current_reg_info_dataframe.iloc[current_tile_id]

                        name_400x = current_dataframe_row['Path_400x']
                        name_100x = current_dataframe_row['Path_100x']
                        name_25x = current_dataframe_row['Path_25x']

                        self.images_400x.append(name_400x)
                        self.images_100x.append(name_100x)
                        self.images_25x.append(name_25x)
                        self.region_id.append(int(current_dataframe_row['Reg_ID'].item()))
                
                # Feature extraction
                self.indexes = np.arange(len(self.images_400x))

                features = []
                for iInstance in self.indexes:
                    print(str(iInstance + 1) + '/' + str(len(self.indexes)), end='\r')

                    # Load patch
                    x400 = Image.open(self.images_400x[self.indexes[iInstance]])
                    x400 = np.asarray(x400).astype('float32')
                    x100 = Image.open(self.images_100x[self.indexes[iInstance]])
                    x100 = np.asarray(x100).astype('float32')
                    x25 = Image.open(self.images_25x[self.indexes[iInstance]])
                    x25 = np.asarray(x25).astype('float32')

                    x400 = self.image_normalization(x400)
                    x100 = self.image_normalization(x100)
                    x25 = self.image_normalization(x25)
                    
                    X400 = torch.tensor(x400).cuda().float()[None, ...]
                    X100 = torch.tensor(x100).cuda().float()[None, ...]
                    X25 = torch.tensor(x25).cuda().float()[None, ...]

                    # Transform image into low-dimensional embedding
                    features.append(torch.squeeze(self.network.bb(X400, X100, X25)).detach().cpu().numpy())
                print('CNN Features: done')

                # Compute attention and bag prediction
                features = torch.tensor(np.array(features)).cuda().float()
                region_info = torch.tensor(np.array(self.region_id)).cuda().float()
                instance_classification = []
                region_embeddings_list = []
                for region_id in range(int(torch.max(region_info).item())+1):
                    region_features = features[torch.where(region_info == region_id, True, False)]

                    A_V = self.network.milAggregation.attentionModule.attention_V(region_features)  # Attention
                    A_U = self.network.milAggregation.attentionModule.attention_U(region_features)  # Gate
                    w_logits = self.network.milAggregation.attentionModule.attention_weights(A_V * A_U)  # Probabilities - softmax over instances
                    instance_classification.append(w_logits)

                    if not len(region_features) == 1:
                        embedding_reg, _ = self.network.milAggregation(torch.squeeze(region_features))
                        region_embeddings_list.append(embedding_reg)
                    else:
                        region_embeddings_list.append(torch.squeeze(region_features, dim=0))

                if len(region_embeddings_list) == 1:
                    patch_classification = w_logits
                    embedding = embedding_reg
                    A_V = self.network.milAggregation.attentionModule.attention_V(embedding_reg)  # Attention
                    A_U = self.network.milAggregation.attentionModule.attention_U(embedding_reg)  # Gate
                    w_logits = self.network.milAggregation.attentionModule.attention_weights(A_V * A_U)  # Probabilities - softmax over instances
                else:
                    embedding, _ = self.network.milAggregation(torch.squeeze(torch.stack(region_embeddings_list)))
                    A_V = self.network.milAggregation.attentionModule.attention_V(torch.squeeze(torch.stack(region_embeddings_list)))  # Attention
                    A_U = self.network.milAggregation.attentionModule.attention_U(torch.squeeze(torch.stack(region_embeddings_list)))  # Gate
                    w_logits = self.network.milAggregation.attentionModule.attention_weights(A_V * A_U)  # Probabilities - softmax over instances
                    patch_classification = torch.cat(instance_classification, dim=0)

                # embedding, w = self.network.milAggregation.attentionModule(torch.squeeze(features))
                global_classification = self.network.classifier(embedding).detach().cpu().numpy()

                patch_classification = patch_classification.detach().cpu().numpy()

                region_classification_aux = w_logits.detach().cpu().numpy()
                region_classification = np.zeros(patch_classification.shape)

                region_info = region_info.detach().cpu().numpy()
                for i in range(len(region_info)):
                    region_classification[i] = region_classification_aux[int(region_info[i])]

                print('MIL classification: done')
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/patch_classification.npy', patch_classification)
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/region_classification.npy', region_classification)
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/global_classification.npy', global_classification)
            else:
                patch_classification = np.load(self.dir_results + self.dir_inference + self.current_wsi_name + '/patch_classification.npy')
                region_classification = np.load(self.dir_results + self.dir_inference + self.current_wsi_name + '/region_classification.npy')
                global_classification = np.load(self.dir_results + self.dir_inference + self.current_wsi_name + '/global_classification.npy')

            # Normalize attention
            min_max_att = np.load(self.dir_results + self.dir_inference + '/min_max_att.npy')
            mean_std_att = np.load(self.dir_results + self.dir_inference + '/mean_std_att.npy')
            patch_classification = (patch_classification - mean_std_att[0]) / mean_std_att[1]
            region_classification = (region_classification - mean_std_att[0]) / mean_std_att[1]

            # Apply sigmoid
            patch_classification = np.clip(patch_classification, 0.001, 0.999)
            region_classification = np.clip(region_classification, 0.001, 0.999)

            # Load thumbnail array for predictions
            colormap_height = int(wsi_thumbnail.height // (self.ori_patch_size / 16)) # SFN
            colormap_width = int(wsi_thumbnail.width // (self.ori_patch_size / 16))

            if not os.path.exists(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap.npy') or not os.path.exists(self.dir_results + self.dir_inference + self.current_wsi_name + '/roi_patch_extraction.jpeg'):
                prediction_colormap = -1 * np.ones((colormap_height, colormap_width), dtype=np.float32)
                prediction_colormap_reg = -1 * np.ones((colormap_height, colormap_width), dtype=np.float32)
                roi_mask = np.ones((colormap_height, colormap_width), dtype=np.float32)
                roi_mask_reg = np.ones((colormap_height, colormap_width), dtype=np.float32)

                x_coordinates = []
                y_coordinates = []
                current_number_of_regions = len([j for j in os.listdir(self.current_wsi_subfolder) if '.' not in j])

                for reg_id in range(current_number_of_regions):

                    # Current region dataframe
                    current_reg_info_dataframe = self.current_tile_info_dataframe.loc[self.current_tile_info_dataframe['Reg_ID'] == reg_id]

                    for current_tile_id in range(len(current_reg_info_dataframe)):

                        current_dataframe_row = current_reg_info_dataframe.iloc[current_tile_id]

                        x_coordinates.append(int(int(current_dataframe_row['400X_coor']) - offset_x_x25 * 16))
                        y_coordinates.append(int(int(current_dataframe_row['400Y_coor']) - offset_y_x25 * 16))

                # Generate WSI map prediction with attention
                for iInstance, (patch_attention_score, reg_attention_score, x_coordinate, y_coordinate) in enumerate(zip(patch_classification, region_classification, x_coordinates, y_coordinates)):
                    print(str(iInstance + 1) + '/' + str(patch_classification.shape[0]), end='\r')

                    x_index = int(x_coordinate // self.ori_patch_size)
                    y_index = int(y_coordinate // self.ori_patch_size)

                    # To be overlayed using alpha channel
                    prediction_colormap[y_index, x_index] = patch_attention_score[0]
                    prediction_colormap_reg[y_index, x_index] = reg_attention_score[0]

                roi_mask[prediction_colormap == -1] = 0
                roi_mask = resize(roi_mask, (wsi_thumbnail.height, wsi_thumbnail.width))
                roi_image = Image.fromarray(np.uint8((roi_mask)*255))
                roi_image.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/roi_patch_extraction.jpeg')

                roi_mask_reg[prediction_colormap_reg == -1] = 0
                roi_mask_reg = resize(roi_mask_reg, (wsi_thumbnail.height, wsi_thumbnail.width))
                roi_image_reg = Image.fromarray(np.uint8((roi_mask_reg)*255))
                roi_image_reg.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/roi_reg_extraction.jpeg')

                prediction_colormap[prediction_colormap == -1] = 0.
                prediction_colormap_reg[prediction_colormap_reg == -1] = 0.

                # Save generated colormap
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap.npy', prediction_colormap)
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap_region.npy', prediction_colormap_reg)
                print('Prediction colormap: done')
            else:
                prediction_colormap = np.load(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap.npy')
                prediction_colormap_reg = np.load(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap_region.npy')

            # Save thumbnail image and the overlayed colormap
            wsi_alpha = np.ndarray(buffer=wsi_thumbnail.write_to_memory(),
                            dtype=np.uint8,
                            shape=[wsi_thumbnail.height, wsi_thumbnail.width, wsi_thumbnail.bands])

            prediction_colormap = resize(prediction_colormap, (wsi_thumbnail.height, wsi_thumbnail.width))
            prediction_colormap = prediction_colormap**2
            prediction_colormap = rank.mean(prediction_colormap, disk(21))
            prediction_colormap_reg = resize(prediction_colormap_reg, (wsi_thumbnail.height, wsi_thumbnail.width))
            prediction_colormap_reg = prediction_colormap_reg**2
            prediction_colormap_reg = rank.mean(prediction_colormap_reg, disk(21))

            alpha = 0.5; wsi_blended = ((plt.cm.jet(prediction_colormap)[:,:,:3] * 255) * alpha + wsi_alpha * (1 - alpha)).astype(np.uint8) # [:,:,:3]
            wsi_blended_reg = ((plt.cm.jet(prediction_colormap_reg)[:,:,:3] * 255) * alpha + wsi_alpha * (1 - alpha)).astype(np.uint8) # [:,:,:3]
            
            background_mask = np.ones((wsi_thumbnail.height, wsi_thumbnail.width))
            full_slide = wsi_thumbnail.extract_area(0,0,wsi_thumbnail.width,wsi_thumbnail.height)
            slide_numpy = full_slide.write_to_memory()
            slide_numpy = np.fromstring(slide_numpy, dtype=np.uint8).reshape(full_slide.height, full_slide.width, 3)

            background_mask[slide_numpy[:,:,1] > 240] = 0
            background_mask = ndimage.binary_closing(background_mask, structure=np.ones((25,25))).astype(background_mask.dtype)
            background_mask = ndimage.binary_opening(background_mask, structure=np.ones((25,25))).astype(background_mask.dtype)
            background_mask = ndimage.binary_opening(background_mask, structure=np.ones((45,45))).astype(background_mask.dtype)
                
            background_mask = skimage.morphology.remove_small_objects(background_mask.astype(bool), min_size=5000)
            wsi_blended[background_mask == 0] = 255
            wsi_blended_reg[background_mask == 0] = 255

            cm = plt.get_cmap('jet')
            # img = Image.fromarray((cm(wsi_blended)[:, :, :3] * 255).astype(np.uint8))
            img = Image.fromarray(wsi_blended)
            img.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction.jpeg')
            img_reg = Image.fromarray(wsi_blended_reg)
            img_reg.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_region.jpeg')
            print('Thumbnail: done')

    def image_normalization(self, x):
        x = x / 255.0
        # channel first
        x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x


########################################################################################################################
dir_out = 'grading/results/trial/'
dir_inference = 'inference/'
network = torch.load(dir_out + '0_network_weights_best.pth')

dir_images = 'Tiles_dataset/test/'

inference = NMILInference(dir_out, network, dir_inference)

test_wsi = os.listdir(dir_images)

for current_wsi_subfolder in reversed(test_wsi):
    inference.infer(current_wsi_subfolder=dir_images + current_wsi_subfolder)