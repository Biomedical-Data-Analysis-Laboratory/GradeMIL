import numpy as np
import pickle
import random
import glob
import sys
import csv
import time
import os

import wandb
import sklearn
import torch
import datetime
import cv2
import sklearn.metrics
import os
from timeit import default_timer as timer
import json
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
from scipy.special import softmax
from PIL import Image
from losses import FocalTverskyLoss
import pandas as pd
import skimage

class Dataset(object):

    def __init__(self, dir_images, list_of_wsi, classes,
                 data_augmentation, channel_first, modality, tiles_per_wsi, clinical_data_csv):

        'Internal states initialization'
        self.dir_images = dir_images
        self.list_of_wsi = list_of_wsi
        self.classes = classes
        self.channel_first = channel_first
        self.modality = modality
        self.tiles_per_wsi = tiles_per_wsi
        self.bag_id = []
        self.data_augmentation = data_augmentation
        self.images_400x = []
        self.images_100x = []
        self.images_25x = []
        self.clinical_dataframe = pd.read_csv(clinical_data_csv)
        self.region_filename_csv = 'tile_info.csv'
        self.D = dict()
        self.colorjitter = torchvision.transforms.ColorJitter(brightness=.5, hue=.3)
        self.autocontrast = torchvision.transforms.RandomAutocontrast(p=0.2)

        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.region_id = []
        i = 0

        for wsi_count, ID in enumerate(self.list_of_wsi):

            print('{}/{}: {}'.format(wsi_count + 1, len(self.list_of_wsi), ID))
            current_wsi_repository = os.path.join(dir_images + ID)

            current_tile_info_dataframe = pd.read_csv(os.path.join(current_wsi_repository, self.region_filename_csv))

            if self.modality == 'nested':

                current_number_of_regions = len([j for j in os.listdir(current_wsi_repository) if '.' not in j])

                number_of_tiles_per_region = int(np.ceil(self.tiles_per_wsi / current_number_of_regions))

                for reg_id in range(current_number_of_regions):

                    # Current region dataframe
                    current_reg_info_dataframe = current_tile_info_dataframe.loc[
                        current_tile_info_dataframe['Reg_ID'] == reg_id]

                    if len(current_reg_info_dataframe) > number_of_tiles_per_region:
                        current_reg_info_dataframe = current_reg_info_dataframe.sample(number_of_tiles_per_region)

                    for current_tile_id in range(len(current_reg_info_dataframe)):
                        current_dataframe_row = current_reg_info_dataframe.iloc[current_tile_id]

                        name_400x = current_dataframe_row['Path_400x']
                        name_100x = current_dataframe_row['Path_100x']
                        name_25x = current_dataframe_row['Path_25x']
                        current_region_id = current_dataframe_row['Reg_ID'].item()

                        if ID not in self.D:
                            self.D[ID] = [i]
                            self.bag_id.append(ID)
                        else:
                            self.D[ID].append(i)

                        self.images_400x.append(name_400x)
                        self.images_100x.append(name_100x)
                        self.images_25x.append(name_25x)

                        self.region_id.append(int(current_region_id))
                        i += 1

            else:

                # Current region dataframe
                if len(current_tile_info_dataframe) > self.tiles_per_wsi:
                    current_tile_info_dataframe = current_tile_info_dataframe.sample(self.tiles_per_wsi)

                for current_tile_id in range(len(current_tile_info_dataframe)):

                    current_dataframe_row = current_tile_info_dataframe.iloc[current_tile_id]

                    name_400x = current_dataframe_row['Path_400x']
                    name_100x = current_dataframe_row['Path_100x']
                    name_25x = current_dataframe_row['Path_25x']
                    current_region_id = current_dataframe_row['Reg_ID'].item()

                    if ID not in self.D:
                        self.D[ID] = [i]
                        self.bag_id.append(ID)
                    else:
                        self.D[ID].append(i)

                    self.images_400x.append(name_400x)
                    self.images_100x.append(name_100x)
                    self.images_25x.append(name_25x)

                    self.region_id.append(int(current_region_id))
                    i += 1

        # Generate indexes according to the number of images / embeddings
        self.indexes = np.arange(len(self.images_400x))

        # Preemptively load input data into memory
        self.y_instances = np.empty(len(self.indexes), dtype=object)

        for i in np.arange(len(self.indexes)):
            # Please revise these lines to make sure Study ID is properly extracted and the label is adequate
            current_SID = self.images_400x[self.indexes[i]].split('/')[-4]
            self.y_instances[self.indexes[i]] = self.clinical_dataframe.loc[self.clinical_dataframe['SID'] ==
                                                                                current_SID, 'WHO04'].item()
        self.y_instances[self.y_instances == 'High grade'] = 1
        self.y_instances[self.y_instances == 'Low grade'] = 0
        self.y_instances = self.y_instances.astype(np.float32)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x400 = Image.open(self.images_400x[self.indexes[index]])
        x400 = np.asarray(x400)
        x100 = Image.open(self.images_100x[self.indexes[index]])
        x100 = np.asarray(x100)
        x25 = Image.open(self.images_25x[self.indexes[index]])
        x25 = np.asarray(x25)

        if self.data_augmentation:
            random_factors_list = [random.random() for _ in range(6)]
            x400_augm = np.squeeze(self.image_transformation(x400, random_factors_list))
            x100_augm = np.squeeze(self.image_transformation(x100, random_factors_list))
            x25_augm = np.squeeze(self.image_transformation(x25, random_factors_list))

            x400_augm = self.image_normalization(x400_augm)
            x100_augm = self.image_normalization(x100_augm)
            x25_augm = self.image_normalization(x25_augm)
        else:
            x400_augm = None
            x100_augm = None
            x25_augm = None

        x400 = self.image_normalization(x400)
        x100 = self.image_normalization(x100)
        x25 = self.image_normalization(x25)

        return x400, x400_augm, x100, x100_augm, x25, x25_augm, self.region_id[self.indexes[index]]

    def image_transformation(self, img, random_factors_list):

        if random_factors_list[0] > 0.5:
            img = np.fliplr(img)
        if random_factors_list[1] > 0.5:
            img = np.flipud(img)
        if random_factors_list[2] > 0.5:
            img = np.asarray(self.colorjitter(Image.fromarray(img.astype(np.uint8))))
        if random_factors_list[3] > 0.5:
            img = np.asarray(self.autocontrast(Image.fromarray(img.astype(np.uint8))))
        if random_factors_list[4] > 0.5:
            angle = random_factors_list[0] * 60 - 30
            img = skimage.transform.rotate(img, angle)
        if random_factors_list[5] > 0.5:
            img = skimage.util.random_noise(img, var=random_factors_list[0] ** 2)

        return img

    def image_normalization(self, x):
        # image resize
        # x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))
        # intensity normalization
        x = x / 255.0
        # channel first
        if self.channel_first:
            x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x


class DataGenerator(object):

    def __init__(self, dataset, batch_size, shuffle, max_instances, modality):

        'Internal states initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.bag_id))
        self.max_instances = max_instances
        self.modality = modality

        self._idx = 0
        self._epoch = 1
        self._reset()

    def __len__(self):

        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):

        return self

    def __next__(self):

        # If dataset is completed, stop iterator
        if self._idx >= len(self.indexes):
            self._epoch += 1
            self._reset()
            raise StopIteration()

        # Select instances from bag
        ID = self.dataset.bag_id[self.indexes[self._idx]]
        images_id = self.dataset.D[ID]

        # Get bag-level label
        Y = np.zeros((2,))
        Y[int(self.dataset.y_instances[images_id[0]])] = 1

        # Memory limitation of patches in one slide
        if self.modality == 'nested':
            sampled_indexes = []
            region_labels = list(map(lambda i: self.dataset.region_id[i], images_id))
            if region_labels.count(region_labels[0]) == len(region_labels):
                reg_indexes = [i for i in range(len(region_labels))]
                current_region_samples = random.sample(reg_indexes, min(int(self.max_instances),
                                                                        region_labels.count(region_labels[0])))
                current_region_samples = [images_id[i] for i in current_region_samples]
                sampled_indexes.extend(current_region_samples)
            else:
                for reg_id in range(max(region_labels)):
                    reg_indexes = [i for i, j in enumerate(region_labels) if j == reg_id]
                    current_region_samples = random.sample(reg_indexes,
                                                           min(int(np.ceil(self.max_instances / max(region_labels))),
                                                               region_labels.count(reg_id)))
                    current_region_samples = [images_id[i] for i in current_region_samples]
                    sampled_indexes.extend(current_region_samples)
        else:
            if len(images_id) >= self.max_instances:
                sampled_indexes = random.sample(images_id, self.max_instances)
            else:
                sampled_indexes = images_id
        self.instances_indexes = sampled_indexes

        # Load images and include into the batch
        X400 = []
        X400_augm = []
        X100 = []
        X100_augm = []
        X25 = []
        X25_augm = []
        region_info = []
        for i in sampled_indexes:
            x400, x400_augm, x100, x100_augm, x25, x25_augm, region_info_id = self.dataset.__getitem__(i)
            X400.append(x400)
            X400_augm.append(x400_augm)
            X100.append(x100)
            X100_augm.append(x100_augm)
            X25.append(x25)
            X25_augm.append(x25_augm)
            region_info.append(region_info_id)

        # Update bag index iterator
        self._idx += self.batch_size

        if self.dataset.data_augmentation:
            return np.array(X400).astype('float32'), np.array(X100).astype('float32'), np.array(X25).astype(
                'float32'), np.array(Y).astype('float32'), \
                   np.array(X400_augm).astype('float32'), np.array(X100_augm).astype('float32'), np.array(
                X25_augm).astype('float32'), np.array(region_info).astype('int64')
        else:
            return np.array(X400).astype('float32'), np.array(X100).astype('float32'), np.array(X25).astype(
                'float32'), np.array(Y).astype('float32'), \
                   None, None, None, np.array(region_info).astype('int64')

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0



class MILTrainer():

    def __init__(self, dir_out, network, lr, id, early_stopping,
                 scheduler, virtual_batch_size, criterion,
                 backbone_freeze, class_weights, loss_function,
                 tfl_alpha, tfl_gamma, opt_name):

        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.best_auc = 0.
        self.init_time = 0
        self.lr = lr
        self.L_epoch = 0
        self.f1_lc_val = []
        self.f1_lc_train = []
        self.k2_lc_val = []
        self.k2_lc_train = []
        self.macro_auc_lc_train = []
        self.macro_auc_lc_val = []
        self.L_lc = []
        self.Lce_lc_val = []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.network = network
        self.test_generator = []
        self.train_generator = []
        self.preds_train = []
        self.refs_train = []
        self.best_criterion = 0
        self.best_epoch = 0
        self.metrics = {}
        self.id = id
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.criterion = criterion
        self.backbone_freeze = backbone_freeze
        self.class_weights = class_weights
        self.loss_function = loss_function
        self.tfl_alpha = tfl_alpha
        self.tfl_gamma = tfl_gamma
        self.opt_name = opt_name
        self.threshold = 0.5

        # Set optimizers
        self.params = list(self.network.parameters())
        # Remove encoder params from optimization, requires_grad would also work
        if self.backbone_freeze:
            encoder_params = list(self.network.bb.parameters())
            for encoder_layer_param in encoder_params:
                self.params.remove(encoder_layer_param)

        if self.opt_name == 'sgd':
            self.opt = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=1 * 1e-6)
        else:
            self.opt = torch.optim.Adam(self.params, lr=self.lr)

        # Set losses
        if self.loss_function == 'tversky':
            # print('Tversky Focal Loss')
            self.L = FocalTverskyLoss(alpha=self.tfl_alpha, beta=1-self.tfl_alpha, gamma=self.tfl_gamma).cuda()
        elif network.mode == 'embedding' or network.mode == 'mixed':
            self.L = torch.nn.BCEWithLogitsLoss(weight=self.class_weights).cuda()
        elif network.mode == 'instance':
            self.L = torch.nn.BCELoss(weight=self.class_weights).cuda()

        if self.criterion == 'loss':
            self.best_criterion = 1000000

    def train(self, train_generator, val_generator, test_generator, epochs):
        self.epochs = epochs
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator

        # Move network to gpu
        self.network.cuda()

        self.init_time = timer()
        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            # init epoch losses
            self.L_epoch = 0
            self.preds_train = []
            self.refs_train = []

            # Loop over training dataset
            print('[Training]: at bag level...')
            for self.i_iteration, (X400, X100, X25, Y, X400_augm, X100_augm, X25_augm, region_info) in enumerate(self.train_generator):

                X400 = torch.tensor(X400).cuda().float()
                X100 = torch.tensor(X100).cuda().float()
                X25 = torch.tensor(X25).cuda().float()
                if X400_augm is None:
                    X400_augm = X400
                    X100_augm = X100
                    X25_augm = X25
                else:
                    X400_augm = torch.tensor(X400_augm).cuda().float()
                    X100_augm = torch.tensor(X100_augm).cuda().float()
                    X25_augm = torch.tensor(X25_augm).cuda().float()
                Y = torch.tensor(Y).cuda().float()
                region_info = torch.tensor(region_info).cuda().float()

                # Set model to training mode and clear gradients
                self.network.train()

                # Forward network
                if self.train_generator.modality == 'nested':
                    Yhat, yhat_reg, yhat_ins = self.network(X400_augm, X100_augm, X25_augm, region_info)
                else:
                    Yhat, yhat_ins = self.network(X400_augm, X100_augm, X25_augm)

                if self.network.mode == 'instance':
                    Yhat = torch.clip(Yhat, min=0.01, max=0.98)

                # Estimate losses
                Lce = self.L(Yhat, torch.squeeze(Y))

                # Backward gradients
                L = Lce / self.virtual_batch_size
                L.backward()

                # Update weights and clear gradients
                if ((self.i_epoch + 1) % self.virtual_batch_size) == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                ######################################
                ## --- Iteration/Epoch end

                # Save predictions
                self.preds_train.append(Yhat.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())

                # Display losses per iteration
                self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, self.iterations,
                                    Lce.cpu().detach().numpy(), 0, 0, 0, 0,
                                    end_line='\r')

                # Update epoch's losses
                self.L_epoch += Lce.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

            if self.early_stopping:
                if self.i_epoch + 1 == (self.best_epoch + 30):
                    break

    def on_epoch_end(self):

        # Obtain epoch-level metrics
        if not np.isnan(np.sum(self.preds_train)):
            macro_auc_train = roc_auc_score(np.squeeze(np.array(self.refs_train)), np.array(self.preds_train), multi_class='ovr')
        else:
            macro_auc_train = 0.5
        self.macro_auc_lc_train.append(macro_auc_train)

        # Update learning curves
        self.L_lc.append(self.L_epoch)

        # Obtain results on train set
        Yhat_mono = (softmax(np.array(self.preds_train))[:, 1] > self.threshold).astype(np.int32)
        Y_mono = (softmax(np.array(self.refs_train))[:, 1] > self.threshold).astype(np.int32)

        acc_train = accuracy_score(Y_mono, Yhat_mono)
        f1_train = f1_score(Y_mono, Yhat_mono, average='macro', zero_division=0)
        k2_train = cohen_kappa_score(Y_mono, Yhat_mono, weights='quadratic')
        self.f1_lc_train.append(f1_train)
        self.k2_lc_train.append(k2_train)

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch,
                            macro_auc_train, acc_train, f1_train, k2_train, end_line='\n')

        # Obtain results on validation set
        Lce_val, macro_auc_val, acc_val, f1_val, k2_val, pre_val, rec_val = self.test_bag_level_classification(self.val_generator, False)

        # Save loss value into learning curve
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)
        self.f1_lc_val.append(f1_val)
        self.k2_lc_val.append(k2_val)

        metrics = {'epoch': self.i_epoch + 1, 'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
                   'AUCval': np.round(self.macro_auc_lc_val[-1], 4), 'F1val': np.round(self.f1_lc_val[-1], 4),
                   'K2val': np.round(self.k2_lc_val[-1], 4)}
        with open(self.dir_results + str(self.id) + 'metrics.json', 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        # Weights AND Biases
        wandb.log({
            'loss_train': self.L_epoch,
            'loss_val': Lce_val,
            'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
            'AUCval': np.round(self.macro_auc_lc_val[-1], 4),
            'F1val': np.round(self.f1_lc_val[-1], 4),
            'K2val': np.round(self.k2_lc_val[-1], 4)
        })

        if (self.i_epoch + 1) > 5:
            if self.criterion == 'auc':
                if self.best_criterion < self.macro_auc_lc_val[-1]:
                    self.best_criterion = self.macro_auc_lc_val[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + str(self.id) + 'network_weights_best.pth')

            elif self.criterion == 'z':
                if self.best_criterion < (-self.constrain_proportion_epoch):
                    self.best_criterion = -self.constrain_proportion_epoch
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + str(self.id) + 'network_weights_best.pth')

            elif self.criterion == 'k2':
                if self.best_criterion < self.k2_lc_val[-1]:
                    self.best_criterion = self.k2_lc_val[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + str(self.id) + 'network_weights_best.pth')

            elif self.criterion == 'f1':
                if self.best_criterion < self.f1_lc_val[-1]:
                    self.best_criterion = self.f1_lc_val[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + str(self.id) + 'network_weights_best.pth')

            if self.criterion == 'loss':
                if self.best_criterion > self.L_lc[-1]:
                    self.best_criterion = self.L_lc[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + str(self.id) + 'network_weights_best.pth')

        # Each xx epochs, test models and plot learning curves
        if (self.i_epoch + 1) % 5 == 0:
            # Save weights
            torch.save(self.network, self.dir_results + str(self.id) + 'network_weights.pth')

            # Plot learning curve
            self.plot_learning_curves()

        if (self.epochs == (self.i_epoch + 1)) or (self.early_stopping and (self.i_epoch + 1 == (self.best_epoch + 30))):
            print('-' * 20)
            print('-' * 20)

            self.network = torch.load(self.dir_results + str(self.id) + 'network_weights_best.pth')

            # Plot learning curve
            self.plot_learning_curves()

            # Obtain results on validation set
            Lce_val, macro_auc_val, acc_val, f1_val, k2_val, pre_val, rec_val = self.test_bag_level_classification(self.val_generator, True)

            # Obtain results on test set
            Lce_test, macro_auc_test, acc_test, f1_test, k2_test, pre_test, rec_test = self.test_bag_level_classification(self.test_generator, False)

            X400 = self.test_generator.dataset.images_400x
            X100 = self.test_generator.dataset.images_100x
            X25 = self.test_generator.dataset.images_25x
            Y = self.test_generator.dataset.y_instances
            acc, f1, k2 = self.test_instance_level_classification(X400, X100, X25, Y, self.test_generator.dataset.classes)


            metrics = {'epoch': self.best_epoch, 'AUCtest': np.round(macro_auc_test, 4), 'acc_test': np.round(acc_test, 4),
                       'f1_test': np.round(f1_test, 4), 'k2_test': np.round(k2_test, 4),
                       'pre_test': np.round(pre_test, 4), 'rec_test': np.round(rec_test, 4),
                       'AUCval': np.round(macro_auc_val, 4), 'acc_val': np.round(acc_val, 4),
                       'f1_val': np.round(f1_val, 4), 'k2_val': np.round(k2_val, 4),
                       'pre_val': np.round(pre_val, 4), 'rec_val': np.round(rec_val, 4), 'acc_ins': np.round(acc, 4),
                       'f1_ins': np.round(f1, 4), 'k2_ins': np.round(k2, 4), 'threshold': self.threshold,
                       }

            with open(self.dir_results + str(self.id) + 'best_metrics.json', 'w') as fp:
                json.dump(metrics, fp)
            print(metrics)

        self.metrics = metrics
        print('-' * 20)
        print('-' * 20)

    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            for i in range(x.shape[0]):
                axes.plot(x[i, :], y[i, :], 'o-', label=['Train','Val'][i])
                axes.legend(loc="upper right")
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        plot_subplot(axes[0, 0], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1, np.array([self.L_lc, self.Lce_lc_val]), "Lce")
        plot_subplot(axes[1, 0], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1, np.array([self.macro_auc_lc_train, self.macro_auc_lc_val]), "AUC")
        plot_subplot(axes[0, 1], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1, np.array([self.f1_lc_train, self.f1_lc_val]), "F1")
        plot_subplot(axes[1, 1], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1, np.array([self.k2_lc_train, self.k2_lc_val]), "K2")

        plt.savefig(self.dir_results + str(self.id) + 'learning_curve.png')

    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc, acc, f1, k2, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f} ; acc={:.4f} ; f1={:.4f} ; k2={:.4f}".format(
            i_epoch, epochs, iteration, total_iterations, Lce, macro_auc, acc, f1, k2)

        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def test_instance_level_classification(self, X400, X100, X25, Y, classes):

        self.network.eval()
        print(['INFO: Testing at instance level...'])

        Yhat = []
        for iInstance in np.arange(0, Y.shape[0]):

            x400 = Image.open(X400[iInstance])
            x400 = np.asarray(x400)
            # Normalization
            x400 = self.test_generator.dataset.image_normalization(x400)
            x400 = torch.tensor(x400).cuda().float()
            x400 = x400.unsqueeze(0)

            x100 = Image.open(X100[iInstance])
            x100 = np.asarray(x100)
            # Normalization
            x100 = self.test_generator.dataset.image_normalization(x100)
            x100 = torch.tensor(x100).cuda().float()
            x100 = x100.unsqueeze(0)

            x25 = Image.open(X25[iInstance])
            x25 = np.asarray(x25)
            # Normalization
            x25 = self.test_generator.dataset.image_normalization(x25)
            x25 = torch.tensor(x25).cuda().float()
            x25 = x25.unsqueeze(0)

            # Make prediction
            yhat = torch.softmax(self.network.classifier(torch.squeeze(self.network.bb(x400, x100, x25))), 0)
            yhat = yhat.detach().cpu().numpy()
            Yhat.append(yhat)

        Yhat = np.array(Yhat)
        Yhat = (Yhat[:,1] > self.threshold).astype(np.int32)

        cr = classification_report(Y, Yhat, target_names=classes, digits=4, zero_division=0)
        acc = accuracy_score(Y, Yhat)
        f1 = f1_score(Y, Yhat, average='macro', zero_division=0)
        cm = confusion_matrix(Y, Yhat)
        k2 = cohen_kappa_score(Y, Yhat, weights='quadratic')

        f = open(self.dir_results + str(self.id) + 'report.txt', 'w')
        f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))
        f.close()

        print('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))

        return acc, f1, k2

    def test_bag_level_classification(self, test_generator, binary):
        self.network.eval()
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0
        for self.i_iteration, (X400, X100, X25, Y, _, _, _, region_info) in enumerate(test_generator):
            X400 = torch.tensor(X400).cuda().float()
            X100 = torch.tensor(X100).cuda().float()
            X25 = torch.tensor(X25).cuda().float()
            Y = torch.tensor(Y).cuda().float()
            region_info = torch.tensor(region_info).cuda().float()

            # Forward network
            if test_generator.modality == 'nested':
                Yhat, _, _ = self.network(X400, X100, X25, region_info)
            else:
                Yhat, _ = self.network(X400, X100, X25)
            # Estimate losses
            if self.network.mode == 'instance':
                Yhat = torch.clip(Yhat, min=0.01, max=0.98)

            Lce = self.L(Yhat, torch.squeeze(Y))
            Lce_e += Lce.cpu().detach().numpy() / len(test_generator)

            Y_all.append(Y.detach().cpu().numpy())
            Yhat_all.append(torch.softmax(Yhat, 0).detach().cpu().numpy())

            # Display losses per iteration
            self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(test_generator),
                                Lce.cpu().detach().numpy(), 0, 0, 0, 0,
                                end_line='\r')
        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if not np.isnan(np.sum(Yhat_all)):
            macro_auc = roc_auc_score(Y_all, Yhat_all, multi_class='ovr')

            # Metrics
            if binary:
                list_of_thresholds = [round(th*0.01, 2) for th in range(101)]
                f1_best = -0.1
                self.threshold = 0.0
                for threshold in list_of_thresholds:
                    Yhat_mono = (Yhat_all[:, 1] > threshold).astype(np.int32)
                    Y_mono = (Y_all[:, 1] > threshold).astype(np.int32)

                    if not(np.sum(Yhat_mono) == len(Yhat_mono) or np.sum(Yhat_mono) == 0):

                        cr_current = classification_report(Y_mono, Yhat_mono, target_names=test_generator.dataset.classes, digits=4, zero_division=0)
                        acc_current = accuracy_score(Y_mono, Yhat_mono)
                        f1_current = f1_score(Y_mono, Yhat_mono, average='macro', zero_division=0)
                        cm_current = confusion_matrix(Y_mono, Yhat_mono)
                        k2_current = cohen_kappa_score(Y_mono, Yhat_mono, weights='quadratic')
                        pre_current = precision_score(Y_mono, Yhat_mono, average='macro', zero_division=0)
                        rec_current = recall_score(Y_mono, Yhat_mono, average='macro', zero_division=0)

                    else:
                        cr_current = 0
                        acc_current = 0
                        f1_current = 0
                        cm_current = 0
                        k2_current = 0
                        pre_current = 0
                        rec_current = 0

                    if f1_current > f1_best:
                        f1_best = f1_current
                        self.threshold = threshold
                        cr = cr_current
                        acc = acc_current
                        f1 = f1_current
                        cm = cm_current
                        k2 = k2_current
                        pre = pre_current
                        rec = rec_current

            else:
                Yhat_mono = (Yhat_all[:, 1] > self.threshold).astype(np.int32)
                Y_mono = (Y_all[:, 1] > self.threshold).astype(np.int32)

                cr = classification_report(Y_mono, Yhat_mono, target_names=test_generator.dataset.classes, digits=4, zero_division=0)
                acc = accuracy_score(Y_mono, Yhat_mono)
                f1 = f1_score(Y_mono, Yhat_mono, average='macro', zero_division=0)
                cm = confusion_matrix(Y_mono, Yhat_mono)
                k2 = cohen_kappa_score(Y_mono, Yhat_mono, weights='quadratic')
                pre = precision_score(Y_mono, Yhat_mono, average='macro', zero_division=0)
                rec = recall_score(Y_mono, Yhat_mono, average='macro', zero_division=0)

            f = open(self.dir_results + str(self.id) + 'report_bag.txt', 'w')
            f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))
            f.close()

            print('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))

        else:

            macro_auc, acc, f1, k2, pre, rec = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Display losses per epoch
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(test_generator),
                            Lce_e, macro_auc, acc, f1, k2,
                            end_line='\n')


        return Lce_e, macro_auc, acc, f1, k2, pre, rec


class NMILArchitecture(torch.nn.Module):

    def __init__(self, classes, mode, aggregation, backbone, include_background,
                 neurons_1, neurons_2, neurons_3, neurons_att_1, neurons_att_2, dropout_rate):
        super(NMILArchitecture, self).__init__()

        """Data Generator object for NMIL.
            CNN based architecture for NMIL classification.
        Args:
          classes: 
          mode:
          aggregation: max, mean, attentionMIL, mcAttentionMIL
          backbone:
          include_background:
        Returns:
          NMILDataGenerator object
        """

        'Internal states initialization'

        self.classes = classes
        self.nClasses = 1
        self.mode = mode
        self.aggregation = aggregation
        self.backbone = backbone
        self.include_background = include_background
        self.neurons_1 = neurons_1
        self.neurons_2 = neurons_2
        self.neurons_3 = neurons_3
        self.neurons_att_1 = neurons_att_1
        self.neurons_att_2 = neurons_att_2
        self.dropout_rate = dropout_rate

        if self.include_background:
            self.nClasses = len(classes) + 1
        else:
            self.nClasses = len(classes)

        # Backbone
        self.bb = Encoder(pretrained=True, backbone=backbone, aggregation=True)

        # Classifiers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1536, self.neurons_2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.neurons_2, self.neurons_3),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.neurons_3, self.nClasses),
        )
        # MIL aggregation
        self.milAggregation = MILAggregation(aggregation=aggregation, nClasses=self.nClasses, mode=self.mode,
                                             L=self.neurons_att_1, D=self.neurons_att_2)

    def forward(self, x400, x100, x25, region_info):

        # Patch-Level feature extraction
        features = self.bb(x400, x100, x25)

        if self.mode == 'instance':
            # Classification
            patch_classification = torch.softmax(self.classifier(torch.squeeze(features)), 1)

            # MIL aggregation
            global_classification = self.milAggregation(patch_classification)

        elif self.aggregation == 'attentionMIL':

            instance_classification = []
            region_embeddings_list = []
            for region_id in range(int(torch.max(region_info).item()) + 1):
                region_features = features[torch.where(region_info == region_id, True, False)]
                # print(region_features)
                if len(region_features) == 0:
                    continue
                elif len(region_features) == 1:
                    region_embeddings_list.append(torch.squeeze(region_features, dim=0))
                    A_V = self.milAggregation.attentionModule.attention_V(region_features)  # Attention
                    A_U = self.milAggregation.attentionModule.attention_U(region_features)  # Gate
                    w_logits = self.milAggregation.attentionModule.attention_weights(
                        A_V * A_U)  # Probabilities - softmax over instances
                    instance_classification.append(w_logits)
                    continue
                embedding_reg, w_reg = self.milAggregation(torch.squeeze(region_features))
                instance_classification.append(w_reg)
                region_embeddings_list.append(embedding_reg)

            if len(region_embeddings_list) == 1:
                embedding = embedding_reg
                patch_classification = w_reg
            else:
                embedding, w = self.milAggregation(torch.squeeze(torch.stack(region_embeddings_list)))
                patch_classification = torch.cat(instance_classification, dim=0)

            global_classification = self.classifier(embedding)


        elif self.aggregation in ['mean', 'max']:
            embedding = self.milAggregation(torch.squeeze(features))
            global_classification = self.classifier(embedding)

        if self.include_background:
            global_classification = global_classification[1:]

        if len(region_embeddings_list) == 1:
            return global_classification, global_classification, patch_classification
        else:
            return global_classification, w, patch_classification



class Encoder(torch.nn.Module):

    def __init__(self, pretrained, backbone, aggregation):
        super(Encoder, self).__init__()

        self.aggregation = aggregation
        self.pretrained = pretrained
        self.backbone = backbone

        if backbone == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=pretrained)
            self.F400 = torch.nn.Sequential(resnet.conv1,
                                         resnet.bn1,
                                         resnet.relu,
                                         resnet.maxpool,
                                         resnet.layer1,
                                         resnet.layer2,
                                         resnet.layer3,
                                         resnet.layer4)
            self.F100 = torch.nn.Sequential(resnet.conv1,
                                         resnet.bn1,
                                         resnet.relu,
                                         resnet.maxpool,
                                         resnet.layer1,
                                         resnet.layer2,
                                         resnet.layer3,
                                         resnet.layer4)
            self.F25 = torch.nn.Sequential(resnet.conv1,
                                         resnet.bn1,
                                         resnet.relu,
                                         resnet.maxpool,
                                         resnet.layer1,
                                         resnet.layer2,
                                         resnet.layer3,
                                         resnet.layer4)
        elif backbone == 'vgg16':
            vgg16 = torchvision.models.vgg16(pretrained=pretrained)
            # self.F = vgg16.features
            # modules = list(vgg16.children())[:-1]
            self.F400 = vgg16.features
            self.F100 = vgg16.features
            self.F25 = vgg16.features

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x400, x100, x25):
        out400 = self.F400(x400)
        out100 = self.F100(x100)
        out25 = self.F25(x25)

        # register the hook
        h400 = out400.register_hook(self.activations_hook)
        h100 = out100.register_hook(self.activations_hook)
        h25 = out25.register_hook(self.activations_hook)

        if self.aggregation:

            out400 = torch.nn.AdaptiveMaxPool2d((1, 1))(out400)
            out100 = torch.nn.AdaptiveMaxPool2d((1, 1))(out100)
            out25 = torch.nn.AdaptiveMaxPool2d((1, 1))(out25)

            out = torch.squeeze(torch.cat((out400, out100, out25), 1))

        return out

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad



class MILAggregation(torch.nn.Module):
    def __init__(self, aggregation, nClasses, mode, L, D):
        super(MILAggregation, self).__init__()

        """Aggregation module for MIL.
        Args:
          aggregation:
        Returns:
          MILAggregation module for CNN MIL Architecture
        """

        self.mode = mode
        self.aggregation = aggregation
        self.nClasses = nClasses
        self.L = L
        self.D = D

        if self.aggregation == 'attentionMIL':
            self.attentionModule = attentionMIL(self.L, self.D, 1)

    def forward(self, feats):

        if self.aggregation == 'max':
            embedding = torch.max(feats, dim=0)[0]
            return embedding
        elif self.aggregation == 'mean':
            embedding = torch.mean(feats, dim=0)
            return embedding
        elif self.aggregation == 'attentionMIL':
            # Attention embedding from Ilse et al. (2018) for MIL. It only works at the binary scenario at instance-level
            embedding, w_logits = self.attentionModule(feats)
            return embedding, torch.softmax(w_logits, dim=0)


class attentionMIL(torch.nn.Module):
    def __init__(self, L, D, K):
        super(attentionMIL, self).__init__()

        # Attention embedding from Ilse et al. (2018) for MIL. It only works at the binary scenario.

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid()
        )
        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, feats):
        # Attention weights computation
        A_V = self.attention_V(feats)  # Attention
        A_U = self.attention_U(feats)  # Gate
        w_logits = self.attention_weights(A_V * A_U)  # Probabilities - softmax over instances

        # Weighted average computation per class
        feats = torch.transpose(feats, 1, 0)
        embedding = torch.squeeze(torch.mm(feats, torch.softmax(w_logits, dim=0)))  # KxL

        return embedding, w_logits