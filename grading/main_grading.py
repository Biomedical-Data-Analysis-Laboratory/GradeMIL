import os
import wandb
from utils_grading import *
import torch
import json
import numpy as np
import argparse


def main(args):

    # Get WSI/patients for each of the sets
    train_subfloder_list = os.path.join(args.dir_images, 'training/')
    val_subfolder_list = os.path.join(args.dir_images, 'validation/')
    test_subfolder_list = os.path.join(args.dir_images, 'test/')

    train_wsi = os.listdir(os.path.join(args.dir_images, 'training/'))
    val_wsi = os.listdir(os.path.join(args.dir_images, 'validation/'))
    test_wsi = os.listdir(os.path.join(args.dir_images, 'test/'))

    # Set data generators
    dataset_train = Dataset(train_subfloder_list, train_wsi, args.classes,
                                        data_augmentation=args.data_augmentation,
                                        channel_first=args.channel_first,
                                        modality=args.modality,
                                        tiles_per_wsi=args.tiles_per_wsi,
                                        clinical_data_csv=args.clinical_data_csv)
    data_generator_train = DataGenerator(dataset_train, batch_size=1, shuffle=True,
                                        max_instances=args.max_instances,
                                        modality=args.modality)

    dataset_val = Dataset(val_subfolder_list, val_wsi, args.classes,
                                        data_augmentation=args.data_augmentation,
                                        channel_first=args.channel_first,
                                        modality=args.modality,
                                        tiles_per_wsi=args.tiles_per_wsi,
                                        clinical_data_csv=args.clinical_data_csv)
    data_generator_val = DataGenerator(dataset_val, batch_size=1, shuffle=False,
                                        max_instances=args.max_instances,
                                        modality=args.modality)

    dataset_test = Dataset(test_subfolder_list, test_wsi, args.classes,
                                        data_augmentation=args.data_augmentation,
                                        channel_first=args.channel_first,
                                        modality=args.modality,
                                        tiles_per_wsi=args.tiles_per_wsi,
                                        clinical_data_csv=args.clinical_data_csv)
    data_generator_test = DataGenerator(dataset_test, batch_size=1, shuffle=False,
                                        max_instances=args.max_instances,
                                        modality=args.modality)

    metrics = []

    for id in range(args.iterations):

        # wandb is unnecessary, it was used to personally keep track of experiments
        # It does not influence model training
        wandb.init(project="project", entity="entity")
        wandb.config.update(args)

        # Set network architecture
        if args.modality == 'nested':
            network = NMILArchitecture(args.classes, mode=args.mode, aggregation=args.aggregation,
                                            backbone=args.backbone, include_background=args.include_background,
                                            neurons_1=args.neurons_1,
                                            neurons_2=args.neurons_2, neurons_3=args.neurons_3,
                                            neurons_att_1=args.neurons_att_1, neurons_att_2=args.neurons_att_2,
                                            dropout_rate=args.dropout_rate)
        else:
            network = MILArchitecture(args.classes, mode=args.mode, aggregation=args.aggregation,
                                            backbone=args.backbone, include_background=args.include_background,
                                            neurons_1=args.neurons_1,
                                            neurons_2=args.neurons_2, neurons_3=args.neurons_3,
                                            neurons_att_1=args.neurons_att_1, neurons_att_2=args.neurons_att_2,
                                            dropout_rate=args.dropout_rate)

        # Perform training
        if args.class_weights_enable:
            class_weights = torch.mul(torch.softmax(torch.tensor(
                            np.array([1, len(dataset_train.y_instances) / sum(dataset_train.y_instances)])), dim=0), 2)
        else:
            class_weights = torch.ones([2])

        trainer = MILTrainer(args.dir_results + args.experiment_name + '/', network,
                                        lr=wandb.config.lr, id=id, # Change fold per id
                                        early_stopping=args.early_stopping, scheduler=args.scheduler,
                                        virtual_batch_size=args.virtual_batch_size,
                                        criterion=args.criterion,
                                        backbone_freeze=args.backbone_freeze, class_weights=class_weights,
                                        loss_function=args.loss_function, tfl_alpha=args.tfl_alpha,
                                        tfl_gamma=args.tfl_gamma, opt_name=args.opt_name)
        trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                                test_generator=data_generator_test, epochs=args.epochs)

        metrics.append([list(trainer.metrics.values())[1:]])

    # Get overall metrics
    metrics = np.squeeze(np.array(metrics))

    if args.cross_validation or args.iterations > 1:
        mu = np.mean(metrics, axis=0)
        std = np.std(metrics, axis=0)

        info = "AUCtest={:.4f}({:.4f}) ; acc={:.4f}({:.4f}) ; f1-score={:.4f}({:.4f}) ; k2={:.4f}({:.4f}) ; pre={:.4f}({:.4f}) ; rec={:.4f}({:.4f})".format(
              mu[0], std[0], mu[1], std[1], mu[2], std[2], mu[3], std[3], mu[4], std[4], mu[5], std[5])
    else:
        info = "AUCtest={:.4f}; acc={:.4f}; f1-score={:.4f}; k2={:.4f}; pre={:.4f}; rec={:.4f}".format(
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5])

    f = open(args.dir_results + args.experiment_name + '/' + 'method_metrics.txt', 'w')
    f.write(info)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--dir_images", default='Tiles_dataset/', type=str)
    parser.add_argument("--dir_results", default='results/', type=str)
    parser.add_argument("--clinical_data_csv", default='clinicopathological_data.csv', type=str)
    parser.add_argument("--experiment_name", default="trial", type=str)

    # Dataset
    parser.add_argument("--classes", default=['Low', 'High'], type=list)
    parser.add_argument("--max_instances", default=128, type=int)
    parser.add_argument("--tiles_per_wsi", default=5000, type=int)
    parser.add_argument("--include_background", default=False, type=bool)
    parser.add_argument("--cross_validation", default=False, type=bool)

    # Architecture
    parser.add_argument("--backbone", default='vgg16', type=str)
    parser.add_argument("--backbone_freeze", default=False, type=bool)
    parser.add_argument("--aggregation", default="attentionMIL", type=str)
    parser.add_argument("--modality", default="mil", type=str)
    parser.add_argument("--mode", default="embedding", type=str)
    parser.add_argument("--set_structure", default=False, type=bool)
    parser.add_argument("--class_weights_enable", default=False, type=bool)
    parser.add_argument("--channel_first", default=True, type=bool)

    # Hyperparameters
    parser.add_argument("--criterion", default='auc', type=str)
    parser.add_argument("--lr", default=1 * 1e-3, type=float)
    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--data_augmentation", default=True, type=bool)
    parser.add_argument("--iterations", default=5, type=int)
    parser.add_argument("--opt_name", default="sgd", type=str)
    parser.add_argument("--loss_function", default="ce", type=str)
    parser.add_argument("--early_stopping", default=True, type=bool)
    parser.add_argument("--scheduler", default=True, type=bool)
    parser.add_argument("--neurons_1", default=1536, type=int)
    parser.add_argument("--neurons_2", default=4096, type=int)
    parser.add_argument("--neurons_3", default=4096, type=int)
    parser.add_argument("--neurons_att_1", default=1536, type=int)
    parser.add_argument("--neurons_att_2", default=4096, type=int)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--tfl_alpha", default=1, type=float)
    parser.add_argument("--tfl_gamma", default=3, type=float)
    parser.add_argument("--virtual_batch_size", default=1, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.dir_results + args.experiment_name):
        os.mkdir(args.dir_results + args.experiment_name)

    with open(args.dir_results + args.experiment_name + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)