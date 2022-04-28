import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from polyaxon_client.tracking import Experiment, get_data_paths
import os

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model


def train(arguments):
    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # polyaxon specific
    # if this throws an error in a normal env just set POLYAXON_NO_OP to true as an env variable
    polyaxon_input_path = None
    polyaxon_output_path = None
    try:
        data_paths = get_data_paths()
        dataset = "/vessel_felix/ultrasound.h5"
        training_data_path = data_paths['data1'] + dataset
        polyaxon_input_path = training_data_path
        output_path = os.environ['POLYAXON_RUN_OUTPUTS_PATH']
        polyaxon_output_path = output_path + '/felixduelmer'
    except:
        print("This experiment/job is not managed by polyaxon")
    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path = get_dataset_path(
        arch_type, json_opts.data_path) if polyaxon_input_path is None else polyaxon_input_path
    ds_transform = get_dataset_transformation(
        arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model, polyaxon_output_path)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(
            *model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train',
                             preload_data=train_opts.preloadData, transform=ds_transform['train'])
    valid_dataset = ds_class(ds_path, split='valid',
                             preload_data=train_opts.preloadData, transform=ds_transform['valid'])
    test_dataset = ds_class(ds_path, split='test',
                            preload_data=train_opts.preloadData, transform=ds_transform['valid'])
    # TODO:  set number of workers up again
    train_loader = DataLoader(
        dataset=train_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=8,
                              batch_size=train_opts.batchSize, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, num_workers=8,
                             batch_size=train_opts.batchSize, shuffle=True)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            for i in range(train_opts.seq_len):
                model.set_input(images[:, i, :, :, :], labels[:, i, :, :, :])

                model.optimize_parameters()
                # model.optimize_parameters_accumulate_grd(epoch_iter)

                errors = model.get_current_errors()
                error_logger.update(errors, split='train')
            # for img, lbl in zip(images.reshape(np.prod(images.shape[:2]), *images.shape[2:]),
            #                                    labels.reshape(np.prod(labels.shape[:2]), *labels.shape[2:])):
            #     model.set_input(torch.unsqueeze(img, axis=0), torch.unsqueeze(lbl, axis=0))
            #     model.optimize_parameters()
            #     # model.optimize_parameters_accumulate_grd(epoch_iter)
            #
            #     # Error visualisation
            #     errors = model.get_current_errors()
            #     error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):
                # Make a forward pass with the model
                for i in range(train_opts.seq_len):
                    model.set_input(images[:, i, :, :, :],
                                    labels[:, i, :, :, :])
                    model.validate()

                    # Error visualisation
                    errors = model.get_current_errors()
                    stats = model.get_segmentation_stats()
                    error_logger.update({**errors, **stats}, split=split)

                    # Visualise predictions
                    visuals = model.get_current_visuals(labels[:, i, :, :, :])
                    visualizer.display_current_results(
                        visuals, epoch=epoch, save_result=False)

                # for img, lbl in zip(images.reshape(np.prod(images.shape[:2]), *images.shape[2:]), labels.reshape(np.prod(labels.shape[:2]), *labels.shape[2:])):
                #     model.set_input(torch.unsqueeze(img, axis=0), torch.unsqueeze(lbl, axis=0))
                #     model.validate()
                #
                #     # Error visualisation
                #     errors = model.get_current_errors()
                #     stats = model.get_segmentation_stats()
                #     error_logger.update({**errors, **stats}, split=split)
                #
                #     # Visualise predictions
                #     visuals = model.get_current_visuals(torch.unsqueeze(lbl, axis=0))
                #     visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(
                epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(
                epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument(
        '-c', '--config', help='training config file', required=True)
    parser.add_argument(
        '-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
