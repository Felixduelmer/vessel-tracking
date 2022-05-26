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


def export(arguments):
    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(
            *model.get_fp_bp_time()))
        exit()

    images = torch.zeros((1, 2, 320, 320)).cuda().float()
    model.init_hidden(images.size(0), images.size(3))
    traced_script_module = torch.jit.trace(model.net, (images, model.states), strict=False)
    traced_script_module.save("traced_butterfly_model.pt")




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument(
        '-c', '--config', help='training config file', required=True)
    parser.add_argument(
        '-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    export(args)
