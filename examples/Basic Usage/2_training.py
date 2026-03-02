import h5py
import torch
torch.backends.cudnn.enabled = False

import numpy as np
from pathlib import Path
from torchvision import transforms

from dtempest.gw import CBCEstimator
import dtempest.core.flow_utils as trans
from dtempest.gw.generation.utils import get_metadata
from dtempest.core.train_utils import H5Dataset, h5_collate_fn


if __name__ == '__main__':
    """
    Basic training routine. 
    Either for a single stage, or for a multistage process.
    """
    
    name = 'GP15_example' # Model name
    device = 'cuda'
    
    
    files_dir = Path('') # Main directory
    project_dir = Path("") # Storage-heavy directory
    rawdat_dir = project_dir / 'Datasets' # Data directory
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'training_test_0' # This one is created automatically, the rest are not
    
    new_model = True  # Train from scratch (see later)
    old_model = train_dir / 'training_test_0'/'GP15_previous_stage.pt'
    
    # Dataset hyperparameters
    imgs_per_seed = 3e6
    valid_fraction = 0.1 # So 10% validation

    dataset_name = f'Data15_{int(imgs_per_seed/1e6)}M_{int(valid_fraction*100)}%.h5'
    # dataset_name = f'Data15_{int(1)}k_{int(valid_fraction * 100)}%.h5' # Test dataset
    with h5py.File(rawdat_dir / dataset_name, 'r') as h_file:
        dataset_metadata = get_metadata(h_file)
        dataset_metadata.update({'name': dataset_name})
        params_list = dataset_metadata["parameters"]

    dataset = H5Dataset(rawdat_dir/dataset_name, 'training')
    valiset = H5Dataset(rawdat_dir/dataset_name, 'validation')


    net_config = {
        'pytorch_net': True,
        'depths': [2, 2, 2, 2],  # [2, 2, 2, 2] for 18, [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
        # 'block': Bottleneck,
        'output_features': 128
    }

    pre_process = transforms.Compose([
            transforms.Normalize((0, 0, 0), (1, 1, 1))])

    epochs = 15

    train_config = {
        'num_epochs': epochs,
        'optim_type': 'Adam',  # 'SGD'
        'learning_rate': 5e-5,  # 5e-5,
        
        'loader_kwargs': {
            'batch_size': 2048,
            'num_workers': 4,
            "collate_fn": h5_collate_fn,
            "pin_memory": True
        },

        'grad_clip': None,
        'sched_kwargs': {
            'type': 'cosine',
            'eta_min': 2e-5,
            'T_max': epochs
        },

        'dataset_info': dataset_metadata
    }

    flow_config = {
        'scales': {'chirp_mass': 120.0,
                'tilt_1': np.pi,
                'tilt_2': np.pi,
                'phi_jl': 2*np.pi,
                'phi_12': 2*np.pi,
                'luminosity_distance': 5000.0,  # 10000.0 if planning on using long distance datasets from the beginning
                'theta_jn': 2 * np.pi,
                'ra': 2*np.pi,
                'dec': np.pi,
                'phase': 2*np.pi,
                'psi': np.pi,
                "geocent_time": 1.0,
                },

        'input_dim': len(params_list),
        'context_dim': net_config['output_features'],

        # Here we can tweak the flow architecture.
        'num_flow_steps': 25,
        'base_transform': trans.d_rq_coupling_and_affine,
        'base_transform_kwargs': {
            'hidden_dim': 256, #512 leads to exploding gradients
            'num_transform_blocks': 5,
            'use_batch_norm': True,
        },
        'middle_transform': trans.random_perm_and_lulinear,
        'middle_transform_kwargs': {},
        'final_transform': trans.random_perm_and_lulinear,
        'final_transform_kwargs': {}
    }

    if new_model:
        '''Flow creation'''
        flow = CBCEstimator(params_list, flow_config, net_config, name=name, device=device,
                            workdir=outdir, mode='net+flow', preprocess=pre_process)
    else:
        '''Training continuation of previous model'''
        flow = CBCEstimator.load_from_file(old_model, device=device, workdir=outdir)
        flow.rename(name)  # new name if preferred


    flow.train(dataset,
               outdir,
               train_config,
               valiset, )
    # flow.turn_net_grad('off')  # If at some point we want to only train the flow
    flow.model_to_device('cpu') # To ensure files can be loaded on a CPU later
    flow.save_to_file(f'{flow.name}.pt')  # Saves it in the workdir
