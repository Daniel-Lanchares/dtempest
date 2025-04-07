import h5py
import numpy as np
from pathlib import Path


from dtempest.gw import CBCEstimator
import dtempest.core.flow_utils as trans
from dtempest.core.train_utils import H5Dataset



if __name__ == '__main__':
    """

    """
    
    name = 'GP12' # Model name
    device = 'cpu' #'cuda'

    files_dir = Path(__file__).parent / 'files'  # Main directory
    dataset_dir = files_dir / 'Datasets'
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'publication_model'
    train_dir.mkdir(exist_ok=True)
    
    new_model = True  # Train from scratch (see later)
    
    
    imgs_per_seed = 1000 * 1000
    
    valid_fraction = 0.1
    

    dataset_name = f'Data12_{int(imgs_per_seed/1000)}k_{int(valid_fraction*100)}%.h5'
    
    dataset = H5Dataset(dataset_dir/dataset_name, 'training')
    valiset = H5Dataset(dataset_dir/dataset_name, 'validation')



    with h5py.File(dataset_dir / dataset_name, 'r') as h_file:

        estimation_parameters = h_file.attrs['parameters']
        # TODO: Add resolution, etc, to model metadata from here

    net_config = {
        'pytorch_net': True,
        'depths': [2, 2, 2, 2],  # [2, 2, 2, 2] for 18, [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
        # 'block': Bottleneck,
        'output_features': 128
    }

    # pre_process = transforms.Compose([
    #         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    pre_process = None

    init_epochs = 15
    n_epochs = 25 
    
    
    init_config = {
        'num_epochs': init_epochs,
        'optim_type': 'Adam',  # 'SGD'
        'learning_rate': 5e-5,  # 5e-5,
        
        'loader_kwargs': {
            'batch_size': 2048,  # 64
            'num_workers': 4,
            'shuffle': True
        },

        'grad_clip': None,
        'sched_kwargs': {
            'type': 'cosine',
            'eta_min': 1e-5,
            'T_max': init_epochs
            }
    }
    
    
    train_config = {
        'num_epochs': n_epochs,
        'optim_type': 'Adam',  # 'SGD'
        'learning_rate': 1e-5,  # 0.001,
        
        'loader_kwargs': {
            'batch_size': 2048,  # 4096 is too much for GPU
            'num_workers': 4,
            # 'shuffle': True
        },

        'grad_clip': None,
        'sched_kwargs': {
            'type': 'cosine',
            'T_max': n_epochs,
            }
    }

    flow_config = {
        'scales': {'chirp_mass': 100.0,
                'tilt_1': np.pi,
                'tilt_2': np.pi,
                'phi_jl': 2*np.pi,
                'phi_12': 2*np.pi,
                'luminosity_distance': 5000.0,
                'theta_jn': 2 * np.pi,
                'ra': 2*np.pi,
                'dec': np.pi,
                'phase': 2*np.pi,
                'psi': np.pi,},

        'input_dim': len(estimation_parameters),
        'context_dim': net_config['output_features'],

        # Here we can tweak the flow architecture.
        'num_flow_steps': 16, 
        'base_transform': trans.d_rq_coupling_and_affine,  # Affine works best in low epochs, overfits too soon
        'base_transform_kwargs': {
            'hidden_dim': 128,
            'num_transform_blocks': 5,
            'use_batch_norm': True,
        },
        'middle_transform': trans.random_perm_and_lulinear,
        'middle_transform_kwargs': {

        },
        'final_transform': trans.random_perm_and_lulinear,
        'final_transform_kwargs': {

        }
    }


    if new_model:
        '''Flow creation'''
        flow = CBCEstimator(estimation_parameters, flow_config, net_config, name=name, device=device,
                            workdir=outdir, mode='net+flow', preprocess=pre_process)
    else:
        '''Training continuation of previous model'''
        old_model = train_dir / 'training_test_x' / 'Old_model.pt'
        flow = CBCEstimator.load_from_file(old_model, device=device, workdir=outdir)
        flow.rename(name)  # new name if preferred



    flow.train(dataset,
               outdir,
               init_config,
               valiset, )
    flow.save_to_file(f'{flow.name}_checkpoint.pt')

    flow.turn_net_grad('off')  # Now we only train the flow
    flow.train(dataset,
               outdir,
               train_config,
               valiset, )
    flow.save_to_file(f'{flow.name}.pt')  # Saves it in the workdir
