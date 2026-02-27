import numpy as np
from pathlib import Path
from torchvision.models.resnet import Bottleneck

from dtempest.gw import CBCEstimator
import dtempest.core.flow_utils as trans
# from dtempest.core.train_utils import H5Dataset



if __name__ == '__main__':
    """

    """
    
    name = 'Loader_test_continuation' # Model name
    device = 'cpu' #'cuda'

    files_dir = Path(__file__).parent / 'files'  # Main directory
    # dataset_dir = files_dir / 'Datasets'
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'loader_model'
    # outdir = ''
    train_dir.mkdir(exist_ok=True)
    
    new_model = False  # Train from scratch (see later)

    estimation_parameters = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_jl', 'phi_12',
              'luminosity_distance', 'theta_jn', 'ra', 'dec', 'phase', 'psi']


    
    
    # imgs_per_seed = 1000 * 1000
    
    # valid_fraction = 0.1
    

    # dataset_name = f'Data12_{int(imgs_per_seed/1000)}k_{int(valid_fraction*100)}%.h5'
    
    # dataset = H5Dataset(dataset_dir/dataset_name, 'training')
    # valiset = H5Dataset(dataset_dir/dataset_name, 'validation')



    # with h5py.File(dataset_dir / dataset_name, 'r') as h_file:
    #
    #     estimation_parameters = h_file.attrs['parameters']
    #     # TODO: Add resolution, etc, to model metadata from here

    net_config = {
        'pytorch_net': True,
        'depths': [3,4,6,3],#[2, 2, 2, 2],  # [2, 2, 2, 2] for 18, [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
        'block': Bottleneck,
        'output_features': 128
    }

    # pre_process = transforms.Compose([
    #         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    pre_process = None

    # init_epochs = 15
    # n_epochs = 25
    
    
    # init_config = {
    #     'num_epochs': init_epochs,
    #     'optim_type': 'Adam',  # 'SGD'
    #     'learning_rate': 5e-5,  # 5e-5,
    #
    #     'loader_kwargs': {
    #         'batch_size': 2048,  # 64
    #         'num_workers': 4,
    #         'shuffle': True
    #     },
    #
    #     'grad_clip': None,
    #     'sched_kwargs': {
    #         'type': 'cosine',
    #         'eta_min': 1e-5,
    #         'T_max': init_epochs
    #         }
    # }
    
    
    # train_config = {
    #     'num_epochs': n_epochs,
    #     'optim_type': 'Adam',  # 'SGD'
    #     'learning_rate': 1e-5,  # 0.001,
    #
    #     'loader_kwargs': {
    #         'batch_size': 2048,  # 4096 is too much for GPU
    #         'num_workers': 4,
    #         # 'shuffle': True
    #     },
    #
    #     'grad_clip': None,
    #     'sched_kwargs': {
    #         'type': 'cosine',
    #         'T_max': n_epochs,
    #         }
    # }

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
        old_model = outdir  / 'Loader_test.pt'
        flow = CBCEstimator.load_from_file(old_model, device=device, workdir=outdir)
        flow.rename(name)  # new name if preferred
        flow.scales = flow._get_scales(flow_config['scales'])



    # flow.train(dataset,
    #            outdir,
    #            init_config,
    #            valiset, )
    # flow.save_to_file(f'{flow.name}_checkpoint.pt')
    #
    # flow.turn_net_grad('off')  # Now we only train the flow
    # flow.train(dataset,
    #            outdir,
    #            train_config,
    #            valiset, )
    # flow.save_to_file(f'{flow.name}.pt')  # Saves it in the workdir

    import bilby
    from simulator import JointLoader, loader_from_bilby_prior, default_injection_config
    from bilby.core.utils.log import logger as bilby_logger

    bilby_logger.setLevel('ERROR')
    prior = bilby.gw.prior.BBHPriorDict()

    injection_kwargs = default_injection_config()
    injection_kwargs['snr_lims'] = (5, 30)

    loader_kwargs = {
        'batch_size': 2 ** 6,
        'vectorized': False
    }
    prior, sim = loader_from_bilby_prior(estimation_parameters, 'dual', prior, **injection_kwargs)
    loader = JointLoader(prior, sim, **loader_kwargs)

    import torch
    import torch.optim as optim
    from itertools import islice
    device = 'cpu'
    train_data = 2**7
    val_data = 2**5
    epochs = 10

    from copy import copy
    model = flow.model
    scales = copy(flow.scales)
    shifts = copy(flow.shifts)
    preprocess = flow.preprocess

    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):#(bar := trange(epochs, unit='epoch')):
        losses, val_losses = [], []

        i = 0
        for theta, (ts, img) in islice(loader, train_data):
            # for image in img:
            #     plt.imshow(torch.permute(image, (1, 2, 0)))
            #     plt.show()
            # print(theta.dtype, ts.dtype, img.dtype)
            # train_loss = loss(theta.to(device), (ts.to(device), img.to(device)))
            # train_loss = -estimator(theta.to(device), (ts.to(device), img.to(device))).mean()

            y = (np.divide(theta, scales) - shifts).to(device)
            x = preprocess(img).to(device)
            # print(y.float())
            train_loss = -model.log_prob(inputs=y.float(), context=x).mean()
            print(f'Epoch {epoch + 1:3d}, batch {i:3d}: {train_loss.item():.4}')
            losses.append(train_loss)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            # print(i, train_loss.item())
            i += 1

        for theta, (ts, img) in islice(loader, val_data):
            y = (np.divide(theta, scales) - shifts).to(device)
            x = preprocess(img).to(device)
            val_losses.append(-model.log_prob(inputs=y.float(), context=x).mean())


        scheduler.step()
        print()
        print(f'Epoch {epoch + 1:3d}')
        print(f'Train loss: {torch.stack(losses).mean().item():.5}')
        print(f'Valid loss: {torch.stack(val_losses).mean().item():.5}')
        print(f'Changing learning rate to: {scheduler.get_last_lr()[0]:.5}')
        print()
        if epoch != 0 and epoch % 4 == 0:
            flow.save_to_file(f'{flow.name}_chkpt_{epoch}.pt')
    flow.save_to_file(f'{flow.name}.pt')

'''
Loader_test
10 epochs of 64 batches of 64 pictures: 
train: -1.8459
valid: -------

# Slow but steady improvement. May be a technique suitable for refinement as opposed to full training

'''