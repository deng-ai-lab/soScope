import os
import yaml
from soScope_model.utils.ReadData import get_global_spatial_dataset
import soScope_model.training as training_module
import numpy as np
import torch

def infer(
        experiment_dir,
        non_negative,
        num_neighbors,
        data_dir,
        result_dir,
        device='cuda'):
    """

    Args:
        experiment_dir: loading optimized model from this directory for inference stage.
        non_negative: True to make the enhanced profiles not negative.
        num_neighbors: edges are built between every neighboring {num_neighbors} nodes, not to be revised. num_neighbors=6 for Visium and num_neighbors=4 for other platforms.
        data_dir: dataset directory contains necessary data mentioned above.
        result_dir: saving directory for results.
        device: 'cuda' or 'cpu'

    Enhanced spatial profiles saved as {result_dir}/infer_subspot.npy
    """
    config_file = None
    # Check if the experiment directory already contains a model
    pretrained = os.path.isfile(os.path.join(experiment_dir, 'model.pt')) \
                 and os.path.isfile(os.path.join(experiment_dir, 'config.yml'))

    if pretrained and not (config_file is None) and not overwrite:
        raise Exception("The experiment directory %s already contains a trained model, please specify a different "
                        "experiment directory or remove the --config-file option to resume training or use the --overwrite"
                        "flag to force overwriting")

    resume_training = pretrained

    if resume_training:
        load_model_file = os.path.join(experiment_dir, 'model.pt')
        config_file = os.path.join(experiment_dir, 'config.yml')
    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Copy it to the experiment folder
    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)

    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, config['trainer'])
    print(TrainerClass)
    trainer = TrainerClass(writer=None, **config['params'])
    trainer.to(device)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Resume the training if specified
    if load_model_file:
        trainer.load(load_model_file)
        print('Pretrained Model Loaded!')
    trainer.to(device)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #############
    # inference #
    #############

    st_data = get_global_spatial_dataset(root_path=data_dir, mode=num_neighbors)
    for i, item in enumerate(st_data):
        st_data[i] = item.to(device)

    sub_adj = np.load(f'{data_dir}/sub_adj.npy')
    subspot = np.load(f'{data_dir}/support_feature.npy')

    subspot = torch.tensor(subspot).float().to(device)
    sub_edge = torch.tensor(sub_adj[:, :2].T).long().to(device)
    sub_edge_value = torch.tensor(sub_adj[:, 2].T).float().to(device)

    sub_data = [subspot, sub_edge, sub_edge_value, None]

    sub_x_pred = trainer.infer(st_data, sub_data, nonnegative=non_negative).cpu().numpy()
    np.save(result_dir + f'/infer_subspot.npy', sub_x_pred)

if __name__=='__main__':
    infer(
        experiment_dir= '/home/lbh/projects_dir/soScope/soScope_demo/experiments/soScope_Gaussian',
        non_negative=True,
        num_neighbors=6,
        data_dir= '/home/lbh/projects_dir/soScope/soScope_demo/DataSet/Gaussian_demo/',
        result_dir='/home/lbh/projects_dir/soScope/soScope_demo/DataSet/Gaussian_demo/',
        device='cuda')