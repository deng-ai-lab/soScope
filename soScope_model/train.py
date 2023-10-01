import os
import yaml
from tqdm import tqdm
from soScope_model.utils.ReadData import get_global_dataset, get_global_spatial_dataset, get_global_expression_spatial_dataset
import soScope_model.training as training_module


def train(trainer, experiment_dir, train_set, st_data, writer, epochs, checkpoint_every, backup_every):
    checkpoint_count = 1
    trainer.iterations = 0
    for epoch in tqdm(range(epochs)):

        for data in train_set:
            trainer.train_step(st_data=st_data, sub_data=data)

        if epoch % checkpoint_every == 0:
            while os.path.isfile(os.path.join(experiment_dir, 'checkpoint_%d.pt' % checkpoint_count)):
                checkpoint_count += 1
            trainer.save(os.path.join(experiment_dir, 'checkpoint_%d.pt' % checkpoint_count))
            checkpoint_count += 1

        if epoch % backup_every == 0:
            trainer.save(os.path.join(experiment_dir, 'model.pt'))

def two_step_train(
        logging,
        vgae_experiment_dir,
        soScope_experiment_dir,
        data_dir,
        vgae_config_file,
        soScope_config_file,
        device,
        checkpoint_every,
        backup_every,
        epochs,
        num_neighbors,
):
    """
Args:
    logging: not None to log the summary in the training process.
    vgae_experiment_dir: saving directory for pre-training stage.
    soScope_experiment_dir: saving directory soScope training.
    data_dir: dataset directory contains necessary data mentioned above.
    vgae_config_file: model configuration for variational graph auto-encoder used in pre-training stage.
    soScope_config_file: model configuration for soScope.
    device: 'cuda' or 'cpu'
    checkpoint_every: save the model in each check point.
    backup_every: update the model in each backup point.
    epochs: training epoches.
    num_neighbors: edges are built between every neighboring {num_neighbors} nodes, not to be revised. num_neighbors=6 for Visium and num_neighbors=4 for other platforms.

	Optimized soScope model is saved in soScope_experiment_dir
    """
    if logging:
        from torch.utils.tensorboard import SummaryWriter
        vgae_writer = SummaryWriter(log_dir=vgae_experiment_dir)
        soScope_writer = SummaryWriter(log_dir=soScope_experiment_dir)
    else:
        os.makedirs(vgae_experiment_dir, exist_ok=True)
        os.makedirs(soScope_experiment_dir, exist_ok=True)
        vgae_writer = None
        soScope_writer = None

    # Load the configuration file
    with open(vgae_config_file, 'r') as file:
        vgae_config = yaml.safe_load(file)

    with open(soScope_config_file, 'r') as file:
        soScope_config = yaml.safe_load(file)

    # Copy it to the experiment folder
    with open(os.path.join(vgae_experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(vgae_config, file)

    with open(os.path.join(soScope_experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(soScope_config, file)

    # Instantiating the trainer according to the specified configuration
    VGAETrainerClass = getattr(training_module, vgae_config['trainer'])
    print('Step 1')
    print(VGAETrainerClass)
    vgae_trainer = VGAETrainerClass(writer=vgae_writer, **vgae_config['params'])
    vgae_trainer.to(device)

    ###########
    # Dataset #
    ###########
    # Loading the dataset
    train_set = get_global_dataset(root_path=data_dir)
    st_data = get_global_spatial_dataset(root_path=data_dir, mode=num_neighbors)
    print('Dataset loaded!')
    ##########

    ##############
    # Train VGAE #
    ##############
    print('========== Initialization the graph encoder ============')
    train(vgae_trainer, vgae_experiment_dir, train_set, st_data, vgae_writer, epochs[0], checkpoint_every, backup_every)
    # for pdac_A set epoch to 10

    ##############
    # Train soScope #
    ##############
    # Instantiating the trainer according to the specified configuration
    soScopeTrainerClass = getattr(training_module, soScope_config['trainer'])
    print('Step 2')
    print(soScopeTrainerClass)
    soScope_trainer = soScopeTrainerClass(writer=soScope_writer, **soScope_config['params'])
    soScope_trainer.to(device)

    load_model_file = os.path.join(vgae_experiment_dir, 'model.pt')
    # Resume the training if specified
    if load_model_file:
        soScope_trainer.load(load_model_file)
        print('Pretrained Model Loaded!')

    print('========== Optimization of soScope ============')
    train(soScope_trainer, soScope_experiment_dir, train_set, st_data, soScope_writer, epochs[1], checkpoint_every, backup_every)

if __name__=='__main__':
    two_step_train(logging=not None,
                   vgae_experiment_dir='/home/lbh/projects_dir/soScope/soScope_demo/experiments/VGAE_Gaussian',
                   soScope_experiment_dir='/home/lbh/projects_dir/soScope/soScope_demo/experiments/soScope_Gaussian',
                   data_dir='/home/lbh/projects_dir/soScope/soScope_demo/DataSet/Gaussian_demo/',
                   vgae_config_file='/home/lbh/projects_dir/soScope/soScope_demo/config/Gaussian/VGAE.yml',
                   soScope_config_file='/home/lbh/projects_dir/soScope/soScope_demo/config/Gaussian/soScope.yml',
                   device='cuda',
                   checkpoint_every=1000,
                   backup_every=200,
                   epochs=[400, 400],
                   num_neighbors=4)