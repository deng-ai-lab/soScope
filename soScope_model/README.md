# soScope model

 `image_inception.py` : Codes for image feature inception.

`BuildGraph.py` : Codes for graph building.

`train.py` : Codes for training soScope.

`inference.py`: Codes for inference using optimized soScope.

 `training`: Within this folder, there are network structures. There is a variational graph autoencoder `vgae.py` for pretraining and 4 soScope models `soScope_model_for_Gaussian.py` , `soScope_model_for_NB.py` , `soScope_model_for_Poission.py` , and `soScope_model_for_Multiomics.py` for fine tune and inference. 

 `utils`:  Basic network blocks and functions used.

## Usage

There are 3 steps to use soScope:

### 1. Image feature inception

By running `image_inception.py` on image patches,  users can get 2048-dimensional image feature  ($Y$) .

The model of pretrained inception-v3 can be accessed by [link](https://www.dropbox.com/scl/fo/igdq4lf0kzlnt5z3ddugs/h?rlkey=kxdkb7q4pisgo2lsn0s5p1qyh&dl=0).

### 2. Graph building

By running `BuildGraph.py` on spatial profiles and image feature,  users can get spatial neighboring relations ($A$) and morphological similarity ($W$) in coordinate format.

### 3. Training and inference

`train.py` and `inference.py` are used to optimize and use model in folder `training`.

#### 1) Import soScope python package



```python
from soScope_model.train import two_step_train  # for model training
from soScope_model.inference import infer  # for enhanced profiles inference
```

#### 2) Train soScope on multimodal data

soScope requires spatial profiles ($N\times G$),  spatial neighboring relations ($3\times N_{edges}$ , a sparse matrix in coordinate format) ,  image features ($NK\times 2048$), and image similarity matrix ($3\times N_{edges}$ , a sparse matrix in coordinate format) for model training. All of these data should be provided in  `data_dir`. After optimization, the soScope model is saved in `soScope_experiment_dir`.

```python
two_step_train(logging,
               vgae_experiment_dir,
               soScope_experiment_dir,
               data_dir,
               vgae_config_file,
               soScope_config_file,
               device,
               checkpoint_every,
               backup_every,
               epochs,
               num_neighbors=4
              )
```
where 

```
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

Returns:
	Optimized soScope model.

```

#### 3) Inference enhanced spatial omics profiles

After optimization, users can directly get enhanced spatial profiles. 

```python
infer(
        experiment_dir,
        non_negative,
        num_neighbors,
        data_dir,
        result_dir,
        device,
)
```

where

```
Args:
    experiment_dir: loading optimized model from this directory for inference stage.
    non_negative: True to make the enhanced profiles not negative.
    num_neighbors: edges are built between every neighboring {num_neighbors} nodes, not to be revised. num_neighbors=6 for Visium and num_neighbors=4 for other platforms.
    data_dir: dataset directory contains necessary data mentioned above.
    result_dir: saving directory for results.
    device: 'cuda' or 'cpu'

Returns:
	Enhanced spatial profiles saved as {result_dir}/infer_subspot.npy
```
