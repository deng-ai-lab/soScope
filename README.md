# A unified generative deep learning model for resolution enhancement across spatial omics platforms 

Spatial omics scope (soScope) is a unified generative framework designed for enhancing data quality and spatial resolution across various omics types obtained from diverse spatial technologies.

## Overview

Tissues are constructed from cells with distinct molecular states and spatial organizations, necessitating a thorough characterization of diverse molecular profiles while preserving their spatial contexts for extensive tissue architecture dissection. Recent advances in spatial omics technologies have enabled the profiling of various molecular categories, including transcript, protein, epigenetic marker, and genomic variation. This unveiling of spatial signatures across diverse molecular profiles offers valuable insights across various biological areas. Despite early successes, two challenges persist. Firstly, tissues are often in frozen or formalin-fixed and paraffin-embedded (FFPE) states before sequencing, potentially impacting molecular states and reducing sequencing accuracy. Secondly, most spatial technologies utilize spatial barcodes at the tissue spot resolution, which, given the multiple-cell composition of each spot, limits spatial resolution in tissue structure.

 To address these challenges, we introduce spatial omics scope (soScope), a generative framework that enhances spatial resolution and data quality by modeling spot-level profiles from diverse spatial omics technologies. SoScope views each spot as an aggregation of "subspots" at an enhanced spatial resolution, integrating omics profiles, spatial relations, and high-resolution morphological images through a multimodal deep learning framework, enabling accurate modeling and reduction of variations in diverse spatial omics types.

<img src="overview.png" alt="image-20230928102849342"  />
**Fig. 1 | An overview of the study.** (**a**) The soScope framework. soScope integrates molecular profiles ($\bold X$), spatial neighboring relations ($\bold A$), and morphological image features ($\bold Y$) from the same tissue using a unified generative model to enhance spatial resolution and refine data quality for diverse spatial omics profiles. (**b**) The probabilistic graphical model representation of soScope. Each of the   spots in the spatial data is considered an aggregation of   subspots at a higher spatial resolution. The subspot omics profile ($\bold{\hat X}$) depends on both the latent states ($\bold Z$) at the spot level and image features ($\bold Y$) at the subspot level. The observed profile is obtained by summing profiles from its subspots.

![image-20230928103553984](model.png)

**Fig. 2 | The model architecture of soScope**

The model includes three parts: Firstly, at the spot resolution, omics profile ($\bold X$), and their spatial neighboring relations ($\bold A$) are encoded by a 3-layer graph transformer and mapped to parameters ($\bold{\mu} ^{(n)}$ and $\bold{\sigma} ^{(n)}$ for spot $s^{(n)}$) defining the latent distribution for $\bold Z$ . Spatial states $\bold Z$ are sampled via the reparameterization trick. Secondly, at the subspot resolution, image patches from subspot regions are converted into deep features $\bold Y$ (**Methods**) and concatenated with the spot representation $\bold Z$ . Thirdly, the combined input is mapped to distribution parameters $\bold{\omega}_k ^{(n)}$ for subspots’ profiles $\bold{\hat X}$ through two sequential ResNet blocks. Here, $\bold{\omega}_k ^{(n)}$ represents likelihood parameters for the  $k$-th subspot enhanced from the $n$-th spot, which is determined by the omics type . An additional image regularization term is used to encourage the consistency between enhanced profile similarity ($\bold{\Lambda}$) and morphological similarity ($\bold W$) at the subspot level (blue line).

## soScope software package

soScope requires the following packages for installation:

- Python >= 3.6
- PyTroch= 1.8.0
- PyG >= 1.7.2
- Numpy >= 1.16.2
- Scipy = 1.10.1
- scikit-learn = 1.2.0

All required python packages can be installed through `pip/conda` command. 

To install soScope package, use

```terminal
git clone https://github.com/deng-ai-lab/soScope
```

## Usage

### Image feature inception

By running `image_inception.py` on image patches,  users can get 2048-dimensional image feature  ($\bold Y$) .

### Graph building

By running `BuildGraph.py` on spatial profiles and image feature,  users can get spatial neighboring relations ($\bold A$) and morphological similarity ($\bold W$) in coordinate format.

### Import soScope python package

After installation, import soScope by

```python
from soScope_model.train import two_step_train  # for model training
from soScope_model.inference import infer  # for enhanced profiles inference
```

### Train soScope on multimodal data

soScope requires spatial profiles ($N\times G$),  spatial neighboring relations ($3\times \# edges$ , a sparse matrix in coordinate format) ,  image features ($NK\times 2048$), and image similarity matrix ($3\times \# edges$ , a sparse matrix in coordinate format) for model training. All of these data should be provided in  `data_dir`.

```python
two_step_train(
        logging,
        vgae_experiment_dir,
        svae_experiment_dir,
        data_dir,
        vgae_config_file,
        svae_config_file,
        device,
        checkpoint_every,
        backup_every,
        epochs,
        num_neighbors,
):
```
where 

```
Parameters:

	logging: Not None to create an event file in a given dictionary and add summaries in training process.
	vgae_experiment_dir: saving directory for pre-training stage.
	svae_experiment_dir: saving directory soScope training.
	data_dir: dataset directory contains necessary data mentioned above.
	vgae_config_file: model configuration for variational graph auto-encoder used in pre-training stage.
	svae_config_file: model configuration for soScope.
	device, checkpoint_every, backup_every, epochs: default training settings.
	num_neighbors: edges are built between every neighboring num_neighbors nodes, not to be revised. For Visium, num_neighbors=6. For other platforms, num_neighbors=4.
```

### Inference enhanced spatial omics profiles

After optimization,  the It can automatically determine the optimal cluster number. 

```python
infer(experiment_dir,
      non_negative,
      num_neighbors,
      data_dir,
      result_dir,
      device)
```

where

```
Parameters:
	experiment_dir: saving directory for inference stage.
	non_negative, device: default training settings.
	num_neighbors: edges are built between every neighboring num_neighbors nodes, not to be revised. For Visium, num_neighbors=6. For other platforms, num_neighbors=4.
	data_dir: dataset directory contains necessary data mentioned above.
	result_dir: saving directory for results.
```

## Demonstration

We provide Jupyter Notebooks (see `soScope_demo`) for the demonstration of applying soScope on negative binomial distribution, Poisson distribution, Gaussian distribution and joint distribution for spatial multiomics. The demonstration includes:

1. Data in aforementioned distribution;
2. Config files for neural networks;
3. Train and inference codes;
4. Visualization of results.

## Copyright
Software provided as is under **MIT License**.

Bohan Li @ 2023 BUAA and Deng ai Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

