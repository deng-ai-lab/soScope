# Demonstration

We provide 4 Jupyter Notebook  demonstrations under `soScope_demo`. The demonstrations include 4 spatial omics:

1.  `soScope_demo/soScope_demo_NB.ipynb`: Negative binomial distribution for spatial transcriptomics;
2.  `soScope_demo/soScope_demo_Poisson.ipynb`: Poisson distribution for spatial-CUT&Tag;
3.  `soScope_demo/soScope_demo_Gaussian.ipynb`: Gaussian distribution for slide-DNA-seq PCs;
4.  `soScope_demo/soScope_demo_Multiomics.ipynb`: Poisson and negative binomial distribution for spatial-CITE.

Our demonstrations are run by Python = 3.6 with packages PyTroch= 1.8.0, PyG = 1.7.2, and Numpy = 1.16.2.  We take the Jupyter Notebooks `soScope_demo/soScope_demo_NB.ipynb` as an example to explain the soScope settings.

#### 1. Data

369 “low-resolution” spots with aggregated gene expressions (X), morphological image features generated from a pretrained Inception-v3 model at high resolution (Y), and spatial neighboring relations (A). Genes analyzed: MT1G, FABP1, EPCAM in the Epithelium region; CNN1, MYH11, TAGLN in the Muscularis region; PTPRC, HLA-DRA, CD74 in the Immune region.

Demo dataset can be accessed by [link](https://www.dropbox.com/scl/fo/igdq4lf0kzlnt5z3ddugs/h?rlkey=kxdkb7q4pisgo2lsn0s5p1qyh&dl=0).

#### 2. Config files for neural networks

vgae_config_file:

```yaml
trainer: VGAETrainer_NB 
# We adopt the negative binomial distribution as the prior distribution to pretrain the graph encoder.

params:
  # Number of analyzed genes
  gene_dim: 9
  
  # Number of subspots in each spot
  sub_node: 7
  
  # Dimension of image features
  sub_dim: 2048
  
  # Dimension of latent states
  z_dim: 128
  
  # Optimizer name
  optimizer_name: 'Adam'
  
  # A hyperparameter indicating the variarance of X.
  scale: 10
  
  # Learing rate
  lr: 0.00_1
  
  # Beta is the weight of KL divergence. We offer a warm-up startegy to  optimize log p(x|z) first with a low initial beta at
  # the first 5000 iterations, and optimize log p(x|z)-beta*KL with a beta=1 after the network is trained 15000 iterations.
  # In practice, we use and suggest a initial beta=1 to optimize the graph varaitional directly.
  beta_start_value: 1
  beta_end_value: 1
  beta_n_iterations: 10000
  beta_start_iteration: 5000
```

soScope_config_file:

```yaml
trainer: soScope_NB
# We adopt the negative binomial distribution as the prior distribution to train soScope.
params:
  # Number of analyzed genes
  gene_dim: 9
  
  # Number of subspots in each spot
  sub_node: 7
  
  # Dimension of image features
  sub_dim: 2048
  
  # Dimension of latent states
  z_dim: 128
  
  # Optimizer name
  optimizer_name: 'Adam'
  
  # A hyperparameter indicating the variance of the white Gaussian noise defined in Method
  scale: 1
  
  # Learing rate
  lr: 0.00_05
  
  # Gamma is the learning rate decay. The network is optimized by lr*loss in the first 5000 iterations
  # and decrease to 0.01*lr*loss after 15000 iterations.
  gamma_start_value: 1
  gamma_end_value: 0.01
  
  # Beta is the weight of KL divergence.
  beta_start_value: 1
  beta_end_value: 1
  beta_n_iterations: 10000
  beta_start_iteration: 5000
```

