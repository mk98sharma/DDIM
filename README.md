# DDIM
# Denoising Diffusion Implicit Model
Denoising Diffusion Implicit Model
This repository contains the implementation of the Denoising Diffusion Implicit Model (DDIM), a deep generative model for image generation and restoration. 

Requirements
The implementation requires the following packages:

Python 3.6+
Tensorflow 2.9+
These packages can be installed using pip or conda.

Usage
The implementation provides two main scripts: train.py and generate.py.

Training
The ddim.py script is used to train the DDIM model on a dataset. The following arguments can be used to configure the training:

--dataset: Path to the dataset directory.
--batch_size: Batch size used during training. Default is 32.
--num_steps: Number of diffusion steps used during training. Default is 1000.
--num_samples: Number of samples used to estimate the diffusion probabilities. Default is 100.
--lr: Learning rate used during training. Default is 1e-4.
--beta1: Adam optimizer beta1 parameter. Default is 0.9.
--beta2: Adam optimizer beta2 parameter. Default is 0.999.
--eps: Adam optimizer epsilon parameter. Default is 1e-8.
--save_dir: Directory where checkpoints and samples will be saved. Default is ./output.
--save_interval: Number of training steps between checkpoint saves. Default is 1000.
--log_interval: Number of training steps between log prints. Default is 100.
--resume: Path to a checkpoint to resume training from.
For example, to train the DDIM model on the CelebA dataset with a batch size of 64 and 500 diffusion steps, run:
dataset /path/to/celeba --batch_size 64 --num_steps 500
Generation
The generate.py script is used to generate images from a trained DDIM model. The following arguments can be used to configure the generation:

--checkpoint: Path to the checkpoint of the trained model.
--num_samples: Number of images to generate. Default is 16.
--save_dir: Directory where generated images will be saved. Default is ./output.
--batch_size: Batch size used during generation. Default is 16.
For example, to generate 64 images from a trained model checkpoint, run:

checkpoint /path/to/checkpoint.pt --num_samples 64
Acknowledgments
This implementation is based on the code provided by the authors of the paper "Training Generative Models with Denoising Diffusion Probabilities".
# Results 
note:this is not actual output, this is made by compressing 25 epochs output
![ddimflowers (1)](https://user-images.githubusercontent.com/77893734/227646759-62c288f6-751c-4fc4-8379-dbdf4dc1eeda.gif)
