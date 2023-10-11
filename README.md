# Img2Voxel

3D shape reconstruction of images to voxels using the ShapeNet dataset.

## Architecture
The architecture is based off of a UNet encoder decoder structure. We first learn an embedding space from voxels to voxels (a), and then try to match that embedding space with an image encoder (b), and then we finetune the final model (c).

![Architecture](figs/arch.png?raw=true)

## Results
The finetuned model achieved an average IoU of 0.41

### Examples
Examples of reconstructions from images

![Examples](figs/results.png?raw=true)

### Latent Interpolation
Examples of interpolating the latent embeddings from the encoder

![Latent Interpolation](figs/latent_interp.png?raw=true)
