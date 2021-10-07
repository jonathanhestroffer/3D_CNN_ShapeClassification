# Basic Shape Classification Using 3D CNNs

<img src="title.png" width=100% height=100%>

INTRO HERE

This repository provides a short tutorial on basic shape classification using three-dimensional (3D) convolutional neural networks (CNNs). All data for training and testing the model in this example has been provided.

## Methodology

### Data Generation and Augmentation

<img src="shapes.png" width=100% height=100%>

The goal of the proposed model is to classify 3D image data according to the primitive shapes they represent. Three basic shapes are considered for classification and they include the *cube*, *cylinder* and *regular tetrahedron*. These shapes were generated with Blender in the form of triangle surface mesh (.stl) and can be seen in the figure above.

For each of the three shapes, and for a pre-determined number of samples desired for each shape, the process of generating a sample image is as follows:

1. Place the Blender generated primitive shape at the center of a bounding box.
2. Sequentially apply a random 3D rotation, scale, and displacement to the shape. (Augmentation)
3. Define a voxel-grid encompassing both the permutated shape and bounding box.
4. Segment the voxels according to their location with respect to the permutated shape. Voxels whose centroids are located within the permutated shape are given a label of "1", if not a label of "0" is assigned.

<img src="data_generation_workflow.png" width=100% height=100%>

As can be seen from the figure above illustrating the data generation workflow, Step 2 represents a series of data augmentation procedures; without it, there would only be 1 possible sample per image. For this example, 1000 samples of data are generated for each shape.

### Model Architecture

```python
if True:
  return
```

### Training

## Results and Analysis
