# AlexNet from Scratch

## Project Overview
This project implements the **AlexNet** architecture from scratch using **PyTorch**. AlexNet was a pioneering deep convolutional neural network, achieving remarkable success in the 2012 ImageNet competition. Here, AlexNet is trained on the **FashionMNIST** dataset to classify fashion products into 10 different categories.

## Tech Stack
- **Language**: Python
- **Framework**: PyTorch
- **Libraries**:
  - `torch`: For building and training the deep learning model.
  - `torchvision`: To load and preprocess the FashionMNIST dataset.
  - `tqdm`: For displaying the training progress.
  - `torchsummary`: For summarizing the AlexNet architecture.
  - `nbimporter`: To import modules within Jupyter notebooks.

## Dataset
- **Dataset Used**: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- **Number of Classes**: 10 (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)
- **Number of Images**: 
  - Training set: 60,000 images.
  - Test set: 10,000 images.
- **Image Size**: The images are resized to **227x227** before being fed into the AlexNet model.

## Model Architecture

The AlexNet architecture consists of the following layers:

1. **Convolutional Layers**:
    - **Conv1**: 96 filters, 11x11 kernel, stride 4, padding 2.
    - **Conv2**: 256 filters, 5x5 kernel, stride 1, padding 2.
    - **Conv3**: 384 filters, 3x3 kernel, stride 1, padding 1.
    - **Conv4**: 384 filters, 3x3 kernel, stride 1, padding 1.
    - **Conv5**: 256 filters, 3x3 kernel, stride 1, padding 1.
  
2. **Pooling and Normalization**:
    - MaxPooling is applied after `Conv1`, `Conv2`, and `Conv5`.
    - Local Response Normalization (LRN) is applied after `Conv1` and `Conv2`.

3. **Fully Connected Layers**:
    - **FC1**: 4096 neurons.
    - **FC2**: 4096 neurons.
    - **FC3**: 100 neurons (adjustable for different datasets).
  
4. **Activation and Dropout**:
    - ReLU activation is used after each convolutional and fully connected layer.
    - Dropout is applied to the fully connected layers to prevent overfitting.

## Training Process

### Training Code (train.ipynb)
- **Data Preprocessing**: The dataset is resized to `227x227` and normalized with `transforms.Compose`.
- **Batch Size**: 128.
- **Optimizer**: SGD (Stochastic Gradient Descent) with momentum (0.9) and weight decay (0.0005).
- **Learning Rate**: 0.01 with a StepLR scheduler that reduces the learning rate after 50 epochs by a factor of 0.1.
- **Loss Function**: CrossEntropyLoss.

### How to Train the Model
1. **Install Required Libraries**:
    ```bash
    pip install torch torchvision tqdm
    ```

2. **Run the Training Script**:
    The training process can be executed using the provided Jupyter notebook `train.ipynb`. It will save checkpoints for each epoch and the final model weights.

    ```bash
    python train.ipynb
    ```

3. **Monitor Training**: Training progress (loss and accuracy) is displayed using `tqdm`.

4. **Saving Checkpoints**: After each epoch, a checkpoint is saved to the `checkpoints/` directory, allowing you to resume training if interrupted.

## Results

### Training Accuracy
- Final training accuracy achieved after 10 epochs: **~92%** on the FashionMNIST dataset.

### Loss Curve
You can plot the loss curve using the saved checkpoint data, helping you monitor convergence during training.

### Evaluation
To evaluate the model, use the saved model weights and the test dataset. You can visualize some predictions by plotting a few test images alongside their predicted labels.

## Future Improvements
- **Experiment with ImageNet**: The original AlexNet was trained on the ImageNet dataset. Training on FashionMNIST allows for easier experimentation, but ImageNet can provide richer results.
- **Data Augmentation**: Augmentations such as random crops, flips, and rotations could further improve model performance.
- **Reduce Learning Rate**: Adding a more aggressive learning rate decay could lead to better convergence, especially in longer training sessions.

## References
1. [Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, *"ImageNet Classification with Deep Convolutional Neural Networks,"* NIPS, 2012.](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
3. [FashionMNIST Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
