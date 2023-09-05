# ECE324 - Machine Intelligence, Software & Neural Network

## Project Overview

This repository contains the code and documentation for my project in ECE324 - Machine Intelligence, Software & Neural Network. In this project, I implemented a Convolutional Neural Network (CNN) to address a specific problem related to my research paper. This README file will provide an overview of the project, discuss the CNN architecture I used, and explain how it contributes to the research paper's objectives.

## Table of Contents

- [Project Overview](#project-overview)
- [CNN Architecture](#cnn-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## CNN Architecture

The heart of this project is the Convolutional Neural Network (CNN) architecture that I designed and implemented. The CNN architecture plays a crucial role in achieving the objectives outlined in our research paper's abstract. Below, I provide an overview of the CNN's structure:

### CNN Architecture Overview

- **Input Layer**: The input layer receives image data. In my research, I used a dataset consisting of [describe the dataset here, e.g., a collection of handwritten digit images].

- **Convolutional Layers**: The CNN comprises several convolutional layers that extract features from the input images. These layers use learnable filters to detect patterns, edges, and textures within the images.

- **Pooling Layers**: After each convolutional layer, I included pooling layers to reduce the spatial dimensions of the feature maps, thus aiding in reducing computation while preserving essential information.

- **Fully Connected Layers**: Following the convolutional and pooling layers, there are fully connected layers. These layers make predictions and classify the input images into their respective categories. The number of neurons in the final fully connected layer corresponds to the number of classes in our research problem.

- **Output Layer**: The output layer provides the final classification probabilities, using an appropriate activation function (e.g., softmax) for multi-class classification.

This CNN architecture was carefully designed and fine-tuned to maximize the model's performance for our specific research problem. The model's architecture is available in the `model.py` file in this repository.

## Dataset

To train and evaluate the CNN, I used a dataset that aligns with the objectives outlined in our research paper's abstract. The dataset consists of [provide a brief description of the dataset, its source, and its relevance to the research problem]. The dataset is included in the `data` directory of this repository.

## Training

To train the CNN model, I used the dataset mentioned above. The training process involved optimizing the model's parameters and weights to minimize the classification error. Key training details include:

- **Data Augmentation**: I applied data augmentation techniques to increase the diversity of training examples and reduce overfitting.

- **Loss Function**: The model was trained using an appropriate loss function, which was chosen to suit the problem's objectives.

- **Optimizer**: I employed a suitable optimizer (e.g., Adam, SGD) to update the model's weights during training.

The training script can be found in the `cnn.py` file. You can use this script to train the model on your own dataset or fine-tune the parameters to fit your specific requirements.

## Evaluation

The model's performance was evaluated using various metrics, including accuracy, precision, recall, and F1-score, depending on the research problem's nature. The evaluation results are provided in the `cnn.py` python file or the Final Report.

## Results

The results of this project are outlined in the research paper's abstract, which served as the basis for this project. The CNN's performance and its contribution to achieving the research objectives are discussed in detail in the paper.

For more detailed information and to reproduce the results, please refer to the research paper linked in the abstract. You can find the research paper in the `Final Report.pdf` file in this repository.

## Contributors

This project was completed as part of the coursework for ECE324 - Machine Intelligence, Software & Neural Network. The author of this project is India Tory, Nida Copty, and Thomas Nguyen (myself). For questions or inquiries, please feel free to contact me at tomasdfgh.nguyen@mail.utoronto.ca.

## Acknowledgments

I would like to express my gratitude to Prof. Guerzhoy for their guidance and support throughout this project. Additionally, I thank my fellow classmates for their valuable insights and discussions, which contributed to the project's success.

---

This project's objective was to implement a Convolutional Neural Network (CNN) to address the research problem outlined in the abstract of the accompanying research paper. The CNN architecture, dataset, training process, and evaluation methods were all designed to align with the research objectives. For a more in-depth understanding of the project and its results, please refer to the research paper.

For access to the complete codebase and datasets, please explore this repository further.

