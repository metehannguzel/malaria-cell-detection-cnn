## 1.	Introduction:

  The objective of this project is to develop an artificial intelligence model that can classify cells as infected or uninfected based on microscopic images. The clinical problem being addressed is the identification of malaria-infected cells, which can be time-consuming and prone to human error. Artificial intelligence can help automate this process and improve accuracy.


## 2.	Dataset:

  The dataset used in this project consists of microscopic images of cells, specifically infected and uninfected cells. The dataset size is 27,558 images, with each image having dimensions of 50x50 pixels and RGB color channels. The dataset can be accessed via the [link](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

## 3.	Method:


  Model Architecture: The model used in this project is a custom-designed convolutional neural network (CNN). It consists of three convolutional layers with 16, 32, and 64 filters, respectively, followed by max-pooling layers. The output from the last convolutional layer is flattened and passed through two fully connected (dense) layers with 500 and 1 unit(s), respectively. ReLU activation function is used in the convolutional and dense layers, except for the output layer which uses a sigmoid activation function.

  Model Training Strategy: The model is trained using supervised learning, where the input images are labeled with the corresponding class (infected or uninfected). This choice is appropriate because the dataset contains labeled examples for training.

  Loss Function and Optimizer: The loss function used is binary cross-entropy, which is suitable for binary classification problems. The optimizer used is Adam, which is a popular choice for training neural networks due to its adaptive learning rate.


## 4.	Experiments:

  Pre-processing Stage: The dataset undergoes pre-processing techniques before training the model. The images are resized to 50x50 pixels and converted to numpy arrays. No other pre-processing techniques are mentioned in the code.

  Augmentation: Data augmentation techniques are employed to increase the diversity of the training data. The techniques used include random rotation (up to 30 degrees), zooming (up to 20%), horizontal shifting (10% of width), and horizontal flipping. These techniques help prevent overfitting and improve generalization.

  Dataset Usage: The dataset is split into training, evaluation, and test sets. The training set comprises 80% of the data, while the evaluation and test sets each contain 10%. The data splitting is performed using the train_test_split function from the scikit-learn library.

  Weight Initialization: The weights of the model are not explicitly initialized in the code provided. The default weight initialization scheme of the Keras library is likely used.

  Dataset Size: The dataset size of 27,558 images is considered sufficient to develop a deep learning model, especially when using data augmentation techniques.

  Learning Curve Analysis: The learning curve shows the performance of the model during training in terms of accuracy and loss on both the training and validation sets. It demonstrates that the model's accuracy improves over epochs, while the loss decreases.

  Overfitting or Underfitting: The learning curve analysis suggests that there is no significant overfitting or underfitting observed. The validation accuracy and loss follow a similar trend to the training accuracy and loss, indicating that the model generalizes well to unseen data.

  Hyperparameters: The hyperparameters used in the model include the number of filters in the convolutional layers (16, 32, and 64), the size of the convolutional filters (2x2), dropout rates (0.2), and the number of units in the dense layers (500, 1).

  Model Results on the Test Set: The model achieves a test loss of 0.16 and a test accuracy of 94.45%. These metrics indicate that the model performs well in classifying infected and uninfected.
