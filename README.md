# mini-project-cnn
We are building a multi-class classification neural network aiming to classify tumor as benign (BEN), malignant (CAN) or normal (NOR).

## Dataset Structure
In order to `flow_from_directory` method of the `ImageDataGenerator` data should be structured this way
```
|- ddsm
  |- train
    |- BEN
    |- CAN
    |- NOR
  |- validation
    |- BEN
    |- CAN
    |- NOR
 ```  
 We then construct two BatchDataset for train and validation directories.
 
 ## Model architecture definition
 Convolutional Neural Network (CNN) is the suitable architecture for treating image datasets. Our choices are justified referring to [this article](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939) 
 ### How many convolutional layers do we want ?
 A CNN typically has three types of layers: a convolutional layer, a pooling layer, and a fully connected layer.
 
 We built a sequential model with layers from the Keras library. Our model is built of:
 * Convolution layers (3)
    * takes as parameters
      * input shape of data (only for the input layer)  
      * the number of kernels
      * the size of kernels
      * activation function
    * benefit
      * extract features out of the image using feature matrices
          
 * Pooling layers (3)
    * takes as parameters
      * size of the pooling matrix 
    * benefit:
      * downsamples the output of a convolutional layers by sliding the filter of some size with some stride size and calculating the maximum (as we used a MaxPooling layer) of the input 
 * Flatten layer (1)

Used to convert all the resultant 2-Dimensional arrays from pooled feature maps into a single long continuous linear vector so it later the data can be fed to a fully connected layer
 * Dropout layer (1)

Helps prevent overfitting by randomly setting input units to 0 with a frequency of rate at each step during training time
 * Dense layer (fully connected layer) (1)
Neurons in this layer have full connectivity with all neurons in the preceding and succeeding layer.

Helps to map the representation between the input and the output. It can be computed as usual by a matrix multiplication followed by a bias effect

 * Output layer (1)

Gives the actual output of the probability of an image belonging to each of the three classes.
 ### What should be the activation function of each layer ?
 Negative weights affects badly the neural network, ReLu function eliminates negative weights by setting them to zero so we use it for all layers except for the output layer.
 
 The output layer contains three nodes (one for each class). The softmax acivation function normalize the output of a network to a probability distribution over predicted output classes
 ### How many hidden units should each layer have ?
Data is less complex is having fewer dimensions or features then neural networks we use lesser units for hidden layers with 16 to 64 kernels to convolutions and 128 node for hidden layer. The output layer must contain 3 and only three nodes as we have three classes.
