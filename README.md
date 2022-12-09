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
