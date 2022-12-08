# mini-project-cnn
academic project

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
