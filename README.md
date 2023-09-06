# Simple_classification_Network

This is a simple classification network based on the GoogLeNet architecture.

The data is organized as follows: there are two folders for storing images, "train" and "val," containing image files labeled with numerical identifiers.
For the corresponding image data in the "train" and "val" folders, two CSV files are required: "train.csv" and "val.csv." 
These CSV files should have the first column containing the image identifier and the second column containing the corresponding class number for each image.
(First row have to be names / label)
For testing, the data is contained in a folder that stores images.

To use this system, you can train it by running the following command: **python train.py**
This command will execute the training process for your classification network.

For testing, you can run the following command: **python test.py**
This command will execute the testing process using your trained model. Make sure to have the necessary data and model weights ready before running these commands.

You can reference the provided environment.yml file for installing the required environment for this project. 
To create a conda environment from an environment.yml file, you can use the following command in your terminal: **conda env create -f environment.yml**
