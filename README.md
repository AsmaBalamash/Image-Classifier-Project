# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

**To run train file using the following command:**
Train a new network on a data set with train.py

**Basic usage:** 
 **Prints out training loss, validation loss, and validation accuracy as the network trains**
 **This will run the code with defult values *learning_rate* 0.003 *hidden_units* 512,256 *epochs* 5 *arch* densenet121**
- python train.py data_directory

**Options:**
***data_dir:* flowers, *arch:* can be densenet121 or vgg13**
***save_directory:* the directory that you want to save your file in (you can specifiy it)**
 - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
 - Choose architecture: python train.py data_dir --arch "vgg13"
 - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
 - Use GPU for training: python train.py data_dir --gpu

**To run predict file using the following command:**
 Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
 
 **Basic usage:**
 **Use this values for testing *image_path:* flowers/test/1/image_06743.jpg *checkpoint:* checkpoint.pth (the file saved from train.py)**
 - python predict.py /path/to/image checkpoint
 **Options:**
 ***input:* image path , *checkpoint:* (checkpoint.pth) the file saved from train.py along with its directory**
 - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
 - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
 - Use GPU for inference: python predict.py input checkpoint --gpu


