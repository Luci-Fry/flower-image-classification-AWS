"""
model_fucntions include fucntion to:
1. Load the data and transform the data and define dataloaders
2. Load in the pretrianed model based one user input
3. Test the network
4. Save the checkpoint"""

#Import statements 
import torch 
from torchvision import datasets, models
import argparse

###########################################################################

def data_loader(data_dir, transforms, batch_size, train_data= False):
    
    """ this function takes in the directory of the images, loads the dataset with the ImageFolder method from datasets and performs transforms on the data with the transforms defined by the user. Also allows the user to define the batch size they want to use. 
    
    Arguments:
    1. data_dir: the directory in which the image data is located 
    2. transforms: user defined transforms. Can be defined with the torch.Compose() method 
    
    """
    #if this is to load the training data, we must return the class to index mapping 
    if train_data:
        #load in the data with the datasets module 
        train_data = datasets.ImageFolder(data_dir, transform = transforms)
       
        class_to_idx= train_data.class_to_idx
    
        #define the dataloader, an iterator that lets out a batchsize of images at every iteration
        dataloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    
        return dataloader, class_to_idx
    else:
        #load in the data with the datasets module 
        data = datasets.ImageFolder(data_dir, transform = transforms)
    
        #define the dataloader, an iterator that lets out a batchsize of images at every iteration... we do not need to shuffle if it is the training data
        dataloader = torch.utils.data.DataLoader(data, batch_size = batch_size)
    
        return dataloader

##########################################################################    
    
def get_args():
    
    """This function gets input arguments of various forms from the user in the command line. Allows user to define:
    1. data_dir: path to data directory
    2. --arch: model architecture
    3. --learning_rate: learning rate for the model
    4. --epochs: number of epochs to train for
    5. --gpu : indicate wheater GPU should be used for training
    
    """
    parser = argparse.ArgumentParser(description = 'Gets arguments like the model architecture, number of epochs, etc from the user', prefix_chars = '-+/')
    parser.add_argument('data_dir',help= 'path to data directory')
    parser.add_argument('--arch', default = 'vgg19', help = 'define the architecture of the model you want')
    parser.add_argument('--learning_rate', default = '0.001', type = float, help = 'define the learning rate for the model')
    parser.add_argument('--epochs', default = '13', type= int, help ='define number of epochs to train for' )
    parser.add_argument('//gpu', '--gpu', action = 'store_true', help = 'indicate wheater GPU should be used for training')
    
    
    
    
    return parser.parse_args()

##########################################################################

def get_model():
    
    """ This function looks at the model the user has sepcified and loads the model accordingly. This prevents the loading of both models in one run of the program. 
    
    Arguments: None
    """
    
    input_args = get_args()
    
    if input_args.arch == 'vgg19':
        model = models.vgg19(pretrained = True)
        
    elif input_args.arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        
        
    return model 
    
    
    
    
    
    
    
    
    
    
    
    
    
    