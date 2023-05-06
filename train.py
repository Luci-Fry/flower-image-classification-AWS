"""
This file is used to train an image classifier for the prediction of flower images. It takes in the kind of pretrained model the user would like and builds an image classification model and trains the model. 

It then outputs the traning and validation loss as well as the validation accuracy and saves the trained model to a checkpoint. 

It employs the argparse module to obtain input from the user. 

"""

# Import statements 
import torch 
from torch import nn
from torch import optim 
from torchvision import models, transforms
import argparse 
from collections import OrderedDict 


from model_functions import *
from workspace_utils import active_session

#define transforms for transforming the data
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(), transforms.RandomRotation(30), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                  'testval': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}

# Get the input arguments from the user 
input_args = get_args()

#load in the data with the data_loader function from model_functions file

train_set = data_loader(input_args.data_dir+'train', data_transforms['train'], 64, train_data = True)

train_loader = train_set[0]

class_to_idx = train_set[1]


valid_loader = data_loader(input_args.data_dir+'valid', data_transforms['testval'], 64)

##############################################################################
# #check if the data_loader function worked 
# for i in range(1):
#     image = next(iter(train_loader))
    
#     print(image)
    
#     break
    
    
###########################################################################
#load in the VVG19 pretrained model
#allow the user to choose between two models vgg19 and densenet121

model = get_model()
#print(model)



####################################################################

#freeze the model's classifier and define your own 

for param in model.parameters():
    param.requires_grad = False 
    
#now we can define our classifier with relu activation and dropout 

#get the number of input features 
in_features = {'vgg19':25088, 'densenet121':1024}


myclassifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(in_features[input_args.arch], 2000)),
                                          ('relu', nn.ReLU()),
                                          ('Drop1', nn.Dropout(p= 0.2)),
                                          ('fc2',nn.Linear(2000, 500)),
                                          ('Drop2', nn.Dropout(p= 0.1)),
                                         ('relu', nn.ReLU()),
                                          ('fc3',nn.Linear(500, 102)),
                                         ('output', nn.LogSoftmax(dim= 1)) 
                                         ]))

model.classifier = myclassifier 

#############################################################################

#define criterion 
criterion = nn.NLLLoss()

#define optimization with the SGD optimization

optimizer = optim.Adam(model.classifier.parameters(), lr = input_args.learning_rate)

#If the user wants to use the GPU, check to see if the GPU is available 
#if the GPU is not available, tell the user and use CPU

if input_args.gpu:
    if torch.cuda.is_available():
        print('GPU is available... Using GPU')
        interface = torch.device('cuda')
        
    else: 
        print('GPU is not available... Turn on if possible... Using CPU')
        interface = torch.device('cpu')
#print(input_args.gpu)

print('Training In Progress...')

###############################################################################

#MODEL TRAINING AND VALIDATION

with active_session(): #keep the session active 
    
    epochs = input_args.epochs
    model.cuda() #move the model to the GPU if GPU is available
    
    for e in range(epochs):
        running_loss = 0 #track the running loss of the training
        
        
        for images, labels in train_loader:
            
            images = images.cuda() 
            labels = labels.cuda() # move the images and labels to the GPU too
            
            #zero the gradients first 
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            
            #perform backward pass and optimize
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            
        #now for the validation 
        
        else:
            valid_loss = 0  #define validation loss variable to store the loss at each epoch during validation
            accuracy = 0
            
            with torch.no_grad(): #turn of the gradients, this is not training 
                model.eval() #put the model in evaluation mode
                
                for images, labels in valid_loader:
                    #send the validation data to the GPU too... the model is already on the GPU
                    
                    images, labels = images.to(interface), labels.to(interface)
                    
                    #do the forward pass.. since the model is in evaluation mode, the gradients are already off
                    logps = model.forward(images)
                    
                    #calculate the loss for the batch
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    #calculate the accuracy 
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim= 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                #print the train loss, validation loss and accuracy for each epoch
                
                print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(train_loader):.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                
                model.train() #put the model back in training mode for the next epoch

            
#########################################################################################
# SAVE THE MODEL TO A CHECKPOINT 


checkpoint = {'input_size': in_features[input_args.arch],
              'output_size': 102,
              'hidden_layers': [each.out_features for each in [model.classifier.fc1, model.classifier.fc2, model.classifier.fc3]],
              'state_dict': model.state_dict(),
             'epochs':16, 
             'optimizer':optimizer.state_dict(),
              'cat_to_name_mapping': class_to_idx
             }

torch.save(checkpoint, 'checkpoint'+input_args.arch+'.pth')



        