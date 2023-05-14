# Imports here
import pandas as pd
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
import torch.optim as optim
import json
import argparse

# Create a parser object and tell it what arguments to expect
parser = argparse.ArgumentParser(description='This is the parser of the training script')
# Define the parser arguments
parser.add_argument('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument('-lr', '--learning_rate', help = "Learning rate", type = float, default=0.001)
parser.add_argument('-hidden_units', '--hidden_units', help = "Number of hidden units in the classifier", type = int, default=1024)
parser.add_argument('-epochs', '--epochs', help = "Number of epochs", type = int, default=4)
parser.add_argument('-GPU', '--GPU', help = "Option to use the GPU for training", default = False, action='store_true')
parser.add_argument('-arch', '--model_arch', help = "Model architecture to be used by the neural network. Default: resnet50", type = str, default = 'resnet50')
parser.add_argument('-save_dir', '--save_dir', help = "Set directory to save checkpoints", type = bool, default = False)
# Run the parser and place the extracted data in a argparse.Namespace object
args = parser.parse_args()

# Check whether the CPU or the GPU will be used for training
if (args.GPU == True) & torch.cuda.is_available():
    device = "cuda"
elif(args.GPU == False) & (torch.cuda.is_available() == False):
    device = "cpu"
else:
    raise Exception("Please check your selection of GPU availability and GPU parsed argument")

# Load the data
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
# Define your transforms for the training, validation, and testing sets
training_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

validation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

testing_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
training_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
testing_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms)

# Using the image datasets and the transforms, define the dataloaders
training_loader = torch.utils.data.DataLoader(training_datasets, batch_size=128, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=32, shuffle=True)
testing_loader = torch.utils.data.DataLoader(testing_datasets, batch_size=32, shuffle=True)

# Import the label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def select_model_architecture(args, cat_to_name):
    if args.model_arch == "resnet50":
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(model.fc.in_features, args.hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.1)),
                                  ('fc2', nn.Linear(args.hidden_units, len(cat_to_name))),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        model.fc = classifier

    elif args.model_arch == "vgg13":
        model = models.vgg13(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(model.classifier[0].in_features, args.hidden_units)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(p = 0.3)),
                                ('fc2', nn.Linear(args.hidden_units, len(cat_to_name))),
                                ('output', nn.LogSoftmax(dim =1))
                                ]))
        model.classifier = classifier
    else:
        raise Exception("Please check the model architecture that you have selected for the neural network. You can only choose either resnet50 or vgg13")
    return model

# Build and train the network
model = select_model_architecture(args, cat_to_name)

# Train the model
model.to(device)

criterion = nn.NLLLoss()
if args.model_arch == "resnet50":
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
elif args.model_arch == "vgg13":
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
steps = 0  # Initialise a step number in order to keep track of how many batches have been processed
print_every = 10   # During each epoch, we will be printing the results every certain number of steps

for e in range(epochs):
    running_loss = 0
    # Train the model
    for images, labels in training_loader:
        
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
    
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Test the model
            valid_loss = 0
            accuracy = 0
            
            model.eval()  # Set the evaluation mode, so that there are no dropouts

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in validation_loader:

                    images, labels = images.to(device), labels.to(device)

                    log_ps = model(images)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
            print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/print_every),
            "Validation Loss: {:.3f}.. ".format(valid_loss/len(validation_loader)),
            "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
        
        model.train()   # Set the training mode, so that there are dropout is switched back on

# Save the checkpoint
if args.save_dir == True:
    if args.model_arch == "resnet50":
        checkpoint = {'arch':args.model_arch,
                  'epochs':args.epochs,
                  'lr':args.learning_rate,
                  'hidden_units':args.hidden_units,
                  'model_class_to_index':training_datasets.class_to_idx,
                  'fc':model.fc,
                  'state_dict':model.state_dict(),
                  'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, 'checkpoint-resnet50.pth')
    elif args.model_arch == "vgg13":
        checkpoint = {'arch':args.model_arch,
                  'epochs':args.epochs,
                  'lr':args.learning_rate,
                  'hidden_units':args.hidden_units,
                  'model_class_to_index':training_datasets.class_to_idx,
                  'Classifier':model.classifier,
                  'state_dict':model.state_dict(),
                  'optimizer':optimizer.state_dict()}

        torch.save(checkpoint, 'checkpoint-vgg13.pth')
    
print("FINISHED!")