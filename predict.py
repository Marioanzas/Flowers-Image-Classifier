import json
import torch
import argparse
from torchvision import transforms, models
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Create a parser object and tell it what arguments to expect
parser = argparse.ArgumentParser(description='This is the parser of the predicting script')
# Define the parser arguments
parser.add_argument('image_dir', help = 'Provide the image data to be predicted directory. Mandatory argument', type = str)
parser.add_argument('checkpoint_name', help = 'Provide the name of the checkpoint to be loaded. Mandatory argument', type = str)
parser.add_argument('json_category_mapping_dir', help = 'Provide the path to the JSON that contains the mapping of the categories. Mandatory argument', type = str)
parser.add_argument('-GPU', '--GPU', help = "Option to use the GPU for training", default = False, action='store_true')
parser.add_argument('-top_k', '--top_k', help = "Option to use select the number of top K classes to be printed", type = int, default = 5)

# Run the parser and place the extracted data in a argparse.Namespace object
args = parser.parse_args()

# Import the label mapping
with open(args.json_category_mapping_dir, 'r') as f:
    cat_to_name = json.load(f)
    
# Write a function that loads a checkpoint and rebuilds the model
def load_saved_model(filepath):
    # Load the checkpoint data to the best currently available location (e.g. model saved with GPU but loaded with CPU)
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)  # load the saved model checkpoint from the specified 'filepath'
    # Now load the pre-trained models and substitute their default classifiers
    if checkpoint["arch"] == "resnet50":
        model = models.resnet50(pretrained=True)   # create a new resnet50 model instance with the pre-trained weights
        model.fc = checkpoint["fc"]  # replace the last fully connected layer of the pre-trained model with the one from the saved checkpoint
    elif checkpoint["arch"] == "vgg13":
        model = models.vgg13(pretrained=True)   # create a new resnet50 model instance with the pre-trained weights
        model.classifier = checkpoint["Classifier"]  # replace the last fully connected layer of the pre-trained model with the one from the saved checkpoint
    
    # Load the rest of the attributes to the model
    model.load_state_dict(checkpoint["state_dict"])  # Load the state dictionary of the saved model checkpoint 
    model.class_to_idx = checkpoint["model_class_to_index"]
    
    return model

# Load the model from the checkpoint
model = load_saved_model(args.checkpoint_name)

# Check whether the CPU or the GPU will be used for training
if (args.GPU == True) & torch.cuda.is_available():
    device = "cuda"
elif(args.GPU == False) & (torch.cuda.is_available() == False):
    device = "cpu"
else:
    raise Exception("Please check your selection of GPU availability and GPU parsed argument")
    
# Image Preprocessing
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    img = PIL.Image.open(image)
    img = img.convert("RGB")
    img = transform(img)
    return img

# Class prediction
def predict(image_path, model, topk):
    model.eval()  # Set the evaluation mode, so that there are no dropouts

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        image = process_image(image_path)
        image = image.float().unsqueeze_(0)

        image = image.to(device)
        model = model.to(device)

        log_ps = model(image)

        ps = torch.exp(log_ps)
        top_p, indices = ps.topk(topk, dim=1)
        top_1, top_index = ps.topk(1, dim=1)

        idx_to_class = {v: k for k, v in model.class_to_idx.items()}  # Invert the dictionary
        top_class = idx_to_class[top_index.item()]  # Find the respective class for each the top_index
        top_classes = [idx_to_class[x] for x in indices.tolist()[0]]  # Find the respective class for each of the indices
        top_flower_names = [cat_to_name[str(i)] for i in top_classes]   # Get the top flower names from the top_classes
        return top_p, top_classes, top_1, top_class, top_flower_names
    
top_p, top_classes, top_1, top_class, top_flower_names = predict(args.image_dir, model, args.top_k)
print("The most likely image class is", cat_to_name[top_class], "and its associated probability is", round(top_1.item(),4))
print("The top", args.top_k, "classes are", top_classes)
print("The top", args.top_k, "flower names are", top_flower_names)
print("The top", args.top_k, "associated probabilities are", top_p)