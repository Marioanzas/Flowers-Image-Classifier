# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I have first developed code for an image classifier built with PyTorch, then converted it into a command line application.

Sample command line for launching the `train.py` script:
```
python train.py flowers cat_to_name.json -GPU
```

Likewise, a sample command line for launching the `predict.py` script:
```
python predict.py flowers checkpoint-resnet50.pth cat_to_name.json -GPU -top_k 5

```
For the meaning of each parameter passed in the command line, please check the `predict.py` script.