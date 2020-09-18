# resnet18_model
 
This project is done with Yu-Chieh Chen.

* This project is about exploring the resnet18 model. We'd trained two models: ResNet18 utilize the resnet18 structure without pretrained parameters and ResNet18_tl utilizes the resnet18 structure with pretrained parameters for transfer learning. We'd added additional layers on both model so that they will be suited for making prediction on Cifar10 dataset.

* The notebook is the detailed version of our work. It shows the entire process from preprocessing the data to building model and finally to evaluation.
* The train.py file is a simplified version of our work. We provide the function for training the model but without plotting and evaluation.
* util.py includes some important functions we'd used for our work.
* model.py includes the model we'd built. 
