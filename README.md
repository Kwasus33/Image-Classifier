# Image-Classifier

## General Information

### Project:
* Main purpose of that project was to learn how Convolutional Neural Networks work and how to build custom one.
* We also evaluated existing and well-known solutions as residual networks family (ResNets) and ViTbase (based on transformers) in topic of transfer-learning
* Results and conclusions are stored in short report - "Projekt 5 - GOLEM CNN.pdf"

### Authors:
* Jakub Kwaśniak
* Jakub Mieczkowski

## Run and locally reproduce the results

### prepare environment

```shell
# Ubuntu / MacOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### run script / reproduce results
#### I option:

```shell
python3 main.py --optim 'SGD' --loss 'CrossEntropy' --backbone 'resnet18'
```
* optim - optimizer chosen from - "SGD", "RMS", "ADADELTA", "ADAMAX"
* loss - loss function - Cross Entropy Loss if flag is equal to "CrossEntropy", Mean Square Error otherwise
* backbone - model's backbone - can be chosen from pretrained ("ViT", "resnet18", "resnet34") or our custom one if other flag than one of shown is passed  

#### II option :
* Run notepad.ipynb prepared for models tests and evaluation (all results collected as plots) 
