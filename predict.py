#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 04:59:57 2019

@author: mishz
"""

import seaborn as sns

from math import floor
from PIL import Image
import train.py


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    img= process_image(image)
    
    img = img.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    
    return ax


def process_image(image):
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
    
    
    # TODO: Process a PIL image for use in a PyTorch model

img = (data_dir + '/test' + '/1/' + 'image_06752.jpg')
img = process_image(img)
print(img.shape)


def predict(image_path, model, topk=5):   
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)


def check_sanity(image_path, model):
    
    #fig, ax = plt.subplots()
    plt.figure(figsize = (6,10))


    probs, classes = predict(image_path, model)
    img = process_image (image_path)
    imshow(img)
    plt.show()
    
    labels = list(cat_to_name.values())
    labels = np.array(labels)
    
    class_names = []
    
    for x in classes:
        print(x)
        c= labels[x]
        class_names= c


        
    plt.subplot(2,1,1)
    sns.barplot(x=probs, y=class_names, color= 'red')
    plt.show()

