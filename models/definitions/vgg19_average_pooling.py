from collections import namedtuple
import torch
from torchvision import models
import torch.nn as nn

class VGG_avg_pool(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()
        
        # Charger VGG19 pré-entraîné
        vgg_pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features

        # Remplacer MaxPooling par AvgPooling
        for i, layer in enumerate(vgg_pretrained_features):
            if isinstance(layer, nn.MaxPool2d):
                vgg_pretrained_features[i] = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)

        self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
        self.offset = 1
        self.content_feature_maps_index = 4  # conv4_2 pour le contenu
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  
        self.style_feature_maps_indices.remove(4)  # Toutes les autres couches pour le style

        # Diviser VGG19 en slices pour extraire les activations intermédiaires
        self.slice1 = nn.Sequential(*vgg_pretrained_features[:1+self.offset])
        self.slice2 = nn.Sequential(*vgg_pretrained_features[1+self.offset:6+self.offset])
        self.slice3 = nn.Sequential(*vgg_pretrained_features[6+self.offset:11+self.offset])
        self.slice4 = nn.Sequential(*vgg_pretrained_features[11+self.offset:20+self.offset])
        self.slice5 = nn.Sequential(*vgg_pretrained_features[20+self.offset:22])
        self.slice6 = nn.Sequential(*vgg_pretrained_features[22:29+self.offset])

        # Geler les poids si on ne veut pas entraîner le réseau
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False    

    def forward(self, x):
        ''' Passe une image à travers les différents slices du réseau. '''
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x

        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        return vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
