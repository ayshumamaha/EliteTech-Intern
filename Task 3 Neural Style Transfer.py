# Neural Style Transfer using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

# Display image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return image

# Load content and style images
content = load_image("content.jpg")
style = load_image("style.jpg")

# Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze model parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Layers for style and content representation
style_layers = ['0', '5', '10', '19', '28']
content_layers = ['21']

def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get features
content_features = get_features(content, vgg, content_layers)
style_features = get_features(style, vgg, style_layers)

# Calculate Gram matrices for style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image
target = content.clone().requires_grad_(True).to(device)

# Style transfer parameters
style_weight = 1e6
content_weight = 1
optimizer = optim.Adam([target], lr=0.003)

# Training loop
for i in range(1, 201):
    target_features = get_features(target, vgg, style_layers + content_layers)
    content_loss = torch.mean((target_features['21'] - content_features['21']) ** 2)

    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_loss = torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_loss / (target_feature.shape[1] ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Iteration {i}, Total loss: {total_loss.item()}")

# Display result
final_img = im_convert(target)
plt.imshow(final_img)
plt.axis("off")
plt.title("Stylized Image")
plt.show()
