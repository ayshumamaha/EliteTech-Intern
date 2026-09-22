Task 3: Neural Style Transfer
Implements the classic Gatys et al. neural style transfer technique.

How it works
Loads a content image and a style image.
Extracts feature maps from specific layers of a pretrained VGG19 network.
Computes content loss (difference in feature activations) and style loss (difference in Gram matrices).
Optimizes a target image (initialized as a copy of the content image) to minimize a weighted combination of both losses.
Displays the final stylized image using Matplotlib.

Requirements
bash
pip install torch torchvision matplotlib pillow

Usage
Place your content image as content.jpg and style image as style.jpg in the same directory as the script.
Run the script:
bash
python Task_3_Neural_Style_Transfer.py
The script runs 200 optimization iterations, printing the loss every 50 steps, and displays the resulting stylized image at the end.

Key parameters
style_weight / content_weight — control the balance between preserving content and applying style.
style_layers / content_layers — VGG19 layers used to extract features.
Runs on GPU automatically if available (cuda), otherwise falls back to CPU.

Author
M. Ayshwarya
