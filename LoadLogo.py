import os
from PIL import Image

# Define the path to the Logo folder
logo_folder = 'E:/Invoice Classification JPL 2023/Logo'

# Create empty lists to store the logo images and labels
logo_images = []
labels = []

# Load the logo images from the Logo folder and assign labels
for filename in os.listdir(logo_folder):
    if filename.endswith('.png'):
        vendor_name = os.path.splitext(filename)[0]
        if vendor_name == 'Flipkart_alt':
            vendor_name = 'Flipkart'
        logo_img = Image.open(os.path.join(logo_folder, filename))
        logo_images.append(logo_img)
        labels.append(vendor_name)

# Print the number of logo images loaded
print(f"Number of logo images: {len(logo_images)}")
print(f"Number of labels: {len(labels)}")