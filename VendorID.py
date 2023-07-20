import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Function to load and resize images from a folder
def load_and_resize_images(folder, size=(100, 100)):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        if img is not None:
            img = img.resize(size)
            images.append(np.array(img))
    return images

# Define the path to the Logo folder
logo_folder = 'Logo'

# Create empty lists to store the logo images and labels
logo_images = []
labels = []

# Load the logo images from the Logo folder and assign labels
for filename in os.listdir(logo_folder):
    if filename.endswith('.png'):
        vendor_name = os.path.splitext(filename)[0]
        if vendor_name == 'Flipkart_alt':
            vendor_name = 'Flipkart'
        logo_img = Image.open(os.path.join(logo_folder, filename)).convert('RGB')
        logo_images.append(logo_img)
        labels.append(vendor_name)

# Resize logo images to a consistent shape and stack them horizontally
logo_images_resized = [img.resize((100, 100)) for img in logo_images]
logo_images = np.hstack([np.array(img) for img in logo_images_resized])

# Load train and test images for each vendor
train_folder = 'Invoices Dataset/Train'
test_folder = 'Invoices Dataset/Test'
train_images = []
train_labels = []
test_images = []
test_labels = []

for i, vendor in enumerate(labels):
    vendor_train_folder = os.path.join(train_folder, vendor)
    train_images_vendor = load_and_resize_images(vendor_train_folder)
    train_images.extend(train_images_vendor)
    train_labels.extend([i] * len(train_images_vendor))

    vendor_test_folder = os.path.join(test_folder, vendor)
    test_images_vendor = load_and_resize_images(vendor_test_folder)
    test_images.extend(test_images_vendor)
    test_labels.extend([i] * len(test_images_vendor))

# Convert train and test images to numpy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)

# Preprocess the data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to one-hot encoding
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the deep learning model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')

model.save('')

# Now you can use the trained model for logo detection on new invoice images
# Load a new invoice image and preprocess it
new_invoice_path = 'Movie4 Debit.png'
new_invoice = Image.open(new_invoice_path).convert('RGB')
new_invoice = new_invoice.resize((100, 100))
new_invoice = np.array(new_invoice)
new_invoice = new_invoice.astype('float32') / 255.0
new_invoice = np.expand_dims(new_invoice, axis=0)

# Use the trained model to predict the vendor logo on the new invoice
prediction = model.predict(new_invoice)
predicted_label_index = np.argmax(prediction)
predicted_vendor = labels[predicted_label_index]

print(f'Predicted vendor: {predicted_vendor}')

model.save('vendor_detection_model.h5')