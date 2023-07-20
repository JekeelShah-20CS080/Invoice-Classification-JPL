from keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('logo_detection_model.h5')

# Function to perform logo detection on a single image
def detect_logo(image_path):
    # Load the image and preprocess it for the model
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform logo detection using the model
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions)

    # Convert the predicted label back to the vendor name
    label_to_vendor = {v: k for k, v in vendor_to_label.items()}
    predicted_vendor = label_to_vendor[predicted_label]

    return predicted_vendor

# Test the logo detection function on a single image
image_path = 'path/to/your/invoice/image.png'
predicted_vendor = detect_logo(image_path)
print(f"Predicted Vendor: {predicted_vendor}")