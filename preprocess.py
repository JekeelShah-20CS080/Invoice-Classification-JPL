from pdf2image import convert_from_path
import os

# Directory path containing the PDF files
pdf_directory = "E:/Invoice Classification JPL 2023/Invoices Dataset"

# Get a list of all PDF files in the directory
pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]

# Output directory path for saving the converted images
output_directory = "E:/Invoice Classification JPL 2023/Converted Invoices"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Convert each PDF file to images
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    # Save each image to the output directory
    for i, image in enumerate(images):
        image_path = os.path.join(output_directory, f"{pdf_file}_{i+1}.jpg")
        image.save(image_path, "JPEG")
        print(f"Conversion complete for: {pdf_path}")

print("All PDF files have been converted and saved as images.")
