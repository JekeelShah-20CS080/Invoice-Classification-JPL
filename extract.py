# Using PyMUPDF to extract text from PDFs

import fitz
import os
import pandas as pd

# Path to folder containing  PDFs
pdf_folder = 'E:/Internship 2023/Invoices Dataset'

pdf_data = []  # List to store PDF text details

# Iterating through PDF files
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        
        with fitz.open(pdf_path) as doc:
            pdf_info = {'filename': filename, 'pages': []}  # Dictionary to store PDF info
            
            # Iterate through each page of the PDF
            for page in doc:
                # Extract text from the page
                text = page.get_text()
                
                # Add page text to PDF info
                pdf_info['pages'].append(text)
            
            # Add PDF info to the list
            pdf_data.append(pdf_info)

# Creating dataframe
df = pd.DataFrame(pdf_data)

# Print the DataFrame
print(df)

# saving df in csv format
df.to_csv('invoices.csv')