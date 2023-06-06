import pandas as pd
import numpy as np
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pywsd.utils import lemmatize_sentence

# nltk.download()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


wordnet_lemmatizer = WordNetLemmatizer()

df = pd.read_csv('invoices.csv')


# def remove_escape_sequences(text):
#     cleaned_text = ' '.join(text)
#     cleaned_text = re.sub(r'\\n', ' ', cleaned_text)
#     cleaned_text = re.sub(r'\\r', ' ', cleaned_text)
#     cleaned_text = re.sub(r'\\t', ' ', cleaned_text)
#     cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace consecutive whitespace characters with a single space
#     return cleaned_text

# # Apply the function to the pages column
# df['pages'] = df['pages'].apply(remove_escape_sequences)
print(df.head())
