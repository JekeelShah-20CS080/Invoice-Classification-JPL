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
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
print("=====================================================================")
print("Before Preprocessing: ")
df = pd.read_csv('invoices.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df.rename(columns={"pages": "data"}, inplace=True)
print(df.head())

print("\n=====================================================================")

print("After removing unnecessary symbols and escape sequences: ")
df['data'] = df['data'].str.lower()

# Remove unnecessary blank spaces
df['data'] = df['data'].str.strip()

df['data'] = df['data'].str.lstrip()
df['data'] = df['data'].apply(lambda x: ' '.join(x.split()))

df['data'] = df['data'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Replace multiple consecutive spaces with a single space
df['data'] = df['data'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Display the updated dataframe
print(df.head())
print("\n================================================================")

print("After Tokenization: ")
df['tokens'] = df['data'].apply(lambda x: word_tokenize(x))
print(df.head())

print("\n================================================================")
print('After Removing Stopwords: ')
stop_words = set(stopwords.words('english'))

df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
print(df.head())

print("\n================================================================")
print('After Lemmatization: ')
lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
print(df.head())

df.to_csv('cleaned_invoices.csv')