#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import string
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
from nltk.corpus import stopwords


# In[ ]:


def extract_article_text(url) :
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find elements containing article text (e.g., <p> tags)
        article_text = ''
        for paragraph in soup.find_all('p'):
            article_text += paragraph.text.strip() + '\n'
        
        return article_text
    except Exception as e:
        print(f"Error extracting text from URL: {url}")
        print(e)
        return None


# In[ ]:


# Create a folder to store extracted text files
if not os.path.exists('Extracted_Text'):
    os.makedirs('Extracted_Text')


# In[ ]:


# Iterate through each row in the DataFrame
for index, row in input_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    text = extract_article_text(url)
    
    if text:
        # Save extracted text to a text file
        with open(f'Extracted_Text/{url_id}.txt', 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text extracted and saved for URL_ID: {url_id}")
    else:
        print(f"Failed to extract text for URL_ID: {url_id}")

print("Data extraction completed.")


# In[ ]:


#Text Analysis


# In[27]:


# Specify the folder name
folder_name = "Extracted_Text"

# Get the current directory
current_directory = os.getcwd()

# Construct the full path to the folder
folder_path = os.path.join(current_directory, folder_name)

# List all files in the folder
files_in_folder = os.listdir(folder_path)

# Loop through the files and read them
for file_name in files_in_folder:
    # Check if the file is a text file (you can change this condition as per your file type)
    if file_name.endswith('.txt'):
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the file with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            
        # Do something with the file content
        print(f"File '{file_name}' content:")
        print(file_content)


# In[2]:


def read_stopwords(folder_path):
    stopwords_sets = []

    # List of stop words files
    stopwords_files = [
        'StopWords_Auditor.txt',
        'StopWords_Currencies.txt',
        'StopWords_DatesandNumbers.txt',
        'StopWords_Generic.txt',
        'StopWords_GenericLong.txt',
        'StopWords_Geographic.txt',
        'StopWords_Names.txt'
    ]

    # Iterate through each stop words file
    for file_name in stopwords_files:
        stopwords_set = set()
        file_path = os.path.join(folder_path, file_name)
        # Read stop words from the file and add them to the set
        with open(file_path, 'r', encoding='latin-1') as file:
            stopwords_set.update(file.read().splitlines())
        stopwords_sets.append(stopwords_set)

    return stopwords_sets

# Define the folder path where the stop words files are stored
stopwords_folder = r'C:\Users\Sanan Husain\Downloads\StopWords'

# Call the read_stopwords function with the folder path
stopwords_sets = read_stopwords(stopwords_folder)

# Print the stopwords sets
for i, stopwords_set in enumerate(stopwords_sets, start=1):
    print(f"Stopwords (File {i}):", stopwords_set)


# In[3]:


def clean_text(text, stopwords):
    # Tokenize text into words
    words = text.split()
    # Remove stop words
    cleaned_words = [word for word in words if word.lower() not in stopwords]
    # Join cleaned words into a single string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text


# In[7]:


text = file_content
stopwords = set.union(*stopwords_sets)


# In[8]:


cleaned_text = clean_text(text, stopwords)
print("Cleaned text:", cleaned_text)


# In[9]:


def read_master_dictionary(folder_path):
    positive_words = set()
    negative_words = set()

    # Read positive words file
    with open(os.path.join(folder_path, 'positive-words.txt'), 'r') as file:
        positive_words.update(file.read().splitlines())

    # Read negative words file
    with open(os.path.join(folder_path, 'negative-words.txt'), 'r') as file:
        negative_words.update(file.read().splitlines())

    return positive_words, negative_words

# Define the folder path where the master dictionary files are stored
master_dict_folder = r'C:\Users\Sanan Husain\Downloads\MasterDictionary'

# Call the read_master_dictionary function with the folder path
positive_words, negative_words = read_master_dictionary(master_dict_folder)

# Test the result
print("Number of positive words:", len(positive_words))
print("Number of negative words:", len(negative_words))


# In[15]:


import os

# Define the folder path where the extracted text files are stored
folder_path = 'Extracted_Text'

# Initialize lists to store cleaned texts and sentiment scores
cleaned_texts = []
sentiment_scores = []

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    # Skip directories
    if os.path.isdir(os.path.join(folder_path, file_name)):
        continue
    
    # Read the file
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Clean the text for this file
    cleaned_text = clean_text(text, stopwords)
    print("Cleaned text for", file_name, ":", cleaned_text)
    
    # Compute sentiment score for this cleaned text
    # (Perform sentiment analysis here)


# In[18]:


text = cleaned_text

# Tokenize the text into words
words = word_tokenize(text)

# Count the number of words
total_words = len(words)

print("Total words in the text:", total_words)


# In[20]:


text = file_content

# Tokenize the text into words
words = word_tokenize(text)

# Count the number of words
total_words = len(words)

print("Total words in the text:", total_words)


# In[25]:


def calculate_readability_metrics(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    average_sentence_length = word_count / (sentence_count + 0.000001)
    complex_word_count = sum(1 for word in words if syllable_count(word) > 2)
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)
    return average_sentence_length, percentage_complex_words, fog_index


# In[30]:


def syllable_count(word):
    vowels = 'aeiouy'
    count = 0
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count += 1
    return count


# In[32]:


def calculate_avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    total_sentences = len(sentences)
    return total_words / total_sentences if total_sentences > 0 else 0


# In[34]:


def calculate_complex_word_count(text):
    words = word_tokenize(text)
    complex_word_count = sum(1 for word in words if syllable_count(word) > 2)
    return complex_word_count


# In[36]:


def calculate_word_count(text):
    words = word_tokenize(text)
    return len(words)


# In[38]:


def calculate_syllable_per_word(text):
    words = word_tokenize(text)
    total_syllables = sum(syllable_count(word) for word in words)
    return total_syllables / max(len(words), 1)  # Avoid division by zero


# In[40]:


def calculate_personal_pronouns(text):
    personal_pronouns = ["i", "we", "my", "ours", "us"]
    words = word_tokenize(text.lower())
    return sum(1 for word in words if word in personal_pronouns)


# In[42]:


def calculate_average_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    return total_characters / total_words if total_words > 0 else 0


# In[46]:


import os
import pandas as pd

# Initialize lists to store computed variables
results = []

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    # Skip directories
    if os.path.isdir(os.path.join(folder_path, file_name)):
        continue
    
    # Read the file
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Clean the text for this file
    cleaned_text = clean_text(text, stopwords)
    
    # Compute sentiment scores for this cleaned text
    positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentiment_scores(cleaned_text, positive_words, negative_words)
    
    # Compute other variables
    average_sentence_length, percentage_complex_words, fog_index = calculate_readability_metrics(cleaned_text)
    average_words_per_sentence = calculate_avg_words_per_sentence(cleaned_text)
    complex_word_count = calculate_complex_word_count(cleaned_text)
    total_words = calculate_word_count(cleaned_text)
    syllable_per_word = calculate_syllable_per_word(cleaned_text)
    personal_pronouns_count = calculate_personal_pronouns(cleaned_text)
    average_word_length = calculate_average_word_length(cleaned_text)
    
    # Store the computed variables in a dictionary
    result = {
        'File Name': file_name,
        'Positive Score': positive_score,
        'Negative Score': negative_score,
        'Polarity Score': polarity_score,
        'Subjectivity Score': subjectivity_score,
        'Average Sentence Length': average_sentence_length,
        'Percentage of Complex Words': percentage_complex_words,
        'Fog Index': fog_index,
        'Average Number of Words Per Sentence': average_words_per_sentence,
        'Complex Word Count': complex_word_count,
        'Word Count': total_words,
        'Syllable Per Word': syllable_per_word,
        'Personal Pronouns': personal_pronouns_count,
        'Average Word Length': average_word_length
    }
    
    # Append the result to the results list
    results.append(result)

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
output_file = 'text_analysis_results.csv'
results_df.to_csv(output_file, index=False)

print("Text analysis results saved to:", output_file)

#I saved the file in .Csv because I were facing some issue saving the Xlsx file then I converted it into xlsx

