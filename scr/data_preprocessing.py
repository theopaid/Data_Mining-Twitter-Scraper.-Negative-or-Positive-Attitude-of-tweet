from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import string
import pandas as pd
import csv
import re

def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

ps = PorterStemmer()
train_data = pd.read_csv('train2017.tsv', sep='\t', header=None, names=['id1', 'id2', 'feeling', 'text'], engine='python', quoting=csv.QUOTE_NONE, encoding='UTF-8')
print(train_data['text'][9937])
train_data['punct_text'] = train_data['text'].apply(lambda x: remove_punct(x))

train_data['tokenized_text'] = train_data['punct_text'].apply(word_tokenize)
stopwords = set(stopwords.words('english'))
count = -1
filtered_sentence = []
for currentLine in train_data['tokenized_text']:
    filtered_sentence = []
    count = count + 1
    for word in currentLine:
        if word not in stopwords:
            word = ps.stem(word)
            filtered_sentence.append(word)
    train_data['tokenized_text'][count] = filtered_sentence

print(train_data)


