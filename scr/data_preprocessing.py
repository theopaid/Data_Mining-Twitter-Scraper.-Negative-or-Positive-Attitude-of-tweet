from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pandas as pd
import csv
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

TokensForPositive = []
TokensForNegative = []
TokensForNeutral = []

for index, row in train_data.iterrows():
    if row['feeling'] == 'positive':
        TokensForPositive.append(row['tokenized_text'])
    if row['feeling'] == 'negative':
        TokensForNegative.append(row['tokenized_text'])
    if row['feeling'] == 'neutral':
        TokensForNeutral.append(row['tokenized_text'])

TokensForPositiveSTR = ' '.join(word for line in TokensForPositive for word in line)
TokensForNegativeSTR = ' '.join(word for line in TokensForNegative for word in line)
TokensForNeutralSTR = ' '.join(word for line in TokensForNeutral for word in line)

wordcloud = WordCloud(max_font_size=40).generate(train_data['tokenized_text'].to_string())
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("everything.png")

wordcloud = WordCloud(max_font_size=40).generate(TokensForPositiveSTR)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("positive.png")

wordcloud = WordCloud(max_font_size=40).generate(TokensForNegativeSTR)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("negative.png")

wordcloud = WordCloud(max_font_size=40).generate(TokensForNeutralSTR)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("neutral.png")




