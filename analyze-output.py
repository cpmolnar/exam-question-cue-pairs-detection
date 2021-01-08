import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

cachedStopWords = stopwords.words("english")
filename = 'filename'

clear = lambda: os.system('cls')

df = pd.read_csv(filename, encoding='ANSI').fillna('N/A')
context = (df['prediction'] != 'N/A')
positives = []

def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return text

i=0
length = df[context].shape[0]
for _, row in df[context].iterrows():
    i=i+1
    text = remove_stopwords(row['text'])
    question = remove_stopwords(row['question'])
    answer = ''
    for key in row['fullkey']:
        answer += row[ord(key) - 62] + ' and '
    answer = re.sub('\ and ', '', answer)
    if (answer in text):
        positives.append(row.tolist())
        print(f'\rprogress: {i} / {length}', end='', flush=True)

positives = sorted(positives, key=lambda x: x[11], reverse=True)

for row in positives:
    position = row[0]
    text = row[1]
    question = row[2]
    distA = row[3]
    distB = row[4]
    distC = row[5]
    distD = row[6]
    fullkey = row[7]
    prediction = row[8]
    start = row[9]
    end = row[10]
    confidence = row[11]

    answer = ''
    for key in fullkey:
        answer += row[ord(key) - 62] + ' and '
    answer = re.sub('\ and ', '', answer)

    clear()
    print(f'position: {position}')
    print(f'text: {text}')
    print(f'question: {question}')
    print(f'choice A: {distA}')
    print(f'choice B: {distB}')
    print(f'choice C: {distC}')
    print(f'choice D: {distD}')
    print(f'answer: {answer}')
    print(f'prediction: {prediction}')
    print(f'confidence: {confidence}')
    input("Press Enter to continue...")