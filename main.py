from bert import QA
import pandas as pd
import re
import csv
from datetime import datetime
from utils import *
time_str = datetime.now().strftime('%Y%m%d%H%M%S')

bank_filepath = 'itembank.csv'
model = QA('model')

# Load the database
content_code = 'A'
df = pd.read_csv(bank_filepath)

# Data cleanup
df = df.replace("[\[].*?[\]]", "",regex=True)           # remove all bracket stylings
df = df.replace("\(select.*?\)", "",regex=True)         # remove all select prompts
df = df.replace('\s+', ' ', regex=True)                 # truncate all multiple spaces

# Data filters
context = (df['ContentCode'] == content_code) \
    & (~df['Stem'].str.contains('arrow|points to|the line|line #|1. |video|label|cursor|click|\d{3}') # Attempt to filter out questions that refer to images
    & (df['FullKey'] > 'D')                             # Filter out questions with more than 4 options
    & (df['FullKey'].str.len() == 1))                   # Filter out questions with multiple answers, will not work with model prediction

# Inits
iterator=0
length = df[context].shape[0]
total = int((length * (length+1)) / 2)

# Create output csv for all possible cue results
with open(f'output_{time_str}.csv', 'w', newline='') as output:
    wr = csv.writer(output, quoting=csv.QUOTE_ALL)
    wr.writerow(['position', 'text', 'question', 'distA', 'distB', 'distC', 'distD', 'fullkey', 'prediction', 'start', 'end', 'confidence'])

# Iterate for each entry
for idx, text_row in df[context].iterrows():
    text = text_row['Stem'].lower()
    text = text.replace(':', '').replace('?', '')

    text_id = text_row['RecordID']
    text_answer = text_row['Dist' + text_row['FullKey']]

    # Iterate for each other entry with a higher index to find all question pairs
    for _, question_row in df[context].loc[idx+1:].iterrows():
        iterator = iterator + 1

        question_answer = question_row['Dist' + question_row['FullKey']]

        # If the answer to the question is not in the text, the model cannot find a cue
        if (question_answer not in text): continue 

        question_id = question_row['RecordID']
        question = question_row['Stem'].lower()
        question = question.replace(':', '?')

        # Assemble a block of text to replicate the trained environment
        build_text = BuildText(text, text_answer, length, df[context])

        # Predict the answer from the text
        prediction = model.predict(build_text,question)

        print(f'\rprogress: {iterator} / {total}', end='', flush=True)

        # Write confidence of cue to csv
        with open(f'output_{time_str}.csv', 'a', newline='') as output:
            wr = csv.writer(output, quoting=csv.QUOTE_ALL)
            wr.writerow([iterator, text, question, \
                question_row['DistA'], question_row['DistB'], question_row['DistC'], question_row['DistD'], question_row['FullKey'], \
                prediction['prediction'], prediction['start'], prediction['end'], prediction['confidence']])

        # If the model guessed the correct answer, output to the console
        if prediction['prediction'] != 'N/A' and prediction['prediction'] in \
            [question_row['DistA'], question_row['DistB'], question_row['DistC'], question_row['DistD']]:
            print()
            print(f'text id: {text_id}')
            print(f'text: {build_text}')
            print(f'question id: {question_id}')
            print(f'question: {question}')
            print(f'question answer: {question_answer}')
            for key, value in prediction.items():
                print(f'{key}: {value}')
            print()