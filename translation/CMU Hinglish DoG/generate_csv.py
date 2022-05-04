import glob
import json
import pandas as pd

suffix_len = len('.json_1') # only take a single translation
hinglish_sentences = []
english_sentences = []

type = 'train' # test, valid

for filename in glob.glob('/home/ubuntu/CMUHinglishDoG/Conversations_Hinglish/' + type +
                          '/*.json_1'): # Hinglish
    id = (filename.split('/'))[-1][:-suffix_len]
    english_filename = '/home/ubuntu/CMUHinglishDoG/Conversations/train/' + id + '.json' # English
    with open(filename, 'r') as hing, open(english_filename, 'r') as eng:
        hinglish_dict = json.load(hing)
        english_dict = json.load(eng)
        for hinglish_his, english_his in zip(hinglish_dict['history'], english_dict['history']):
            hinglish_sentences.append(hinglish_his['text'])
            english_sentences.append(english_his['text'])

df = pd.DataFrame()
df['Hinglish'] = hinglish_sentences
df['English'] = english_sentences
df.to_csv(type + '_hinglish_english.csv', index=False)