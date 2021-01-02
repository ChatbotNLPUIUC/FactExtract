import spacy
import glob
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

# load english language model
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])

text = "The children love cream biscuits"

# create spacy 
doc = nlp(text)

#for token in doc:
    #print(token.text,'->',token.pos_)

for token in doc:
    # extract subject
    if (token.dep_=='nsubj'):
        print(token.text)
    # extract object
    elif (token.dep_=='dobj'):
        print(token.text)

folders = glob.glob('../DAQIANNtest/Converted sessions/Session*')
df = pd.DataFrame(columns={'Country','Speech','Session','Year'})
# Read speeches by India
i = 0 
for f in folders:  
    speech = glob.glob(f + '/IND*.txt')
    print(i)
    with open(speech[0],encoding='utf8') as fe:
        # Speech
        df.loc[i,'Speech'] = fe.read()
        # Year 
        df.loc[i,'Year'] = speech[0].split('_')[-1].split('.')[0]
        # Session
        df.loc[i,'Session'] = speech[0].split('_')[-2]
        # Country
        df.loc[i,'Country'] = speech[0].split('_')[0].split("\\")[-1]
        # Increment counter
        i += 1 
    
'''
No lemmatization or stemming to not change the POS tag
'''
def clean(text):
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    return text

# preprocessing speeches
df['Speech_clean'] = df['Speech'].apply(clean)
print(df.head())
