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

#for token in doc:
    # extract subject
    #if (token.dep_=='nsubj'):
        #print(token.text)
    # extract object
    #elif (token.dep_=='dobj'):
        #print(token.text)

folders = glob.glob('../DAQIANNtest/Converted sessions/Session*')
df = pd.DataFrame(columns={'Country','Speech','Session','Year'})
# Read speeches by India
i = 0
for f in folders:  
    speech = glob.glob(f + '/IND*.txt')
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
        i +=1 
    
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

# split sentences
def sentences(text):
    # split sentences and questions
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent

# sentences
df['sent'] = df['Speech_clean'].apply(sentences)

# Create a dataframe containing sentences
df2 = pd.DataFrame(columns=['Sent','Year','Len'])

# List of sentences for new df
row_list = []

# for-loop to go over the df speeches
for i in range(len(df)):
    
    # for-loop to go over the sentences in the speech
    for sent in df.loc[i,'sent']:
        
        wordcount = len(sent.split())  # Word count
        year = df.loc[i,'Year']  # Year
        dict1 = {'Year':year,'Sent':sent,'Len':wordcount}  # Dictionary
        row_list.append(dict1)  # Append dictionary to list
    
# Create the new df
df2 = pd.DataFrame(row_list)
#print(df2.head())

'''
Start of information extraction
'''

import spacy
from spacy.matcher import Matcher 

from spacy import displacy 
import visualise_spacy_tree
from IPython.display import Image, display

# Function to find sentences containing PMs of India
# Prime Minister specific
def find_names(text):
    
    names = []
    
    # Create a spacy doc
    doc = nlp(text)
    
    # Define the pattern
    pattern = [{'LOWER':'prime'},
              {'LOWER':'minister'},
              {'POS':'ADP','OP':'?'},
              {'POS':'PROPN'}]
                
    # Matcher class object 
    matcher = Matcher(nlp.vocab) 
    matcher.add("names", None, pattern) 

    matches = matcher(doc)

    # Finding patterns in the text
    for i in range(0,len(matches)):
        
        # match: id, start, end
        token = doc[matches[i][1]:matches[i][2]]
        # append token to list
        names.append(str(token))
    
    # Only keep sentences containing Indian PMs
    for name in names:
        if (name.split()[2] == 'of') and (name.split()[3] != "India"):
                names.remove(name)
            
    return names

# Apply function
df2['PM_Names'] = df2['Sent'].apply(find_names)
