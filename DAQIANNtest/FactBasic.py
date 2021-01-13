import spacy
import glob
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from Rules import rule1, rule2, rule2_mod

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
'''
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
'''
row_list = []
# df2 contains all sentences from all speeches
for i in range(len(df2)):
    sent = df2.loc[i,'Sent']
    
    if (',' not in sent) and (len(sent.split()) <= 15):
        
        year = df2.loc[i,'Year']
        length = len(sent.split())
        
        dict1 = {'Year':year,'Sent':sent,'Len':length}
        row_list.append(dict1)
        
# df with shorter sentences
df3 = pd.DataFrame(columns=['Year','Sent',"Len"])
df3 = pd.DataFrame(row_list)

from random import randint
def rand_sent(df):
    index = randint(0, len(df))
    print("Index = ", index)
    doc = nlp(df.loc[index,'Sent'].lstrip().rstrip())

    return doc

# function to check output percentage for a rule
def output_per(df,out_col):
    
    result = 0
    
    for out in df[out_col]:
        if len(out)!=0:
            result+=1
    
    per = result/len(df)
    per *= 100
    
    return per

# Create a df containing sentence and its output for rule 1
row_list = []

for i in range(len(df3)):
    
    sent = df3.loc[i,'Sent']
    year = df3.loc[i,'Year']
    output = rule1(sent)
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)
    
df_rule1 = pd.DataFrame(row_list)
df_rule1_all = pd.DataFrame(row_list)

# selecting non-empty output rows
df_show = pd.DataFrame(columns=df_rule1_all.columns)
for row in range(len(df_rule1_all)):
    if len(df_rule1_all.loc[row,'Output'])!=0:
        df_show = df_show.append(df_rule1_all.loc[row,:])

# reset the index
df_show.reset_index(inplace=True)
df_show.drop('index',axis=1,inplace=True)

# separate subject, verb and object

verb_dict = dict()
dis_dict = dict()
dis_list = []

# iterating over all the sentences
for i in range(len(df_show)):
    
    # sentence containing the output
    sentence = df_show.loc[i,'Sent']
    # year of the sentence
    year = df_show.loc[i,'Year']
    # output of the sentence
    output = df_show.loc[i,'Output']
    
    # iterating over all the outputs from the sentence
    for sent in output:
        
        # separate subject, verb and object
        n1 = sent.split()[:1]
        v = sent.split()[1]
        n2 = sent.split()[2:]
        
        # append to list, along with the sentence
        dis_dict = {'Sent':sentence,'Year':year,'Noun1':n1,'Verb':v,'Noun2':n2}
        dis_list.append(dis_dict)
        
        # counting the number of sentences containing the verb
        # verb = sent.split()[1]
        if v in verb_dict:
            verb_dict[v]+=1
        else:
            verb_dict[v]=1

df_sep = pd.DataFrame(dis_list)
print(df_sep.head())

# Create a df containing sentence and its output for rule 2
row_list = []

for i in range(len(df2)):
    
    sent = df2.loc[i,'Sent']
    year = df2.loc[i,'Year']
    # Rule 2
    output = rule2(sent)
    if len(output) != 0:
        dict1 = {'Year':year,'Sent':sent,'Output':output}
        row_list.append(dict1)

df_rule2_show = pd.DataFrame(row_list)
print(df_rule2_show.head())