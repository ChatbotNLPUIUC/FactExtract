import spacy
import glob
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

'''
RULE: Noun-Verb-Noun
'''
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])

# To download dependency graphs to local folder
from pathlib import Path

# Function for rule 1: noun(subject), verb, noun(object)
def rule1(text):  
    doc = nlp(text)
    sent = []
    
    for token in doc:
        
        # If the token is a verb
        if (token.pos_=='VERB'):
            
            phrase =''
            
            # Only extract noun or pronoun subjects
            for sub_tok in token.lefts:
                
                if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):
                    
                    # Add subject to the phrase
                    phrase += sub_tok.text

                    # Save the root of the word in phrase
                    phrase += ' '+token.lemma_ 

                    # Check for noun or pronoun direct objects
                    for sub_tok in token.rights:
                        
                        # Save the object in the phrase
                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):
                                    
                            phrase += ' '+sub_tok.text
                            sent.append(phrase)
            
    return sent


'''
RULE: Adjective Noun Structure

Look for tokens that have a Noun POS tag and have subject or object dependency

Look at the child nodes of these tokens and append it to the phrase only if it modifies the noun
'''

# function for rule 2
def rule2(text):    
    doc = nlp(text)
    pat = []

    # iterate over tokens
    for token in doc:
        phrase = ''
        # if the word is a subject noun or an object noun
        if (token.pos_ == 'NOUN')\
            and (token.dep_ in ['dobj','pobj','nsubj','nsubjpass']):
            
            # iterate over the children nodes
            for sub in token.children:
                # if word is an adjective or has a compound dependency
                if (sub.pos_ == 'ADJ') or (sub.dep_ == 'compound'):
                    phrase += sub.text + ' '
                    
            if len(phrase) != 0:
                phrase += token.text
             
        if len(phrase) != 0:
            pat.append(phrase)
        
    return pat

def rule2_mod(text,index): 
    doc = nlp(text)
    phrase = ''
    
    for token in doc:
        
        if token.i == index:
            
            for subtoken in token.children:
                if (subtoken.pos_ == 'ADJ'):
                    phrase += ' '+subtoken.text
            break
    
    return phrase
