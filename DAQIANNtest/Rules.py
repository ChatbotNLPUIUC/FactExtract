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