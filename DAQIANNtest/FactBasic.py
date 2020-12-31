# import spacy
import spacy
import glob

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

folders = glob.glob('../Converted sessions/Session*')