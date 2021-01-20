import Rules
import spacy
import re

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

nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])
test = "Thus, Poisson processes tie together two important distributions, one discrete and one continuous, and the use of a common symbol for both the Poisson and Exponential parameters is felicitous notation, for lambda is the arrival rate in the process that unites the two distributions."
test = clean(test)
doc = nlp(test)

for word in doc:
    print(word, word.tag_)

temp1 = Rules.rule1_mod(test)
temp2 = Rules.rule2(test)
temp3 = Rules.rule3_mod(test)

print(temp1)
print(temp2)
print(temp3)