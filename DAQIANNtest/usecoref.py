# Download https://github.com/huggingface/neuralcoref from source
# Using neuralcoref for coreference resolution
# import neuralcoref
import re
import spacy
nlp = spacy.load('en_core_web_lg')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
doc = nlp("My sister has a dog. The girl loves him. She gives him pets. When he is sad, pets cheer him up.")
test = nlp("I voted for Obama because he was most aligned with my values")
trial = nlp("She said")
#print(trial._.has_coref)
#print(doc._.coref_clusters)

def splitCoref(doc):
    value = doc._.coref_clusters
    if not value:
        print("No coreference")
        return None
    
    sents = str(doc.text).split('.') # Replace with finding clauses function
    res = ""
    index = 0
    for i in range(len(value)):
        index = 0
        replace = value[i][0]
        valueReplace = value[i][1: ].copy()
        while valueReplace and index < len(sents):
            ind = sents[index].find(str(valueReplace[0]))
            if ind != -1:
                #sents[index] = sents[index].replace(str(valueReplace[0]), str(replace), 1)
                sents[index] = re.sub(r"\b%s\b" % str(valueReplace[0]) , str(replace), sents[index])
                valueReplace.pop(0)
            else:
                index += 1
            
    return sents

print(splitCoref(doc))