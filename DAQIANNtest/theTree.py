import spacy
from nltk import Tree

nlp = spacy.load('en_core_web_sm')

doc = nlp("Language is one of the most uniquely human capacities that our species possesses, and one that is involved in all others, including consciousness, sociality and culture.")

def to_nltk_tree(node):
    res = {}
    if node.n_lefts + node.n_rights > 0:
        res[node.orth_] = [to_nltk_tree(child) for child in node.children]
    else:
        print(node.orth_)
    return res

for token in doc:
    if token.dep_ == 'nsubj' or token.pos_ == 'NOUN': # Or other forms of subjects / objects
        print(token.lemma_+"'s:")
        for a in token.ancestors:
            #print(a)
            #print(a.pos_)
            #if a.pos_ == 'VERB': # Or however you determine your selection
                #for atok in a.children:
                    #if atok.dep_ == 'acomp': # Note, you should look for more than just acomp
            print(a.text)

# doc2 = nlp("I live in New York")
# print("Before:", [token.text for token in doc2])

# with doc2.retokenize() as retokenizer:
#     retokenizer.merge(doc2[3:5], attrs={"LEMMA": "new york"})
# print("After:", [token.text for token in doc2])