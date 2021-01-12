# Download https://github.com/huggingface/neuralcoref from source
# Using neuralcoref for coreference resolution
# import neuralcoref
import spacy
nlp = spacy.load('en_core_web_lg')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
doc = nlp(u'My sister has a dog. She loves him.')
test = nlp("I voted for Obama because he was most aligned with my values")
trial = nlp("She said")
print(trial._.has_coref)
print(trial._.coref_clusters)
