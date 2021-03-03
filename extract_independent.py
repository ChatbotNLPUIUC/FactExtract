from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from nltk import ParentedTree
from tqdm import tqdm

def split_complete_sents(sentences):
    print("Downloading AllenNLP Constituency Parser...")
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    print("Done!")
    split_sentences = []
    print("Separating independent clauses...")
    for sent in sentences:
        result = extract_independent_clauses(sent, predictor)
        # TODO: delete later
        if len(result) > 1:
            print("-------------------------------")
            print(f"original: {sent}")
            print(f"new: {result}")
            print("-------------------------------")
        split_sentences.extend(result)
    return split_sentences

def filt_r(x):
    if x.label() == "S":
        right_siblings = []
        r = x
        while r:
            if r.label() != ".":
                right_siblings.append(r.label())
            r = r.right_sibling()
    else:
        return False
    if right_siblings == ["S", ",", "CC", "S"] or right_siblings == ["S", ":", "S"] or right_siblings == ["S", ",", "S"]:
        return True
    return False

def filt_l(x):
    if x.label() == "S":
        left_siblings = []
        l = x
        while l:
            if l.label() != ".":
                left_siblings.append(l.label())
            l = l.left_sibling()
    else:
        return False
    if left_siblings[::-1] == ["S", ",", "CC", "S"] or left_siblings[::-1] == ["S", ":", "S"] or left_siblings == ["S", ",", "S"]:
        return True
    return False


def extract_independent_clauses(input_sent, predictor):
    output = predictor.predict(sentence=input_sent)
    tree_str = output["trees"]
    t = ParentedTree.fromstring(tree_str)
    candidate_nodes = list(t.subtrees(filter=lambda x: filt_r(x) or filt_l(x)))
    for node in candidate_nodes:
        if node.parent() in candidate_nodes:
            candidate_nodes.remove(node.parent())
    sub_sentences = []
    for candidate in candidate_nodes:
        temp = []
        for subtree in candidate:
            temp += subtree.leaves()
        sub_sentences.append(temp)
    sub_sentences = sub_sentences if sub_sentences else [t.leaves()]
    sentences = []
    for sentence in sub_sentences:
        temp = ""
        for i, word in enumerate(sentence):
            if i == 0:
                temp += word[0].title() + word[1:]
            elif word in [".", "!", "?", ",", ";"]:
                temp += word
            else:
                temp += " " + word
        temp = temp.replace(" ’", "’")
        temp = temp.replace(" n’", "n’")
        sentences.append(temp)
    return sentences