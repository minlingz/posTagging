"""Components of a part-of-
speech hidden markov model Generator.

Minling Zhou, 2023
Meixiang Du, 2023

"""
from typing import List, Tuple
import nltk
import numpy as np
from viterbi import viterbi


def train(tagged_sentences: List[List[Tuple[str, str]]]):
    """Generate initial state distribution.

    Args:
        tagged_sentences (List[List[Tuple[str, str]]]): A list of tagged sentences.

    """
    pi = np.zeros(len(tag_tokens))
    A = np.ones((len(tag_tokens), len(tag_tokens)))  # add-one smoothing
    B = np.ones((len(tag_tokens), len(word_tokens)))  # add-one smoothing

    # count pi, A, B
    for sentence in tagged_sentences:
        pi[tag_tokens.index(sentence[0][1])] += 1
        for i in range(len(sentence) - 1):
            B[tag_tokens.index(sentence[i][1]), word_tokens.index(sentence[i][0])] += 1
            if i != len(sentence) - 1:
                A[
                    tag_tokens.index(sentence[i][1]),
                    tag_tokens.index(sentence[i + 1][1]),
                ] += 1

    # normalize pi, A, B
    pi /= pi.sum()
    A /= A.sum(axis=1, keepdims=True)
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B


sentences = nltk.corpus.brown.sents()[:10000]
train_tagset = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]

# map words to indices
word_tokens = list(set([word for sentence in sentences for word in sentence]))
word_tokens.append("UNK")
tag_tokens = list(set([tag for tagset in train_tagset for _, tag in tagset]))
print("tags:", tag_tokens)

# train training set
pi, A, B = train(train_tagset)

# predict test set using viterbi and compare agasint the actual tags
test_tagset = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]

for test in test_tagset:
    obs = []
    s = []
    for word in test:
        s.append(tag_tokens.index(word[1]))
        if word[0] in word_tokens:
            obs.append(word_tokens.index(word[0]))
        else:
            obs.append(word_tokens.index("UNK"))

    print("predicted tags:", s)
    qs, ps = viterbi(obs, pi, A, B)
    print("actual tags   :", qs)
