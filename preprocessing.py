import re
import nltk
import re
import numpy as np
import pandas as pd

from datasets import Dataset

from nltk.corpus import words
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
from nltk.stem import *

vocab = set(words.words())

def clean_sentence(input, stemmed_words, lemmatizer = WordNetLemmatizer(), stemmer = PorterStemmer()):
    unknown_words = set()
    remove_trailing_punctuation = re.sub('[,\\.:?!]+$|[,\\.:?!]+\\s', ' ', input)
    splitted = remove_trailing_punctuation.split()
    result = []
    for word in splitted:
        lower_word = word.lower()
        stemmed = stemmer.stem(word)
        lemmatized = lemmatizer.lemmatize(lower_word)
        
        if lemmatized in vocab:
            if lemmatized not in stopwords.words('english') and lower_word not in stopwords.words('english'):
                result.append(lemmatized)
        elif stemmed in stemmed_words:
            if stemmed not in stopwords.words('english') and lower_word not in stopwords.words('english'):
                result.append(stemmed) 
        else:
            result.append(word)
            if word not in unknown_words:
                unknown_words.add(word)
    return " ".join(result), unknown_words

def clean_dataset(dataset, stemmed_words):
    unknown_words = set()
    cleaned_sentences_1 = []
    cleaned_sentences_2 = []
    scores = []
    for i in range(dataset.num_rows):
        cleaned_sentence1_and2 = ['', '']
        for j in range(2):
            sentence = dataset[i]['sentence{j}'.format(j=j+1)]
            cleaned_sentence, unknown = clean_sentence(sentence, stemmed_words)
            cleaned_sentence1_and2[j] = cleaned_sentence
            unknown_words = unknown_words.union(unknown)
        cleaned_sentences_1.append(cleaned_sentence1_and2[0])
        cleaned_sentences_2.append(cleaned_sentence1_and2[1])
        scores.append(dataset[i]['score'])

    cleaned_df = pd.DataFrame({
        "sentence1": cleaned_sentences_1,
        "sentence2": cleaned_sentences_2,
        "score": scores
    })
    return Dataset.from_pandas(cleaned_df), unknown_words