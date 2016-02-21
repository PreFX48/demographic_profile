import re
from nltk.stem.snowball import SnowballStemmer

def tokenize (text):
    words = re.finditer(r'(-?[\d]+([\.,][\d]+)?)|([A-Za-z]+(-[A-Za-z]+)*)|([А-Яа-я]+(-[А-Яа-я]+)*)', text)
    return map(lambda word: word.span(), words)


def prepareText(text):
    stemmer = SnowballStemmer('russian')
    words = tokenize(text)
    stemmedWords = [stemmer.stem(text[word[0]:word[1]]) for word in words]
    return stemmedWords