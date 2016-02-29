import re
from nltk.stem.snowball import SnowballStemmer

def tokenize (text):
    words = re.finditer(r'(-?[\d]+([\.,][\d]+)?)|([A-Za-z]+(-[A-Za-z]+)*)|([А-Яа-я]+(-[А-Яа-я]+)*)', text)
    return map(lambda word: word.span(), words)


# TODO: поискать лемматизаторы (pymorphy?)
def prepareText(text):
    # Удаляем html-теги (он тут вроде бы только один)
    text = re.sub('<br>', ' ', text)
    # Удаляем URL-ы. Спасибо Imme Emosol (https://gist.github.com/imme-emosol/731338)
    text = re.sub(r'(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?',
           '', text)
    # Удаляем обращения
    text = re.sub(r'\[id[1-9][\d]*\|(([A-Za-z]+(-[A-Za-z]+)*)|([А-Яа-я]+(-[А-Яа-я]+)*))\],', '', text)

    stemmer = SnowballStemmer('russian')

    words = tokenize(text)

    # TODO: пробежаться и удалить стоп-слова?
    stemmedWords = [stemmer.stem(text[word[0]:word[1]]) for word in words]
    return stemmedWords