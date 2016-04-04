import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

def tokenize (text):
    #TODO: нужны ли числа? возможно, стоит удалить
    words = re.finditer(r'(-?[\d]+([\.,][\d]+)?)|([A-Za-z]+(-[A-Za-z]+)*)|([А-Яа-я]+(-[А-Яа-я]+)*)', text)
    return map(lambda word: word.span(), words)


# TODO: поискать лемматизаторы (pymorphy?)
def prepareText(text):
    # Удаляем html-теги
    text = re.sub('(<br>)|(\&gt;)|(\&lt;)', ' ', text)
    # Удаляем URL-ы. Спасибо Imme Emosol (https://gist.github.com/imme-emosol/731338)
    text = re.sub(r'(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?',
           '', text)
    # Удаляем обращения
    text = re.sub(r'\[id[1-9][\d]*\|(([A-Za-z]+(-[A-Za-z]+)*)|([А-Яа-я]+(-[А-Яа-я]+)*))\],', '', text)

    words = tokenize(text)

    words = [word for word in words if text[word[0]:word[1]] not in stopwords.words('russian')]

    stemmer = SnowballStemmer('russian')
    stemmedWords = [stemmer.stem(text[word[0]:word[1]]) for word in words]
    return stemmedWords