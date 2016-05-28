from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import re
from nltk.stem.snowball import SnowballStemmer
import numpy
import scipy.sparse

class CustomClassifier:
    def __init__(self):
        self.stop_words = list(set(stopwords.words('russian')).union(set(stopwords.words('english'))))
        self.vectorizer = CountVectorizer(ngram_range=(1, 3), max_df=0.75)
        self.transformer = TfidfTransformer()
        self.scaler = MaxAbsScaler()
        self.classifier = LogisticRegression()
        self.swearings_list = []
        with open('swearings.txt', 'r') as file:
            self.swearings_list.append(file.readline())

    def tokenize (self, text):
        #TODO: нужны ли числа? возможно, стоит удалить
        words = re.finditer(r'(-?[\d]+([\.,][\d]+)?)|([A-Za-z]+(-[A-Za-z]+)*)|([А-Яа-я]+(-[А-Яа-я]+)*)', text)
        return list(map(lambda word: word.span(), words))

    def cleanText(self, text):
        # Удаляем html-теги
        text = re.sub('(<br>)|(\&gt;)|(\&lt;)', ' ', text)
        # Удаляем URL-ы. Спасибо Imme Emosol (https://gist.github.com/imme-emosol/731338)
        text = re.sub(r'(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?',
               '', text)
        # Удаляем обращения
        text = re.sub(r'\[id[1-9][\d]*\|(([A-Za-z]+(-[A-Za-z]+)*)|([А-Яа-я]+(-[А-Яа-я]+)*))\],', '', text)
        # Удаляем теги vk
        text = re.sub(r'(\s|^)#\w+', '', text)
        return text

# TODO: поискать лемматизаторы (pymorphy?)
    def prepareText(self, text):
        text = self.cleanText(text)

        words = self.tokenize(text)
        words = [word for word in words if text[word[0]:word[1]] not in self.stop_words]

        #и без стеммера тоже лучше
        #stemmer = SnowballStemmer('russian')
        #stemmedWords = [stemmer.stem(text[word[0]:word[1]]) for word in words]
        return list(map(lambda w: text[w[0]:w[1]], words))
        #return stemmedWords

    def punctuationDensity(self, text):
        text = self.cleanText(text)
        if len(text) == 0:
            return 0.0
        punctuation = len(re.findall(r'^[\s\dA-Za-zА-Яа-я]', text))
        return punctuation/len(text)

    def swearingsDensity(self, text):
        text = self.cleanText(text)
        words = self.tokenize(text)
        if len(words) == 0:
            return 0
        swearings = 0
        for word in words:
            if word in self.swearings_list:
                swearings += 1
        return swearings/len(words)

    def fit(self, raw_comments, age_categories):
        comments = list(map(self.prepareText, raw_comments))
        word_numbers = list(map(len, comments))
        word_numbers_array = numpy.empty((len(word_numbers), 1))
        for i in enumerate(word_numbers):
            word_numbers_array[i[0]][0] = i[1]
        word_numbers_array = scipy.sparse.csr_matrix(word_numbers_array)
        punctuation = list(map(self.punctuationDensity, raw_comments))
        punctuation_array = numpy.empty((len(punctuation), 1))
        for i in enumerate(punctuation):
            punctuation_array[i[0]][0] = i[1]
        punctuation_array = scipy.sparse.csr_matrix(punctuation_array)
        swearings = list(map(self.swearingsDensity, raw_comments))
        swearings_array = numpy.empty((len(swearings), 1))
        for i in enumerate(swearings):
            swearings_array[i[0]][0] = i[1]
        swearings_array = scipy.sparse.csr_matrix(swearings_array)
        word_counts = self.vectorizer.fit_transform(list(map(' '.join, comments)))
        weighted_counts = self.transformer.fit_transform(word_counts)
        final_feature_matrix = scipy.sparse.hstack([weighted_counts, word_numbers_array, punctuation_array, swearings_array])
        final_feature_matrix = self.scaler.fit_transform(final_feature_matrix)
        self.classifier.fit(final_feature_matrix, age_categories)

    def predict(self, raw_comments):
        result = []
        for comment in raw_comments:
            features = self.transformer.transform(self.vectorizer.transform([' '.join(self.prepareText(comment))]))
            words_number = numpy.empty((1, 1))
            words_number[0][0] = len(self.prepareText(comment))
            words_number = scipy.sparse.csr_matrix(words_number)
            text_punctuation = numpy.empty((1, 1))
            text_punctuation[0][0] = self.punctuationDensity(comment)
            text_punctuation = scipy.sparse.csr_matrix(text_punctuation)
            text_swearings = numpy.empty((1, 1))
            text_swearings[0][0] = self.swearingsDensity(comment)
            text_swearings = scipy.sparse.csr_matrix(text_swearings)
            features = scipy.sparse.hstack((features, words_number, text_punctuation, text_swearings))
            features = self.scaler.transform(features)
            result.append(self.classifier.predict(features)[0])
        return result
