import json
from text_preparation import prepareText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
import sklearn.preprocessing
import numpy
import scipy.sparse
numpy.set_printoptions(threshold=numpy.nan)


def initClassifierEnvironment(jsonFilePath):
    jsonData = []
    comments = []
    bdates = []
    with open(jsonFilePath, 'r') as fp:
        jsonData = json.load(fp)
    minYear = 2016
    maxYear = 1
    totalYear = 0
    peopleNumber = 0
    for person in jsonData:
        if peopleNumber >= 1000:
            break
        if 1950 <= person['bdate'] <= 2010:
            minYear = min(minYear, person['bdate'])
            maxYear = max(maxYear, person['bdate'])
            totalYear += person['bdate']
            peopleNumber += 1
        for comment in person['user_comments']:
            comments.append(comment)
            bdates.append(person['bdate'])
    comments = list(map(prepareText, comments))
    editedTexts = list(map(" ".join, comments))
    lengths = list(map(len, comments))
    lengthsArray = numpy.empty((len(lengths), 1))
    for i in range(0, len(lengths)):
        lengthsArray[i][0] = lengths[i]
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    wordCounts = vectorizer.fit_transform(editedTexts).toarray()
    transformer = TfidfTransformer()
    weightedCounts = transformer.fit_transform(wordCounts).toarray()
    finalFeatureMatrix = numpy.concatenate((weightedCounts, lengthsArray), axis=1)
    scaler = MaxAbsScaler()
    finalFeatureMatrix = scaler.fit_transform(finalFeatureMatrix)
    classifier = MultinomialNB()
    classifier.fit(finalFeatureMatrix, bdates)


    mistake = 0
    number = 0
    for comment, bdate in zip(comments, bdates):
        features = transformer.transform(vectorizer.transform([' '.join(comment)]))
        length = numpy.empty((1, 1))
        length[0][0] = len(comment)
        length = scipy.sparse.csr_matrix(length)
        features = scipy.sparse.hstack((features, length))
        features = scaler.transform(features)
        mistake += abs(bdate - classifier.predict(features))
        number += 1
    print('Min year:', minYear)
    print('Max year:', maxYear)
    print('Average year:', totalYear / peopleNumber)
    print('Average mistake:', mistake / number)


initClassifierEnvironment('data.json')
