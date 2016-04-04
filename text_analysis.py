import json
from text_preparation import prepareText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy
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
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    wordCounts = vectorizer.fit_transform(editedTexts).toarray()
    transformer = TfidfTransformer()
    weightedCounts = transformer.fit_transform(wordCounts).toarray()


    classifier = MultinomialNB()
    classifier.fit(weightedCounts, bdates)


    mistake = 0
    number = 0
    for comment, bdate in zip(comments, bdates):
        mistake += abs(bdate - classifier.predict(transformer.transform(vectorizer.transform([' '.join(comment)]))))
        number += 1
    print('Min year:', minYear)
    print('Max year:', maxYear)
    print('Average year:', totalYear / peopleNumber)
    print('Average mistake:', mistake / number)

    return (vectorizer, transformer, classifier)

environment = initClassifierEnvironment('data.json')
vectorizer = environment[0]
tfidfTransformer = environment[1]
classifier = environment[2]

