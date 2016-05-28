import json
from text_preparation import prepareText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy
import scipy.sparse


def getAgeCategory(birth_year):
    CURRENT_YEAR = 2016
    age = CURRENT_YEAR - birth_year
    if 0 < age <= 18:
        age_category = '0-18'
    elif 19 <= age <= 25:
        age_category = '19-25'
    elif 26 <= age <= 35:
        age_category = '26-35'
    elif 36 <= age <= 45:
        age_category = '36-45'
    else:
        age_category = '46+'
    return age_category


def initClassifierEnvironment(json_file_path):
    json_data = []
    raw_comments = []
    bdates = []
    with open(json_file_path, 'r') as fp:
        json_data = json.load(fp)
    min_year = 2016
    max_year = 1
    total_year = 0
    people_number = 0
    for person in json_data:
        if people_number >= 1000:
            break
        if 1950 <= person['bdate'] <= 2010:
            min_year = min(min_year, person['bdate'])
            max_year = max(max_year, person['bdate'])
            total_year += person['bdate']
            people_number += 1
        for comment in person['user_comments']:
            raw_comments.append(comment)
            bdates.append(person['bdate'])
    age_categories = list(map(getAgeCategory, bdates))
    comments = list(map(prepareText, raw_comments))
    word_numbers = list(map(len, comments))
    word_numbers_array = numpy.empty((len(word_numbers), 1))
    for i in enumerate(word_numbers):
        word_numbers_array[i[0]][0] = i[1]
    word_numbers_array = scipy.sparse.csr_matrix(word_numbers_array)
    edited_texts = list(map(" ".join, comments))
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    word_counts = vectorizer.fit_transform(edited_texts)
    transformer = TfidfTransformer()
    weighted_counts = transformer.fit_transform(word_counts)
    final_feature_matrix = scipy.sparse.hstack([weighted_counts, word_numbers_array])
    scaler = MaxAbsScaler()
    final_feature_matrix = scaler.fit_transform(final_feature_matrix)
    NB_classifier = MultinomialNB()
    NB_classifier.fit(final_feature_matrix, age_categories)
    logistic_classifier = LogisticRegression()
    logistic_classifier.fit(final_feature_matrix, age_categories)


    bayes_mistake = 0
    logistic_mistake = 0
    number = 0
    for raw_comment, comment, age_category in zip(raw_comments, comments, age_categories):
        features = transformer.transform(vectorizer.transform([' '.join(comment)]))
        words_number = numpy.empty((1, 1))
        words_number[0][0] = len(comment)
        words_number = scipy.sparse.csr_matrix(words_number)
        features = scipy.sparse.hstack((features, words_number))
        features = scaler.transform(features)
        if age_category != NB_classifier.predict(features):
            bayes_mistake += 1
        number += 1
        if age_category != logistic_classifier.predict(features):
            logistic_mistake += 1
    print('Min year:', min_year)
    print('Max year:', max_year)
    print('Average year:', total_year / people_number)
    print('Naive Bayes correct in:', str(round(100 - bayes_mistake*100 / number, 1)) + '%')
    print('Logistic regression correct in:', str(round(100 - logistic_mistake*100 / number, 1)) + '%')

import timeit
start = timeit.default_timer()
initClassifierEnvironment('data.json')
stop = timeit.default_timer()
print("TIME:", stop - start)
