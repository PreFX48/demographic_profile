import json
from text_analysis import CustomClassifier
import numpy
from sklearn.metrics import accuracy_score, classification_report
from sklearn import cross_validation

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

def testSamples(json_file_path, nfolds):
    json_data = []
    raw_comments = []
    bdates = []
    with open(json_file_path, 'r') as fp:
        json_data = json.load(fp)
    people_number = 0
    for person in json_data:
        if people_number >= 1000:
            break
        people_number += 1
        for comment in person['user_comments']:
            raw_comments.append(comment)
            bdates.append(person['bdate'])
    age_categories = list(map(getAgeCategory, bdates))

    raw_comments = numpy.array(raw_comments)
    age_categories = numpy.array(age_categories)
    kfold = cross_validation.KFold(len(raw_comments), n_folds=nfolds, shuffle=True)
    totalAccuracy = 0
    classifier = CustomClassifier()
    fold_number = 1
    for train, test in kfold:
        classifier.fit(raw_comments[train], age_categories[train])
        result = numpy.array(classifier.predict(raw_comments[test]))
        accuracy = round(accuracy_score(age_categories[test], result)*100, 1)
        print('Fold #'+str(fold_number), 'accuracy:', accuracy)
        totalAccuracy += accuracy
        print(classification_report(age_categories[test], result))
        fold_number += 1
    print('Average accuracy:', totalAccuracy/nfolds)
    return totalAccuracy/nfolds



import timeit
start = timeit.default_timer()
totalAccuracy = 0
testSamples('data.json', 5)
stop = timeit.default_timer()
print("TIME:", round(stop - start, 1), 'secs')
