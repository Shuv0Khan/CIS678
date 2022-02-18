import regex
from sklearn.naive_bayes import GaussianNB
import numpy as nb
from nltk import word_tokenize
from nltk import corpus
from project2.naive_bayes import NB
from collections import defaultdict


def sampling(filename):
    """
    10-fold cross-validation files. each training set will have 90% of each
    class.
    :param filename: dictionary of lists containing filenames for each class
    :return:
    """
    hams = []
    spams = []
    with open(filename, mode='r', encoding='utf8') as fin:
        for line in fin:
            if line.startswith('ham'):
                hams.append(line.strip())
            else:
                spams.append(line.strip())

    filenames = {
        'ham': [],
        'spam': []
    }

    len_train_ham = int(len(hams) * 0.9)
    len_test_ham = len(hams) - len_train_ham
    index = 0
    while index < len(hams):
        file_train = f'ham_train_{index}.txt'
        file_test = f'ham_test_{index}.txt'
        filenames['ham'].append((file_train, file_test))

        upto = index + len_test_ham
        if upto > len(hams):
            upto = len(hams)

        with open(file_train, mode='w', encoding='utf8') as fout_train, open(
                file_test, mode='w', encoding='utf8') as fout_test:
            for i in range(0, index, 1):
                fout_train.write(f'{hams[i]}\n')

            for i in range(index, upto, 1):
                fout_test.write(f'{hams[i]}\n')

            for i in range(upto, len(hams), 1):
                fout_train.write(f'{hams[i]}\n')

        index += len_test_ham

    len_train_spam = int(len(spams) * 0.9)
    len_test_spam = len(spams) - len_train_spam
    index = 0
    while index < len(spams):
        file_train = f'spam_train_{index}.txt'
        file_test = f'spam_test_{index}.txt'
        filenames['spam'].append((file_train, file_test))

        upto = index + len_test_spam
        if upto > len(spams):
            upto = len(spams)

        with open(file_train, mode='w', encoding='utf8') as fout_train, open(
                file_test, mode='w', encoding='utf8') as fout_test:
            for i in range(0, index, 1):
                fout_train.write(f'{spams[i]}\n')

            for i in range(index, upto, 1):
                fout_test.write(f'{spams[i]}\n')

            for i in range(upto, len(spams), 1):
                fout_train.write(f'{spams[i]}\n')

        index += len_test_spam

    return filenames


def textual_data(filename):
    train_test_files = sampling(filename)
    stop_words = set(corpus.stopwords.words('english'))
    TP = TN = FP = FN = 0

    for i in range(0, len(train_test_files['ham']), 1):
        train_ham, test_ham = train_test_files['ham'][i]
        train_spam, test_spam = train_test_files['spam'][i]
        train_filename = 'training_set.txt'
        test_filename = 'test_set.txt'
        with open(train_filename, mode='w', encoding='utf8') as fout_train, open(
                test_filename, mode='w', encoding='utf8') as fout_test, open(
                train_ham, mode='r', encoding='utf8') as fin_train_ham, open(
                train_spam, mode='r', encoding='utf8') as fin_train_spam, open(
                test_ham, mode='r', encoding='utf8') as fin_test_ham, open(
                test_spam, mode='r', encoding='utf8') as fin_test_spam:

            trains = fin_train_ham.readlines()
            trains.extend(fin_train_spam.readlines())
            test = fin_test_ham.readlines()
            test.extend(fin_test_spam.readlines())

            for line in trains:
                fout_train.write(f'{line.strip()}\n')

            for line in test:
                fout_test.write(f'{line.strip()}\n')

        nb = NB()
        nb.train_for_text(datafile=train_filename)

        with open(test_filename, mode='r', encoding='utf8') as fin:
            for line in fin:
                new_instance = line.strip()

                if len(new_instance) == 0:
                    continue

                # lowercase trained model, so lowercase test
                new_instance = new_instance.lower()

                # remove punctuations
                new_instance = regex.sub(r"[.,?!:;()]|\d+", "", new_instance)

                # removing stopwords
                words = word_tokenize(new_instance)
                valid_words = []
                for w in words:
                    if w not in stop_words:
                        valid_words.append(w)

                words = valid_words

                # words = new_instance.split()
                # do pre-processing

                actual_cls = words[0]
                prediction_cls = nb.test(words[1:], no_print=True)

                if actual_cls == 'ham' and prediction_cls == 'ham':
                    TP += 1
                elif actual_cls == 'spam' and prediction_cls == 'spam':
                    TN += 1
                elif actual_cls == 'ham' and prediction_cls == 'spam':
                    FN += 1
                elif actual_cls == 'spam' and prediction_cls == 'ham':
                    FP += 1

    print(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')


def tabular_data():
    filename = input("Enter training dataset name: ")
    nb = NB()
    nb.train(class_at_column=1, header=False, filepath=filename, delimeter=' ')

    while True:
        new_instance = input("Enter new Instance, separated by comma: ")
        nb.test(new_instance.split(','))

        s = input("Continue?(Y/n): ")
        if s == 'n':
            break


def gaussian_nb():
    X = []
    Y = []
    classes = {
        'Yes': 1,
        'No': 0
    }
    wind = {
        'Strong': 0,
        'Weak': 1
    }
    air = {
        'WarmAir': 0,
        'ColdAir': 1
    }
    water = {
        'Warm': 0,
        'Moderate': 1,
        'Cold': 2
    }
    sky = {
        'Sunny': 0,
        'Cloudy': 1,
        'Rainy': 2
    }
    with open('fishing.data', mode='r') as fin:
        for line in fin:
            parts = line.strip().split(' ')
            Y.append(classes[parts[0]])

            X.append([wind[parts[1]], air[parts[2]], water[parts[3]], sky[parts[4]]])

    classifier = GaussianNB()
    classifier.fit(nb.array(X), nb.array(Y))

    while True:
        new_instance = input("Enter new Instance, separated by comma: ")
        parts = new_instance.split(',')
        new_instance = [[wind[parts[0]], air[parts[1]], water[parts[2]], sky[parts[3]]]]
        c = classifier.predict(new_instance)

        if c == classes['Yes']:
            print(f'Classify: Yes')
        else:
            print(f'Classify: No')

        s = input("Continue?(Y/n): ")
        if s == 'n':
            break
