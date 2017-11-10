import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


# create a dictionary of most common words used across mails
# delete special characters etc
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
                    break

    dict = Counter(all_words)

    list_to_remove = dict.keys()
    for item in list_to_remove:
        if not item.isalpha():
            del dict[item]
        elif len(item) == 1:
            del dict[item]
    dict = dict.most_common(3000)
    return dict


train_dir = os.getcwd() + '/training-data/'
dictionary = make_Dictionary(train_dir)

print(dictionary[0], dictionary[1])


def extract_features(train_dir):
    files = [os.path.join(train_dir, fi) for fi in os.listdir(train_dir)]
    # Number of files * 3000 (feature vector)
    features_matrix = np.zeros((len(files), 3000))
    docID = 0
    # for all files in directory
    for file in files:
        with open(file) as fi:
            print(file)
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for j, d in enumerate(dictionary):
                            # How many time word occurs in current mail
                            if d[0] == word:
                                wordID = j
                                # add feature value to current mail train and feature column j
                                features_matrix[docID, wordID] = words.count(word)
                    break
            # go to next mail train data
            docID = docID + 1
    return features_matrix

train_labels = np.zeros(702)
train_labels[351:701] = 1 # Spam mails in the end
train_matrix = extract_features(train_dir)

print('Number of Training Examples: ' + str(len(train_matrix)))

NB = MultinomialNB()
LSVC = LinearSVC()

NB.fit(train_matrix, train_labels)
LSVC.fit(train_matrix, train_labels)

test_dir = os.getcwd() + '/test-data/'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1

result_nb = NB.predict(test_matrix)
result_lsvc = LSVC.predict(test_matrix)

labels = [1, 0]
print confusion_matrix(test_labels, result_nb, labels)
print confusion_matrix(test_labels, result_lsvc, labels)
