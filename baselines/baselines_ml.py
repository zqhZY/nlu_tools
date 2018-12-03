from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import random

def load_data(data_type):
    texts = []
    labels = []
    with open("../data/smp/" + data_type + "_raw.txt") as f:
        for line in f:
            tokens = line.strip().split("\t")
            texts.append(tokens[2])
            labels.append(tokens[1])
    return texts, labels

def load_label_ids(filename):
    label_map = {}
    id2label = {}
    with open(filename) as f:
        for line in f:
            tokens = line.strip().split("\t")
            label_map[tokens[0]] = int(tokens[1]) - 1
            id2label[int(tokens[1])-1] = tokens[0]
    return label_map, id2label

label_map, id2label = load_label_ids("../data/smp/label_ids.txt")

train_text, train_labels = load_data("train")
dev_text, dev_labels = load_data("dev")
test_text, test_labels = load_data("test")

train_labels = [label_map[label] for label in train_labels]
dev_labels = [label_map[label] for label in dev_labels]
test_labels = [label_map[label] for label in test_labels]

data_set = train_text + dev_text + test_text
tf_vector = CountVectorizer()
bow_features = tf_vector.fit_transform(data_set)

feature_names = tf_vector.get_feature_names()
print("total vocab num is {}".format(len(feature_names)))
# print(feature_names)

train_features = tf_vector.transform(train_text + dev_text)
train_labels = train_labels + dev_labels
#dev_features = tf_vector.transform(dev_text)
test_features = tf_vector.transform(test_text)

print(train_features.shape)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 5, 10, 20, 100, 1000], 'gamma': [0.01]}
clf = GridSearchCV(SVC(kernel='rbf', C=1e3, gamma=0.1), parameters, verbose=2)
# tuned_parameters = [{"C": [1, 2, 5, 10, 20, 100], "kernel": ["linear"]}]
#clf = SVC(kernel='linear', C=1e3, gamma=0.1)
clf.fit(train_features, train_labels)

preds = clf.predict(test_features)
acc = accuracy_score(test_labels, preds)
recall = recall_score(test_labels, preds, average='macro')
f1 = f1_score(test_labels, preds, average='macro')
print("acc is {}, recall is {}, f1 is {}".format(acc, recall, f1))


clf = MultinomialNB()
clf.fit(train_features, train_labels)

preds = clf.predict(test_features)
acc = accuracy_score(test_labels, preds)
recall = recall_score(test_labels, preds, average='macro')
f1 = f1_score(test_labels, preds, average='macro')
print("acc is {}, recall is {}, f1 is {}".format(acc, recall, f1))



# random
preds = [random.randint(0, len(label_map) - 1) for _ in range(len(test_labels))]
acc = accuracy_score(test_labels, preds)
recall = recall_score(test_labels, preds, average='macro')
f1 = f1_score(test_labels, preds, average='macro')
print("acc is {}, recall is {}, f1 is {}".format(acc, recall, f1))
