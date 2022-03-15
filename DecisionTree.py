# Load libraries
import pandas as pd
import fileinput
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import six
import sys

sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO
# from IPython.display import Image
from IPython.display import display, Image
import pydotplus
from subprocess import check_call


# col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
# pima = pd.read_csv("diabetes.csv", header=None, names=col_names)
# dataset = pd.read_csv("diabetes.csv", header=0)
def dataProcessing(filename):
    classification = ''
    data = []
    attributes = []
    attributeFlag = 0
    targetFlag = 0
    dataFlag = 0
    for lines in fileinput.input([filename]):
        lines = lines.strip()
        entry = lines.split()

        if (entry != []):
            if attributeFlag == 1:
                label = entry[0].strip('#:')
                attributes.append(label)
            if targetFlag == 1:
                classification = entry[0].strip('#:')
                attributes.append(classification)
            if dataFlag == 1:
                val = entry[0].split(',')
                data.append(val)
            if entry[0] == '#attributes':
                attributeFlag = 1
                targetFlag = 0
                dataFlag = 0
            if entry[0] == '#target':
                attributeFlag = 0
                targetFlag = 1
                dataFlag = 0
            if entry[0] == '#data':
                attributeFlag = 0
                targetFlag = 0
                dataFlag = 1
        if entry == []:
            attributeFlag = 0
            targetFlag = 0
            dataFlag = 0

    data_df = pd.DataFrame(data, columns=attributes)  # Creating DataFrame
    return data_df, classification


def generate_encodings(df: pd.DataFrame):
    attrs = dict()
    with open('fishing.data', mode='r') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('#attributes') or line.startswith('#target'):
                for l in fin:
                    l = l.strip()
                    if len(l) == 0:
                        break
                    parts = l.split(":")
                    attrs[parts[0][1:]] = parts[1].strip().split(',')

    for col in attrs.keys():
        series = df[col].tolist()
        for i in range(len(series)):
            series[i] = attrs[col].index(series[i])
        df[f'{col}_encoded'] = series


dataset, classification = dataProcessing('fishing.data')
generate_encodings(dataset)
print("Dataset", dataset)

# Reading data file and processing into data structure
count = 0
data = []
Class = list(dataset.columns)[-1]
print(Class)
header_label = list(dataset.columns)[:-1]
print("header label", header_label)
# dataset = pd.DataFrame(data,columns=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'])
# data.head()

# split dataset in features and target variable
feature_cols = header_label
X = dataset[feature_cols]  # Features
y = dataset.Class  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
# f = open("InputFile.dot", "w")
# f.write(''.join(pydotplus.graph_from_dot_data(dot_data.getvalue())))
# f.close()

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# check_call(['dot','-Tpng','InputFile.dot','-o','OutputFile.png'])
# print(graph)
# graph.write_png('diabetes.png')
graph.write_raw('InputFile.dot')
check_call(['dot', '-Tpng', 'InputFile.dot', '-o', 'OutputFile.png'])
# graph.write_png('diabetes.png')
# Image(graph.create_png())
