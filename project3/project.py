from collections import defaultdict

import graphviz
import pandas as pd
from sklearn import tree

from project3.decision_tree import DecisionTree


def pre_processing(filepath: str) -> tuple[pd.DataFrame, dict]:
    data = []
    attrs = defaultdict(list)
    dtypes = dict()
    with open(filepath, mode='r') as fin:
        for line in fin:
            line = line.strip()

            if line.startswith("#attributes") or line.startswith("#target"):
                for l in fin:
                    l = l.strip().upper()
                    if len(l) == 0:
                        break
                    parts = l.split(":")
                    attrs[parts[0][1:]] = parts[1].strip().split(",")

                    if parts[1].strip() == 'NUMERIC':
                        dtypes[parts[0][1:]] = 'float'
                    else:
                        dtypes[parts[0][1:]] = 'string'

            elif line.startswith("#data"):
                for l in fin:
                    l = l.strip().upper()
                    if len(l) == 0:
                        break
                    parts = l.split(",")
                    data.append(parts)

    '''Load data into DataFrame'''
    df = pd.DataFrame(data)

    '''Rename columns with attribute names'''
    df.columns = list(attrs.keys())

    '''Convert to best possible dtypes'''
    # df = df.convert_dtypes()
    df = df.astype(dtypes)

    print(df.info())

    return df, attrs


def custom_dtree(filename: str):
    print('****************************************************')
    print('\tCustom implementation of Decision Tree.')
    print('****************************************************')

    print('\nStep 1: Pre-processing and Initialization.')
    df, attr = pre_processing(filename)
    dt = DecisionTree(df, attr)

    print('\nStep 2: Training...', end='')
    dt.train()
    print('Done')

    print('\nStep 2: Plotting Tree...', end='')
    dt.plot_tree()
    print('Done')

    print('\nStep 3: Prediction.')
    q = 'a'
    inp = defaultdict(str)
    for item in attr:
        inp[item] = ''
    while q != 'q':
        for key in inp:
            val = input(f"{key}: ")
            inp[key] = val

        # dt.predict({'WIND': 'WEAK', 'WATER': 'MODERATE', 'AIR': 'WARMAIR', 'SKY': 'RAINY'})
        dt.predict(inp)
        q = input('Press "q" to quit, any key to continue: ')

    print('\n\nStep 4: Rule Extraction.')
    dt.extract_rules()


def load_and_encode(filepath: str) -> tuple[pd.DataFrame, dict]:
    df, attrs = pre_processing(filepath)

    for col in attrs.keys():
        series = df[col].tolist()
        if pd.api.types.is_string_dtype(df[col]):
            for i in range(len(series)):
                series[i] = attrs[col].index(series[i])
        df[f'{col}_encoded'] = series

    return df, attrs


def sklearn_dtree(filename: str):
    print('****************************************************')
    print('\tsklearn implementation of Decision Tree.')
    print('****************************************************')

    print('\nStep 1: Pre-processing and Initialization.')
    df, attrs = load_and_encode(filename)
    cols = list(attrs.keys())
    print(df.head())

    print('\nStep 2: Training...', end='')
    x = []
    for i in df.index:
        x.append(df.iloc[i, len(cols):2 * len(cols) - 1].tolist())

    y = df[df.columns[-1]].tolist()

    dt = tree.DecisionTreeClassifier(criterion='entropy')
    dt = dt.fit(x, y)
    print('Done')

    print('\nStep 2: Plotting Tree...', end='')
    dot_tree = tree.export_graphviz(dt, out_file=None,
                                    feature_names=cols[:-1],
                                    class_names=cols[-1],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_tree)
    graph.render('decision_tree')
    print('Done')

    print('\nStep 3: Prediction.')
    q = 'a'
    inp = []
    while q != 'q':
        for key in cols[:-1]:
            val = input(f"{key}: ")
            inp.append(attrs[key].index(val.upper()))

        # dt.predict({'WIND': 'WEAK', 'WATER': 'MODERATE', 'AIR': 'WARMAIR', 'SKY': 'RAINY'})
        cls = dt.predict([inp])[0]
        print(f"Prediction: {attrs[cols[-1]][cls]}")
        q = input('Press "q" to quit, any key to continue: ')


def bagging(filename: str):
    pass


def run():
    filename = input('Enter dataset name: ')
    custom_dtree(filename)
    sklearn_dtree(filename)
    bagging(filename)
