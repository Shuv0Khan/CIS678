from collections import defaultdict

import graphviz
import pandas as pd
from sklearn import tree

from project3.decision_tree import DecisionTree, pre_processing


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


def load_and_encode(filepath: str) -> tuple[pd.DataFrame, dict]:
    df, attrs = pre_processing(filepath)

    for col in attrs.keys():
        series = df[col].tolist()
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
        x.append(df.iloc[i, len(cols):2*len(cols)-1].tolist())

    y = df[df.columns[-1]].tolist()

    dt = tree.DecisionTreeClassifier()
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


def run():
    filename = input('Enter dataset name: ')
    custom_dtree(filename)
    sklearn_dtree(filename)
