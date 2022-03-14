import math

import pandas as pd
from collections import defaultdict
from pyvis.network import Network
from queue import Queue


class DecisionTree:
    __data = None
    __attrs = None
    __cls = None
    __tree = defaultdict(dict)

    def __init__(self, df: pd.DataFrame, attrs=None):
        self.__data = df
        self.__gen_attr_set(attrs)

    def __gen_attr_set(self, attrs=None):
        """
        If attrs is None, generates the unique values for each attribute
        from the dataset. If any value of an attribute is not present in
        the dataset, this will fail to discover all possible values of an
        attribute.
        In that case attrs must not be None.

        Considers class to be the last column.
        :return:
        """
        self.__attrs = dict()
        self.__cls = dict()

        if attrs is not None:
            k, l = attrs.popitem()
            self.__cls = {k: l}
            self.__attrs = attrs.copy()
            return

        columns = self.__data.columns.to_list()
        for col in columns[:-1]:
            self.__attrs[col] = self.__data[col].unique().tolist()

        self.__cls[columns[-1]] = self.__data[columns[-1]].unique().tolist()

    def __entropy_of(self, cls_freq: dict) -> float:
        total = 0
        for i in cls_freq.values():
            total += i

        entropy = 0
        for i in cls_freq.values():
            entropy -= (i / total) * math.log2((i / total))

        return entropy

    def __gain_of(self, node_entropy, col, col_cls_name, df):
        S = defaultdict(dict)
        d = df.groupby(by=[col, col_cls_name]).count().iloc[:, 0]

        for key in d.keys():
            attr, cls = key
            S[attr][cls] = d[key]

        gain = node_entropy
        for key in S:
            cls_freq = S[key]
            attr_total = d.loc[key].sum()
            total = d.sum()
            S[key]['entropy'] = self.__entropy_of(cls_freq)
            gain -= (S[key]['entropy'] * attr_total / total)

        return S, gain

    def __split(self, node: dict, attrs: list):
        """
        Splits the node on the attribute with the highest gain value.
        Uses Entropy for gain calculation.
        :param node: node to split
        :param attrs: allowed attributes to split on
        """

        col_cls_name = list(self.__cls.keys())[0]
        df = self.__data.loc[node['data']]

        max_gain = -1
        max_S = None
        split_col = None
        for col in attrs:
            S, gain = self.__gain_of(node['S']['entropy'], col, col_cls_name, df)

            if max_gain < gain:
                max_gain = gain
                max_S = S
                split_col = col

        node['split']['attr'] = split_col
        node['split']['gain'] = max_gain
        node['branches'] = dict()

        for key in max_S:
            cls_freq = max_S[key].copy()
            cls_freq.pop('entropy')
            n = {
                'data': [i for i in df[df[split_col] == key].index],
                'S': {
                    'cls_freq': cls_freq,
                    'entropy': max_S[key]['entropy']
                },
                'split': {'attr': None, 'gain': None},
                'branches': None,
                'label': f'{split_col}={key}'
            }
            node['branches'][key] = n

    def __make_leaf(self, node: dict):
        """
        Marks a node as a leaf, by assigning the majority class as the label.
        :param node:
        :return:
        """
        max_freq = -1
        label = ''
        for key in node['S']['cls_freq']:
            freq = node['S']['cls_freq'][key]
            if max_freq < freq:
                max_freq = freq
                label = key

        node['label'] = label

    def __tree_at(self, node: dict, attrs: list):
        """
        Uses Depth-first-search approach to build the tree recursively using ID3 algorithm.
        :param node: the current node to split
        :return:
        """
        if node['S']['entropy'] == 0 or len(attrs) == 0:
            self.__make_leaf(node)
            return

        self.__split(node, attrs)

        # TODO: remove split attr from list and recursive call

    def train(self):
        cls_col_name = list(self.__cls.keys())[0]
        attr_names = list(self.__attrs.keys())

        self.__tree = {
            'data': [i for i in range(len(self.__data))],
            'S': {
                'cls_freq': self.__data.groupby(cls_col_name).count()[attr_names[0]].to_dict()
            },
            'split': {'attr': None, 'gain': None},
            'branches': None,
            'label': 'Root'
        }

        self.__tree['S']['entropy'] = self.__entropy_of(self.__tree['S']['cls_freq'])

        self.__tree_at(node=self.__tree, attrs=attr_names)

    def plot_tree(self):
        print(self.__tree)
        net = Network()
        net.height = '100%'
        net.width = '100%'

        node = self.__tree
        node['id'] = 1
        title = f"{node['S']['cls_freq']} || {node['S']['entropy']:.3f} || data: {[i+1 for i in node['data']]}"
        net.add_node(1, label=node['label'], title=title)

        q = Queue()
        q.put(node)

        self.__bfs_plot(q, net)

        net.show('tree.html')

    def __bfs_plot(self, q: Queue, net: Network):
        i = 1
        while not q.empty():
            node = q.get()
            uid = node['id']

            if node['branches'] is not None:
                for key in node['branches'].keys():
                    n = node['branches'][key]
                    n['id'] = uid + i
                    title = f"{n['S']['cls_freq']} || {n['S']['entropy']:.3f} || data: {[i+1 for i in n['data']]}"
                    net.add_node(uid + i, label=n['label'], title=title)
                    net.add_edge(uid, uid + i)
                    i += 1
                    q.put(n)




def main():
    data = []
    attrs = defaultdict(list)
    with open('fishing.data', mode='r') as fin:
        for line in fin:
            line = line.strip()

            if line.startswith("#attributes") or line.startswith("#target"):
                for l in fin:
                    l = l.strip()
                    if len(l) == 0:
                        break
                    parts = l.split(":")
                    attrs[parts[0][1:]] = parts[1].strip().split(",")

            elif line.startswith("#data"):
                for l in fin:
                    l = l.strip()
                    if len(l) == 0:
                        break
                    parts = l.split(",")
                    data.append(parts)

    df = pd.DataFrame(data)
    df.columns = list(attrs.keys())
    print(df.info())

    dt = DecisionTree(df, attrs)
    dt.train()
    dt.plot_tree()
