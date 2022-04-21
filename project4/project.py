from perceptron import Perceptron
from pandas import DataFrame


def run_perceptron():
    print('\n\n********************** AND **********************\n\n')
    p = Perceptron(2)
    df = DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = [0, 0, 0, 1]
    p.train(df, targets)
    print('\n********** Classify **********')
    p.classify([0, 1])

    print('\n\n********************** OR **********************\n\n')
    p = Perceptron(2)
    df = DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = [0, 1, 1, 1]
    p.train(df, targets)
    print('\n********** Classify **********')
    p.classify([0, 1])

    print('\n\n********************** NAND **********************\n\n')
    p = Perceptron(2)
    df = DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = [1, 1, 1, 0]
    p.train(df, targets)
    print('\n********** Classify **********')
    p.classify([0, 1])

    print('\n\n********************** NOR **********************\n\n')
    p = Perceptron(2)
    df = DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = [1, 0, 0, 0]
    p.train(df, targets)
    print('\n********** Classify **********')
    p.classify([0, 1])


if __name__ == '__main__':
    run_perceptron()
