from collections import defaultdict
from nltk import corpus, word_tokenize
import math
import regex


class NB:
    __class_col = None
    __attribute_names = None
    __prior_counts = None
    __priors = None
    __likelihood_counts = None
    __likelihoods = None
    __total_instances = None

    def __init__(self):
        self.__class_col = None
        self.__attribute_names = []

        self.__prior_counts = defaultdict(int)
        self.__priors = defaultdict(float)

        self.__likelihood_counts = defaultdict(float)
        self.__likelihoods = defaultdict(float)

        self.__total_instances = 0

    def train(self, class_at_column=1, header=True, filepath='', delimeter=','):
        """
        Trying to create a Generalized function for training with any
        type of textual dataset. .csv/.tsv and with/without header in
        file support.

        :param class_at_column: The column that contains the class.
        All other columns are considered attributes.
        Default class at 1st column.
        :param header: If the attribute names present at the first row of the file.
        Default is True.
        :param filepath: The full/relative to .py file location of data set path.
        :param delimeter: The attribute separator for each line.
        :return: None
        """
        self.__class_col = class_at_column - 1

        with open(filepath, mode='r', encoding='utf8') as fin:
            """
            Take the first line and based on header(True/False)
            Create Prior, Likelihood and Variable name lists
            """

            lines = fin.readlines()
            parts = lines[0].strip().split(delimeter)

            if header:
                i = 1
                for col in parts:
                    if col in self.__attribute_names:
                        raise IOError("Column Names not Unique.")
                    self.__attribute_names.append(col.strip())
                    i += 1

            """
            Parse the full file and count all data
            """
            for line in lines:
                self.__total_instances += 1
                parts = line.strip().split(delimeter)

                cls = parts[self.__class_col].strip()

                for i in range(0, len(parts), 1):
                    if i == self.__class_col:
                        self.__prior_counts[cls] += 1
                        continue

                    self.__likelihood_counts[f'{parts[i].strip()}|{cls}'] += 1

        for cls in self.__prior_counts:
            self.__priors[cls] = self.__prior_counts[cls] / self.__total_instances

        for key in self.__likelihood_counts:
            cls = key.split('|')[-1]
            self.__likelihoods[key] = self.__likelihood_counts[key] / self.__prior_counts[cls]

        """
        For tabular data, if any attribute is previously unseen, we assign probability to 0.
        For textual data, if any word is previously unseen, we assign a 
        probability using - 1/(total #word + size of vocab)
        
        This helps keep the test code same for both tabular and textual data.
        """
        self.__likelihoods['_any_|_any_'] = 0

    def test(self, new_instance=None, no_print=False):
        if new_instance is None:
            raise AttributeError(f"Invalid New Instance.")

        if not no_print: print('\n\n**********************************\n')
        if not no_print: print(f'Training Instances: {self.__total_instances}\n')
        if not no_print: print('Prior Probabilities')

        for cls in self.__priors:
            if not no_print: print(f'#{cls}: {self.__prior_counts[cls]}\t\t', end='')
            if not no_print: print(f'P({cls}): {self.__priors[cls]}\t\t', end='')
            if not no_print: print(f'log(P({cls})): {math.log(self.__priors[cls])}')

        if not no_print: print('\n**********************************\n\n')
        if not no_print: print(f'New instance: {str(new_instance)}\n')

        posteriors = {}

        for cls in self.__priors:
            posteriors[cls] = self.__priors[cls]
            posteriors[f'log({cls})'] = math.log(self.__priors[cls])

            for attr in new_instance:
                key = f'{attr}|{cls}'

                if key in self.__likelihoods:
                    lkh = self.__likelihoods[key]
                else:
                    lkh = self.__likelihoods['_any_|_any_']

                posteriors[cls] *= lkh
                posteriors[f'log({cls})'] += math.log(lkh)
                if not no_print: print(f'P({attr}|{cls}): {lkh}\t', end='')
                if not no_print: print(f'log(P({attr}|{cls})): {math.log(lkh)}\t\t', end='')

            if not no_print: print('')

        if not no_print: print('\nClass probabilities\n')
        max_prob = -999999999
        verdict = ''
        total = 0
        for cls in posteriors:
            if not cls.startswith('log'):
                total += posteriors[cls]
                if not no_print: print(f'{cls}: {posteriors[cls]}\t\t', end='')
                continue

            if not no_print: print(f'{cls}: {posteriors[cls]}')

            if max_prob < posteriors[cls]:
                max_prob = posteriors[cls]
                verdict = cls

        if not no_print: print(f'\nClassify: {verdict[4:-1]}')

        if total > 0:
            if not no_print: print(f'Conditional Probability for class "{verdict[4:-1]}" : {posteriors[verdict[4:-1]] * 100 / total:.2f}%\n\n')

        return verdict[4:-1]

    def __fix_likelihoods_for_text(self):
        size_of_vocab = len(self.__likelihood_counts)
        total_word_count = 0
        for value in self.__likelihood_counts.values():
            total_word_count += value

        for key in self.__likelihoods:
            self.__likelihoods[key] = (self.__likelihood_counts[key] + 1) / (total_word_count + size_of_vocab)

        self.__likelihoods['_any_|_any_'] = 1 / (total_word_count + size_of_vocab)

    def train_for_text(self, datafile):
        lines = []
        max_len = 0
        # total_instances_by_cls = defaultdict(int)
        class_at_column = 1
        stop_words = set(corpus.stopwords.words('english'))

        with open(datafile, mode='r', encoding='utf8') as fin:
            for line in fin:
                # case-insensitive prediction
                line = line.strip().lower()

                # remove punctuations
                line = regex.sub(r"[.,?!:;()]|\d+", "", line)

                # removing stopwords
                words = word_tokenize(line)
                valid_words = []
                for w in words:
                    if w not in stop_words:
                        valid_words.append(w)

                # if max_len < len(valid_words):
                #     max_len = len(valid_words)

                # line = line.replace(',', ' ')
                # valid_words = line.split()

                # total_instances_by_cls[valid_words[class_at_column - 1]] += 1

                lines.append(valid_words)

        newfile = 'processed_texts.csv'
        with open(newfile, mode='w', encoding='utf8') as fout:

            # for cls in total_instances_by_cls:
            #     fout.write(f'#{cls}_{total_instances_by_cls[cls]}\n')

            for line in lines:
                length = len(line)
                for w in line:
                    fout.write(f'{w},')

                for i in range(0, max_len - length, 1):
                    fout.write(',')

                fout.write('\n')

        self.train(class_at_column=class_at_column, header=False, filepath=newfile, delimeter=',')
        self.__fix_likelihoods_for_text()
