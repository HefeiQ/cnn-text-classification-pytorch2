import re
import os
import random
from torchtext import data


class MR(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        def clean_str(string):
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            examples = []
            with open(os.path.join('data', path), encoding="utf-8", errors="ignore") as f:
                for line in f.readlines():
                    #print(line)
                    if line[-2] == '0':
                        #print(line[:line.find('|')], '----negative')
                        examples += [
                            data.Example.fromlist([line[:line.find('|')], 'negative'], fields)]
                    else:
                        #print(line[:line.find('|')], '----positive')
                        examples += [
                            data.Example.fromlist([line[:line.find('|')], 'positive'], fields)]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True, **kwargs):

        example1 = cls(text_field, label_field, path='train.fmt', **kwargs).examples
        example2 = cls(text_field, label_field, path='dev.fmt', **kwargs).examples
        example3 = cls(text_field, label_field, path='test.fmt', **kwargs).examples
        if shuffle:
            random.shuffle(example1)
            random.shuffle(example2)
            random.shuffle(example3)

        return (cls(text_field, label_field, examples=example1),
                cls(text_field, label_field, examples=example2),
                cls(text_field, label_field, examples=example3))
