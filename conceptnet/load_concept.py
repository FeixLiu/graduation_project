from hyper import Hyperparameters as hp
import sys
import json
from nltk.corpus import stopwords


class Load_concept():
    def __init__(self, origin_path, filter_path):
        self._stop_words = set(stopwords.words('english'))
        self._origin_path = origin_path
        self._filter_path = filter_path
        self._load_conceptnet()

    def _get_lan_and_w(self, arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]

    def _load_conceptnet(self):
        index = 0.
        writer = open(self._filter_path, 'w', encoding='utf-8')
        with open(self._origin_path, 'r', encoding='utf-8') as file:
            for line in file:
                index += 1.0
                if index % 10000 == 0:
                    print(index / 32755210, file=sys.stderr)
                fs = line.split('\t')
                relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]
                lan1, w1 = self._get_lan_and_w(arg1)
                flag = 1
                if lan1 != 'en':
                    continue
                lan2, w2 = self._get_lan_and_w(arg2)
                if lan2 != 'en':
                    continue
                obj = json.loads(fs[-1])
                if obj['weight'] < 1:
                    continue
                if w1 in self._stop_words or w2 in self._stop_words:
                    continue
                writer.write('%s %s %s\n' % (relation, w1, w2))
        writer.close()

lc = Load_concept(origin_path=hp.conceptNet_path, filter_path=hp.conceptNet_filter)
