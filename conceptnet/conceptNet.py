import sys
import nltk


class ConcetNet():
    def __init__(self, path):
        self._path = path
        self._data = {}
        for triple in open(self._path, 'r', encoding='utf-8'):
            triple = triple.replace('\n', '')
            relation, arg1, arg2 = triple.split(' ')
            try:
                self._data[arg1][arg2] = relation
            except KeyError:
                self._data[arg1] = {}
                self._data[arg1][arg2] = relation
            try:
                self._data[arg2][arg1] = relation
            except KeyError:
                self._data[arg2] = {}
                self._data[arg2][arg1] = relation
        print('Loaded concept net', file=sys.stderr)

    def _get_relation(self, word1, word2):
        word1 = '_'.join(word1.lower().split())
        word2 = '_'.join(word2.lower().split())
        if not word1 in self._data:
            return '<NULL>'
        return self._data[word1].get(word2, '<NULL>')

    def _get_extension(self, refa_word, refb_word):
        for i in range(len(refa_word)):
            for word in refb_word:
                ref = refa_word[i]
                if ref in refb_word:
                    continue
                relation = self._get_relation(ref, word)
                if relation != '<NULL>' and ref != word:
                    adding = '(' + ref + ' ' + relation + ' ' + word + ')'
                    refa_word.insert(i + 1, adding)
                    break
                ref = ' '.join(refa_word[i : i + 1])
                relation = self._get_relation(ref, word)
                if relation != '<NULL>' and ref != word:
                    adding = '(' + ref + ' ' + relation + ' ' + word + ')'
                    refa_word.insert(i + 2, adding)
                    break
                ref = ' '.join(refa_word[i : i + 2])
                relation = self._get_relation(ref, word)
                if relation != '<NULL>' and ref != word:
                    adding = '(' + ref + ' ' + relation + ' ' + word + ')'
                    refa_word.insert(i + 3, adding)
                    break
        return refa_word

    def a_and_b_relation(self, refa, refb):
        #refa_extend = ''
        refb_extend = ''
        refa_word = nltk.word_tokenize(refa)
        refb_word = nltk.word_tokenize(refb)
        #refa_word = self._get_extension(refa_word, refb_word)
        refb_word = self._get_extension(refb_word, refa_word)
        '''
        for word in refa_word:
            if len(word) > 1 and len(refa_extend) != 0:
                refa_extend += ' '
            refa_extend += word
        '''
        for word in refb_word:
            if len(word) > 1 and len(refb_extend) != 0:
                refb_extend += ' '
            refb_extend += word
        #return refa_extend.strip(),
        return refb_extend.strip()
