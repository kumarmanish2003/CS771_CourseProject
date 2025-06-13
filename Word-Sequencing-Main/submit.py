import numpy as np
import random

class TextTree:
    def _init_(self, min_leaf_size=1, max_depth=6):
        self.root = None
        self.words = None
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth

    def train(self, words, verbose=False):
        self.words = words
        self.root = TextNode(depth=0, parent=None)
        self.root.train(all_words=self.words, indices=np.arange(len(self.words)), 
                        min_leaf_size=self.min_leaf_size, max_depth=self.max_depth, verbose=verbose)

    def predict(self, bigrams, max_results=5):
        return self.root.predict(bigrams, max_results)

class TextNode:
    def _init_(self, depth, parent):
        self.depth = depth
        self.parent = parent
        self.words = None
        self.indices = None
        self.children = {}
        self.is_leaf = True
        self.split_criterion = None
        self.history = []

    def generate_bigram(self):
        return chr(ord('a') + random.randint(0, 25)) + chr(ord('a') + random.randint(0, 25))

    def get_bigrams(self, word, limit=5):
        bigrams = [''.join(bg) for bg in zip(word, word[1:])]
        return tuple(sorted(set(bigrams)))[:limit]

    def handle_leaf(self, all_words, indices, history, verbose):
        self.indices = indices

    def handle_split(self, all_words, indices, history, verbose):
        split_criterion = self.generate_bigram()
        split_groups = {True: [], False: []}
        for idx in indices:
            bigrams = self.get_bigrams(all_words[idx])
            split_groups[split_criterion in bigrams].append(idx)
        return split_criterion, split_groups

    def train(self, all_words, indices, min_leaf_size, max_depth, indent="    ", verbose=False):
        self.words = all_words
        self.indices = indices
        if len(indices) <= min_leaf_size or self.depth >= max_depth:
            self.is_leaf = True
            self.handle_leaf(self.words, self.indices, self.history, verbose)
        else:
            self.is_leaf = False
            self.split_criterion, split_groups = self.handle_split(self.words, self.indices, self.history, verbose)
            for condition, group in split_groups.items():
                self.children[condition] = TextNode(depth=self.depth + 1, parent=self)
                self.children[condition].history = self.history + [self.split_criterion]
                self.children[condition].train(self.words, group, min_leaf_size, max_depth, indent, verbose)

    def predict(self, bigrams, max_results=5):
        node = self
        results = []

        def has_bigrams(word, bigrams):
            word_bigrams = self.get_bigrams(word)
            return all(bg in word_bigrams for bg in bigrams)

        while len(results) < max_results and not node.is_leaf:
            node = node.children.get(any(bg in bigrams for bg in node.get_bigrams(self.words[node.indices[0]])), node)

        for idx in node.indices:
            word = self.words[idx]
            if has_bigrams(word, bigrams):
                results.append(word)
                if len(results) == max_results:
                    break

        return results

################################
# Non Editable Region Starting #
################################
def my_fit(word_list):
################################
#  Non Editable Region Ending  #
################################
    tree = TextTree(min_leaf_size=1, max_depth=6)
    tree.train(word_list)
    return tree

################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
################################
#  Non Editable Region Ending  #
################################
    return model.predict(bigram_list)
