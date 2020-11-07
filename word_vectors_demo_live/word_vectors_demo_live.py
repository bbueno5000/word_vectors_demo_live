"""
DOCSTRING
"""
import codecs
import gensim.models.word2vec as word2vec
import glob
import logging
import multiprocessing
import nltk
import os
import pandas
import pprint
import re
import seaborn
import sklearn.manifold

class Demo:
    """
    DOCSTRING
    """
    def __call__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        book_filenames = sorted(glob.glob('/*.txt'))
        print('Found books:')
        print(book_filenames)
        corpus_raw = u""
        for book_filename in book_filenames:
            print('Reading:{}'.format(book_filename))
            with codecs.open(book_filename, 'r', 'utf-8') as book_file:
                corpus_raw += book_file.read()
            print('Corpus is now {0} characters long'.format(len(corpus_raw)))
            print()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(corpus_raw)
        sentences = list()
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.sentence_to_wordlist(raw_sentence))
        print(raw_sentences[5])
        print(self.sentence_to_wordlist(raw_sentences[5]))
        token_count = sum([len(sentence) for sentence in sentences])
        print('The book corpus contains {0:,} tokens'.format(token_count))
        num_features = 300
        min_word_count = 3
        num_workers = multiprocessing.cpu_count()
        context_size = 7
        downsampling = 1e-3
        seed = 1
        thrones2vec = word2vec.Word2Vec(
            sg=1,
            seed=seed,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context_size,
            sample=downsampling)
        thrones2vec.build_vocab(sentences)
        print('Word2Vec vocabulary length:', len(thrones2vec.vocab))
        thrones2vec.train(sentences)
        if not os.path.exists('trained'):
            os.makedirs('trained')
        thrones2vec.save(os.path.join('trained', 'thrones2vec.w2v'))
        thrones2vec = word2vec.Word2Vec.load(os.path.join('trained', 'thrones2vec.w2v'))
        tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
        all_word_vectors_matrix = thrones2vec.syn0
        all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
        points = pandas.DataFrame([
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[thrones2vec.vocab[word].index])
                for word in thrones2vec.vocab]], columns=['word', 'x', 'y'])
        points.head(10)
        seaborn.set_context('poster')
        points.plot.scatter('x', 'y', s=10, figsize=(20, 12))
        self.plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))
        self.plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))
        thrones2vec.most_similar('Stark')
        thrones2vec.most_similar('Aerys')
        thrones2vec.most_similar('direwolf')
        self.nearest_similarity_cosmul('Stark', 'Winterfell', 'Riverrun')
        self.nearest_similarity_cosmul('Jaime', 'sword', 'wine')
        self.nearest_similarity_cosmul('Arya', 'Nymeria', 'dragons')

        def nearest_similarity_cosmul(self, start1, end1, end2):
            """
            DOCSTRING
            """
            similarities = thrones2vec.most_similar_cosmul(
                positive=[end2, start1],
                negative=[end1])
            start2 = similarities[0][0]
            print('{start1} is related to {end1}, as {start2} is related to {end2}'.format(**locals()))
            return start2

        def plot_region(self, x_bounds, y_bounds):
            """
            DOCSTRING
            """
            slice = points[
                (x_bounds[0] <= points.x) &
                (points.x <= x_bounds[1]) & 
                (y_bounds[0] <= points.y) &
                (points.y <= y_bounds[1])]
            ax = slice.plot.scatter('x', 'y', s=35, figsize=(10, 8))
            for i, point in slice.iterrows():
                ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

        def sentence_to_wordlist(self, raw):
            """
            DOCSTRING
            """
            clean = re.sub("[^a-zA-Z]", ' ', raw)
            words = clean.split()
            return words

class Thrones2Vec:
    """
    DOCSTRING
    """
    def __call__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        nltk.download('punkt')
        nltk.download('stopwords')
        book_filenames = sorted(glob.glob('data/clean/*.txt'))
        print('Found books:')
        book_filenames
        corpus_raw = u""
        for book_filename in book_filenames:
            print('Reading:{}'.format(book_filename))
            with codecs.open(book_filename, 'r', 'utf-8') as book_file:
                corpus_raw += book_file.read()
            print('Corpus is now {0} characters long'.format(len(corpus_raw)))
            print()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(corpus_raw)
        sentences = list()
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.sentence_to_wordlist(raw_sentence))
        print(raw_sentences[5])
        print(self.sentence_to_wordlist(raw_sentences[5]))
        token_count = sum([len(sentence) for sentence in sentences])
        print('The book corpus contains {0:,} tokens'.format(token_count))
        num_features = 300
        min_word_count = 3
        num_workers = multiprocessing.cpu_count()
        context_size = 7
        downsampling = 1e-3
        seed = 1
        thrones2vec = word2vec.Word2Vec(
            sg=1,
            seed=seed,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context_size,
            sample=downsampling)
        thrones2vec.build_vocab(sentences)
        print('Word2Vec vocabulary length:', len(thrones2vec.vocab))
        thrones2vec.train(sentences)
        if not os.path.exists('trained'):
            os.makedirs('trained')
        thrones2vec.save(os.path.join('trained', 'thrones2vec.w2v'))
        thrones2vec = word2vec.Word2Vec.load(os.path.join('trained', 'thrones2vec.w2v'))
        tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
        all_word_vectors_matrix = thrones2vec.syn0
        all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
        points = pandas.DataFrame([
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[thrones2vec.vocab[word].index])
                for word in thrones2vec.vocab]], columns=['word', 'x', 'y'])
        points.head(10)
        seaborn.set_context('poster')
        points.plot.scatter('x', 'y', s=10, figsize=(20, 12))
        self.plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))
        self.plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))
        thrones2vec.most_similar('Stark')
        thrones2vec.most_similar('Aerys')
        thrones2vec.most_similar('direwolf')
        self.nearest_similarity_cosmul('Stark', 'Winterfell', 'Riverrun')
        self.nearest_similarity_cosmul('Jaime', 'sword', 'wine')
        self.nearest_similarity_cosmul('Arya', 'Nymeria', 'dragons')

    def nearest_similarity_cosmul(self, start1, end1, end2):
        """
        DOCSTRING
        """
        similarities = thrones2vec.most_similar_cosmul(
            positive=[end2, start1], negative=[end1])
        start2 = similarities[0][0]
        print('{start1} is related to {end1}, as {start2} is related to {end2}'.format(**locals()))
        return start2

    def plot_region(self, x_bounds, y_bounds):
        """
        DOCSTRING
        """
        slice = points[
            (x_bounds[0] <= points.x) &
            (points.x <= x_bounds[1]) &
            (y_bounds[0] <= points.y) &
            (points.y <= y_bounds[1])]
        ax = slice.plot.scatter('x', 'y', s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

    def sentence_to_wordlist(self, raw):
        """
        Convert into a list of words.
        """
        clean = re.sub("[^a-zA-Z]",' ', raw)
        words = clean.split()
        return words

if __name__ == '__main__':
    demo = Demo()
    demo()
