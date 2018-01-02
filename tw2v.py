import os
import sys
import logging
import json
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import word2vec
import gensim
translator = str.maketrans('', '', string.punctuation)
from string import punctuation
#import tsne # on ubuntu: sudo apt-get install libblas-dev libatlas-base-dev


def train(trans, model_file, min_count, size):
    file = 'bibles-master/%s/%s.json' % (trans, trans)
    with open(file, 'r') as f:
        bible = json.load(f)

    corpus = ''
    for book in bible:
        for chapter in bible[book]:
            for verse in bible[book][chapter]:
                # print(bible[book][chapter][verse])
                corpus += bible[book][chapter][verse].lower() + ' '

    sents = []
    for sent in sent_tokenize(corpus):
        sent = sent.translate(translator)
        sents.append(sent.split())

    print('Training model')

    model = word2vec.Word2Vec(sents, min_count=min_count, size=size, workers=3)
    # model.save(model_file)
    model.wv.save_word2vec_format(model_file,binary=True)
    return model


def analogies(model, analogies):
    for analogy in analogies:
        print('%s is to %s as %s is tos: ' % (analogy[0], analogy[1], analogy[2]))
        # matches = model.most_similar(positive=['woman', 'king'], negative=['man'])
        matches = model.most_similar(positive=[analogy[1], analogy[2]], negative=[analogy[0]])
        for match in matches:
            print('\t%s\t\t\t%3.2f' % (match[0], match[1]))

def similar(model, words):
    for similar_word in words:
        print('Word similar to %s: ' % similar_word)
        matches = model.most_similar([similar_word])
        for match in matches:
            print('\t%s\t\t\t%3.2f' % (match[0], match[1]))

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    trans = sys.argv[1]
    min_count = int(sys.argv[2])
    size = int(sys.argv[3])
    model_file = 'models/%s_%s_%s.bin' % (trans, min_count, size)

    if os.path.isfile(model_file):
        print('Model already exists: %s  No training required.' % model_file)
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file,binary=True)
        # model = word2vec.Word2Vec.load(model_file)
        # model = word2vec.Word2Vec.load_word2vec_format(model_file,binary=True)
        # model = gensim.models.Word2Vec.load(model_file)
        # model = gensim.models.KeyedVectors.load_word2vec_format(model_file,binary=True)
        # model = word2vec.Word2Vec.load_word2vec_format(model_file)
        # .load_word2vec_format(model_file)
    else:
        print('Model %s does not exist.' % model_file)
        model = train(trans, model_file, min_count, size)

    # sim_words = ['god', 'jesus', 'david', 'man', 'faith', 'sin', 'sinner', 'love', 'righteousness', 'angel', 'demon', 'prophesy',
    #              'water', 'israel', 'babylon', 'pharisee', 'life', 'salvation', 'moses', 'confess', 'world', 'virgin', 'abraham',
    #              'abram','apostles','matthew','chemosh']
    sim_words = ['hagar']
    similar(model, sim_words)

    # analogy_words = [['jesus', 'salvation', 'man'],['water', 'life', 'jesus'],['jesus', 'god', 'abraham']
    #     ,['abraham', 'isaac', 'father'],['sin','god','man'],['abraham','sarah','joseph'],['lord','shall','man']
    #     ,['death','jesus','resurrection'],['law','god','salvation'],['salvation','gift','sin'],['lamb','sacrifice','jesus']]
    analogy_words = [['jonah','fish','daniel']]
    analogies(model, analogy_words)

    print('done')
