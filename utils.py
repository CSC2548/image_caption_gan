from build_vocab import Vocabulary
import pickle
import pdb

vocab_path = './data/birds_vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)


def print_sentence(sampled_ids):
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break

    sentence = ' '.join(sampled_caption)
    print(sentence)

