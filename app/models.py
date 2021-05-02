from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pythainlp.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import deepcut


class DataTools:
    @staticmethod
    def sample_output(probs, temperature=1.0):
        p_pow_t = probs ** (1/temperature)
        sum_p = np.sum(p_pow_t)
        fT = p_pow_t/sum_p
        sample_index = np.argmax(fT)
        
        x = np.random.uniform()
        c = 0
        for i, p in enumerate(list(fT)) :
            c += p
            if c > x :
                return i
        return sample_index

    @staticmethod
    def temperature_sampling_decode(seed_text, max_gen_length, model_obj, temperature):
        current_text = [model_obj.word_to_index[w] for w in  word_tokenize(seed_text , engine="newmm")]
        probs = []
        reach_eos = False
        for _ in range(max_gen_length):
            if current_text[-1] == model_obj.word_to_index["<EOS>"] :
                break
            padded_seq_text = pad_sequences([current_text], maxlen=10,padding='pre')
            pred = model_obj.model.predict(padded_seq_text)
            output_token = DataTools.sample_output(pred[0], temperature)
            output_word = model_obj.index_to_word[output_token]
            current_text.append(model_obj.word_to_index[output_word])
        return "".join([model_obj.index_to_word[w] for w in current_text[:-1]])

    @staticmethod
    def quote_sample_output(probs, temperature=1.0):
        ps = np.array(probs)

        ft_p = np.power(ps, 1/temperature) 
        ft_p = ft_p/np.sum(ft_p)
        cu_ft_p = np.cumsum(ft_p)

        sampled = np.random.rand()

        return np.argmax(cu_ft_p > sampled)

    @staticmethod
    def gen_sent(seed_text, model_obj):
        sentence = deepcut.tokenize(seed_text)
        gen_max_length = 30
        while True:
            x = pad_sequences([[model_obj.word_to_index[word] for word in sentence]], maxlen=5)
            probs = model_obj.model.predict(x)
            index = DataTools.quote_sample_output(probs, 2)
            pred_word = model_obj.index_to_word[index]
            sentence.append(pred_word)
            if pred_word == '<END>' or len(sentence) >= gen_max_length:
                break
        return ''.join(sentence[:-1])


class DinosaturdayModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.vocab_path = '../model_files/dinosaturday_vocab.pkl'
        self.word_to_index = dict()
        self.index_to_word = dict()
        self.get_word_and_index()

    def get_word_and_index(self):
        with open(self.vocab_path, "rb") as f:
            dinosaturday_vocab_list = pickle.load(f)
            self.word_to_index = dict([(v,i) for (i,v) in enumerate(dinosaturday_vocab_list)])
            self.index_to_word = dict([(v,k) for (k,v) in self.word_to_index.items()])


class T047Model:
    def __init__(self):
        self.model = load_model('../model_files/t047_gen_LSTM_final.h5')
        self.vocab_path = '../model_files/t047_vocab.pkl'
        self.word_to_index = dict()
        self.index_to_word = dict()
        self.get_word_and_index()

    def get_word_and_index(self):
        with open(self.vocab_path, "rb") as f:
            t047_vocab_list = pickle.load(f)
            self.word_to_index = dict([(v,i) for (i,v) in enumerate(t047_vocab_list)])
            self.index_to_word = dict([(v,k) for (k,v) in self.word_to_index.items()])


class MixModel:
    def __init__(self):
        self.model = load_model('../model_files/dino_t047_mix_gen_LSTM_final.h5')
        self.vocab_path = '../model_files/mix_dino_t047_vocab.pkl'
        self.word_to_index = dict()
        self.index_to_word = dict()
        self.get_word_and_index()

    def get_word_and_index(self):
        with open(self.vocab_path, "rb") as f:
            mix_vocab_list = pickle.load(f)
            self.word_to_index = dict([(v,i) for (i,v) in enumerate(mix_vocab_list)])
            self.index_to_word = dict([(v,k) for (k,v) in self.word_to_index.items()])


class QuoteModel:
    def __init__(self):
        index_to_word, word_to_index, vocab_size = self.get_word_and_index()

        self.model = load_model('../model_files/kamkom_97e.keras')
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def get_word_and_index(self):
        with open('../model_files/prepros.pickle','rb') as f:
            (index_to_word, word_to_index, vocab_size) = pickle.load(f)
        return index_to_word, word_to_index, vocab_size


DinoModel_obj = DinosaturdayModel('../model_files/dinosad_gen_LSTM_final.h5')
T047Model_obj = T047Model()
MixModel_obj = MixModel()
QuoteModel_obj = QuoteModel()
DinoV2Model_obj = DinosaturdayModel('../model_files/dinosad_v2.h5')
