import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import torch
import torch.nn as nn
import torch.utils.data
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import gc
import random
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
import time
import math
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
warnings.filterwarnings("ignore")
from gensim.models import KeyedVectors
import re
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU, CuDNNLSTM, add, Conv2D, Reshape
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
import pickle
from keras.losses import binary_crossentropy
import gc
from keras.models import load_model
print('Модули загружены')

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"_":" ", "`":" "}
swear_words = [
    ' 4r5e ',
    ' 5h1t ',
    ' 5hit ',
    ' a55 ',
    ' anal ',
    ' anus ',
    ' ar5e ',
    ' arrse ',
    ' arse ',
    ' ass ',
    ' ass-fucker ',
    ' asses ',
    ' assfucker ',
    ' assfukka ',
    ' asshole ',
    ' assholes ',
    ' asswhole ',
    ' a_s_s ',
    ' b!tch ',
    ' b00bs ',
    ' b17ch ',
    ' b1tch ',
    ' ballbag ',
    ' balls ',
    ' ballsack ',
    ' bastard ',
    ' beastial ',
    ' beastiality ',
    ' bellend ',
    ' bestial ',
    ' bestiality ',
    ' biatch ',
    ' bitch ',
    ' bitcher ',
    ' bitchers ',
    ' bitches ',
    ' bitchin ',
    ' bitching ',
    ' bloody ',
    ' blow job ',
    ' blowjob ',
    ' blowjobs ',
    ' boiolas ',
    ' bollock ',
    ' bollok ',
    ' boner ',
    ' boob ',
    ' boobs ',
    ' booobs ',
    ' boooobs ',
    ' booooobs ',
    ' booooooobs ',
    ' breasts ',
    ' buceta ',
    ' bugger ',
    ' bum ',
    ' bunny fucker ',
    ' butt ',
    ' butthole ',
    ' buttmuch ',
    ' buttplug ',
    ' c0ck ',
    ' c0cksucker ',
    ' carpet muncher ',
    ' cawk ',
    ' chink ',
    ' cipa ',
    ' cl1t ',
    ' clit ',
    ' clitoris ',
    ' clits ',
    ' cnut ',
    ' cock ',
    ' cock-sucker ',
    ' cockface ',
    ' cockhead ',
    ' cockmunch ',
    ' cockmuncher ',
    ' cocks ',
    ' cocksuck ',
    ' cocksucked ',
    ' cocksucker ',
    ' cocksucking ',
    ' cocksucks ',
    ' cocksuka ',
    ' cocksukka ',
    ' cok ',
    ' cokmuncher ',
    ' coksucka ',
    ' coon ',
    ' cox ',
    ' crap ',
    ' cum ',
    ' cummer ',
    ' cumming ',
    ' cums ',
    ' cumshot ',
    ' cunilingus ',
    ' cunillingus ',
    ' cunnilingus ',
    ' cunt ',
    ' cuntlick ',
    ' cuntlicker ',
    ' cuntlicking ',
    ' cunts ',
    ' cyalis ',
    ' cyberfuc ',
    ' cyberfuck ',
    ' cyberfucked ',
    ' cyberfucker ',
    ' cyberfuckers ',
    ' cyberfucking ',
    ' d1ck ',
    ' damn ',
    ' dick ',
    ' dickhead ',
    ' dildo ',
    ' dildos ',
    ' dink ',
    ' dinks ',
    ' dirsa ',
    ' dlck ',
    ' dog-fucker ',
    ' doggin ',
    ' dogging ',
    ' donkeyribber ',
    ' doosh ',
    ' duche ',
    ' dyke ',
    ' ejaculate ',
    ' ejaculated ',
    ' ejaculates ',
    ' ejaculating ',
    ' ejaculatings ',
    ' ejaculation ',
    ' ejakulate ',
    ' f u c k ',
    ' f u c k e r ',
    ' f4nny ',
    ' fag ',
    ' fagging ',
    ' faggitt ',
    ' faggot ',
    ' faggs ',
    ' fagot ',
    ' fagots ',
    ' fags ',
    ' fanny ',
    ' fannyflaps ',
    ' fannyfucker ',
    ' fanyy ',
    ' fatass ',
    ' fcuk ',
    ' fcuker ',
    ' fcuking ',
    ' feck ',
    ' fecker ',
    ' felching ',
    ' fellate ',
    ' fellatio ',
    ' fingerfuck ',
    ' fingerfucked ',
    ' fingerfucker ',
    ' fingerfuckers ',
    ' fingerfucking ',
    ' fingerfucks ',
    ' fistfuck ',
    ' fistfucked ',
    ' fistfucker ',
    ' fistfuckers ',
    ' fistfucking ',
    ' fistfuckings ',
    ' fistfucks ',
    ' flange ',
    ' fook ',
    ' fooker ',
    ' fuck ',
    ' fucka ',
    ' fucked ',
    ' fucker ',
    ' fuckers ',
    ' fuckhead ',
    ' fuckheads ',
    ' fuckin ',
    ' fucking ',
    ' fuckings ',
    ' fuckingshitmotherfucker ',
    ' fuckme ',
    ' fucks ',
    ' fuckwhit ',
    ' fuckwit ',
    ' fudge packer ',
    ' fudgepacker ',
    ' fuk ',
    ' fuker ',
    ' fukker ',
    ' fukkin ',
    ' fuks ',
    ' fukwhit ',
    ' fukwit ',
    ' fux ',
    ' fux0r ',
    ' f_u_c_k ',
    ' gangbang ',
    ' gangbanged ',
    ' gangbangs ',
    ' gaylord ',
    ' gaysex ',
    ' goatse ',
    ' God ',
    ' god-dam ',
    ' god-damned ',
    ' goddamn ',
    ' goddamned ',
    ' hardcoresex ',
    ' hell ',
    ' heshe ',
    ' hoar ',
    ' hoare ',
    ' hoer ',
    ' homo ',
    ' hore ',
    ' horniest ',
    ' horny ',
    ' hotsex ',
    ' jack-off ',
    ' jackoff ',
    ' jap ',
    ' jerk-off ',
    ' jism ',
    ' jiz ',
    ' jizm ',
    ' jizz ',
    ' kawk ',
    ' knob ',
    ' knobead ',
    ' knobed ',
    ' knobend ',
    ' knobhead ',
    ' knobjocky ',
    ' knobjokey ',
    ' kock ',
    ' kondum ',
    ' kondums ',
    ' kum ',
    ' kummer ',
    ' kumming ',
    ' kums ',
    ' kunilingus ',
    ' l3itch ',
    ' labia ',
    ' lmfao ',
    ' lust ',
    ' lusting ',
    ' m0f0 ',
    ' m0fo ',
    ' m45terbate ',
    ' ma5terb8 ',
    ' ma5terbate ',
    ' masochist ',
    ' master-bate ',
    ' masterb8 ',
    ' masterbat3 ',
    ' masterbate ',
    ' masterbation ',
    ' masterbations ',
    ' masturbate ',
    ' mo-fo ',
    ' mof0 ',
    ' mofo ',
    ' mothafuck ',
    ' mothafucka ',
    ' mothafuckas ',
    ' mothafuckaz ',
    ' mothafucked ',
    ' mothafucker ',
    ' mothafuckers ',
    ' mothafuckin ',
    ' mothafucking ',
    ' mothafuckings ',
    ' mothafucks ',
    ' mother fucker ',
    ' motherfuck ',
    ' motherfucked ',
    ' motherfucker ',
    ' motherfuckers ',
    ' motherfuckin ',
    ' motherfucking ',
    ' motherfuckings ',
    ' motherfuckka ',
    ' motherfucks ',
    ' muff ',
    ' mutha ',
    ' muthafecker ',
    ' muthafuckker ',
    ' muther ',
    ' mutherfucker ',
    ' n1gga ',
    ' n1gger ',
    ' nazi ',
    ' nigg3r ',
    ' nigg4h ',
    ' nigga ',
    ' niggah ',
    ' niggas ',
    ' niggaz ',
    ' nigger ',
    ' niggers ',
    ' nob ',
    ' nob jokey ',
    ' nobhead ',
    ' nobjocky ',
    ' nobjokey ',
    ' numbnuts ',
    ' nutsack ',
    ' orgasim ',
    ' orgasims ',
    ' orgasm ',
    ' orgasms ',
    ' p0rn ',
    ' pawn ',
    ' pecker ',
    ' penis ',
    ' penisfucker ',
    ' phonesex ',
    ' phuck ',
    ' phuk ',
    ' phuked ',
    ' phuking ',
    ' phukked ',
    ' phukking ',
    ' phuks ',
    ' phuq ',
    ' pigfucker ',
    ' pimpis ',
    ' piss ',
    ' pissed ',
    ' pisser ',
    ' pissers ',
    ' pisses ',
    ' pissflaps ',
    ' pissin ',
    ' pissing ',
    ' pissoff ',
    ' poop ',
    ' porn ',
    ' porno ',
    ' pornography ',
    ' pornos ',
    ' prick ',
    ' pricks ',
    ' pron ',
    ' pube ',
    ' pusse ',
    ' pussi ',
    ' pussies ',
    ' pussy ',
    ' pussys ',
    ' rectum ',
    ' retard ',
    ' rimjaw ',
    ' rimming ',
    ' s hit ',
    ' s.o.b. ',
    ' sadist ',
    ' schlong ',
    ' screwing ',
    ' scroat ',
    ' scrote ',
    ' scrotum ',
    ' semen ',
    ' sex ',
    ' sh!t ',
    ' sh1t ',
    ' shag ',
    ' shagger ',
    ' shaggin ',
    ' shagging ',
    ' shemale ',
    ' shit ',
    ' shitdick ',
    ' shite ',
    ' shited ',
    ' shitey ',
    ' shitfuck ',
    ' shitfull ',
    ' shithead ',
    ' shiting ',
    ' shitings ',
    ' shits ',
    ' shitted ',
    ' shitter ',
    ' shitters ',
    ' shitting ',
    ' shittings ',
    ' shitty ',
    ' skank ',
    ' slut ',
    ' sluts ',
    ' smegma ',
    ' smut ',
    ' snatch ',
    ' son-of-a-bitch ',
    ' spac ',
    ' spunk ',
    ' s_h_i_t ',
    ' t1tt1e5 ',
    ' t1tties ',
    ' teets ',
    ' teez ',
    ' testical ',
    ' testicle ',
    ' tit ',
    ' titfuck ',
    ' tits ',
    ' titt ',
    ' tittie5 ',
    ' tittiefucker ',
    ' titties ',
    ' tittyfuck ',
    ' tittywank ',
    ' titwank ',
    ' tosser ',
    ' turd ',
    ' tw4t ',
    ' twat ',
    ' twathead ',
    ' twatty ',
    ' twunt ',
    ' twunter ',
    ' v14gra ',
    ' v1gra ',
    ' vagina ',
    ' viagra ',
    ' vulva ',
    ' w00se ',
    ' wang ',
    ' wank ',
    ' wanker ',
    ' wanky ',
    ' whoar ',
    ' whore ',
    ' willies ',
    ' willy ',
    ' xrated ',
    ' xxx '    
]

ft_common_crawl = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
embeddings_index = KeyedVectors.load_word2vec_format(ft_common_crawl)

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])    
    for p in punct:
        text = text.replace(p, f' {p} ')     
    return text

def handle_swears(text):
    text = re.sub(replace_with_fuck, ' fuck ', text)
    return text

replace_with_fuck = []
for swear in swear_words:
    if swear[1:(len(swear)-1)] not in embeddings_index:
        replace_with_fuck.append(swear)   
replace_with_fuck = '|'.join(replace_with_fuck)

#train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish','muslim', 'black', 'white', 'psychiatric_or_mental_illness']
AUX_COLUMNS = ['target', 'severe_toxicity','obscene','identity_attack','insult','threat']

#x_train = train['comment_text'].astype(str)
x_test = test['comment_text'].astype(str)

#x_train = x_train.apply(lambda x: clean_contractions(x, contraction_mapping))
#x_train = x_train.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
#x_train = x_train.apply(lambda x: handle_swears(x))
x_test = x_test.apply(lambda x: clean_contractions(x, contraction_mapping))
x_test = x_test.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
x_test = x_test.apply(lambda x: handle_swears(x))
print('Очистили текст')

del embeddings_index
del test#, #train,

max_features=410047  #410047 ВЕРНУТЬ ЭТУ ЦИФРУ!
maxlen=300
embed_size=300
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

tok=text.Tokenizer(num_words=max_features,lower=False) #, filters=CHARS_TO_REMOVE)
tok.fit_on_texts(list(x_test)) #list(x_train)+
#x_train=tok.texts_to_sequences(x_train)
x_test=tok.texts_to_sequences(x_test)
#x_train=sequence.pad_sequences(x_train,maxlen=maxlen, truncating='pre')
x_test=sequence.pad_sequences(x_test,maxlen=maxlen, truncating='pre')
print('провели токенизацию')

EMBEDDING_FILES = [
    #'../input/glove-twitter-27b-200d-txt/glove.twitter.27B.200d.txt',
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix
    
embedding_matrix = np.concatenate(
    [build_matrix(tok.word_index, f) for f in EMBEDDING_FILES], axis=-1)

word_index = len(tok.word_index)+1


del tok

print('сформировали эмбеддинг матрицу')


####################################################################
#PYTORCH PART
####################################################################
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = LSTM_UNITS * 4

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(word_index, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS) #  + LSTM_UNITS * 2
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS) #  + LSTM_UNITS * 2
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1) #  lstm_attention1
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out
        
class NeuralNetFastAi(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNetFastAi, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(word_index, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS) #  + LSTM_UNITS * 2
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS) #  + LSTM_UNITS * 2
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1) #  lstm_attention1
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

batch_size=512

x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=False)

print('pytorch ready to predict')

WEIGHTS_PYTORCH_LSTM = [
    # Pytorch LSTM Script version 6
    ['../input/pytorchweights/model 1 epoch 1.dms',
    '../input/pytorchweights/model 1 epoch 2.dms',
    '../input/pytorchweights/model 1 epoch 3.dms',
    '../input/pytorchweights/model 1 epoch 4.dms',],
    # Pytorch LSTM Script version 8
    ['../input/pytorchlstmv8/model 1 epoch 1',
    '../input/pytorchlstmv8/model 1 epoch 2',
    '../input/pytorchlstmv8/model 1 epoch 3',
    '../input/pytorchlstmv8/model 1 epoch 4']
    ]
    
WEIGHTS_PYTORCH_FASTAI = [
    ['../input/fastailstm/model 1 epoch 1.dms',
    '../input/fastailstm/model 1 epoch 2.dms',
    '../input/fastailstm/model 1 epoch 3.dms',
    '../input/fastailstm/model 1 epoch 4.dms',
    '../input/fastailstm/model 1 epoch 5.dms',],
    
    ['../input/pretextlstm/model 1 epoch 1',
    '../input/pretextlstm/model 1 epoch 2',
    '../input/pretextlstm/model 1 epoch 3',
    '../input/pretextlstmp2/model 1 epoch 4',
    '../input/pretextlstmp2/model 1 epoch 5']
    ]

predictions_pytorch = []

###################################

for model in WEIGHTS_PYTORCH_LSTM:
    #embedding_matrix = np.zeros((423873, 600))
    
    checkpoint_predictions = []
    checkpoint_weights = [2 ** epoch for epoch in range(len(model))]
    
    for weight in model:
            
        model_pytorch = NeuralNetFastAi(embedding_matrix, 6).cuda()
        temp_dict = torch.load(weight)
        del temp_dict['embedding.weight']
        temp_dict['embedding.weight'] = torch.tensor(embedding_matrix)
        
        model_pytorch.load_state_dict(temp_dict)
        model_pytorch.eval()
        
        test_preds = np.zeros(len(x_test))
        for i, x_batch in enumerate(test_loader):
            with torch.no_grad():
                y_pred = model_pytorch(*x_batch)
            test_preds[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:,0]
        # Сохраняем предсказание
        checkpoint_predictions.append(test_preds.flatten())
        print('предсказание ок')
        del model_pytorch
        gc.collect()
    
    predictions_pytorch.append(np.average(checkpoint_predictions, weights=checkpoint_weights, axis=0))

###################################
for model in WEIGHTS_PYTORCH_FASTAI:
    #embedding_matrix = np.zeros((410048, 600)) #просто заглушка
    
    checkpoint_predictions = []
    checkpoint_weights = [2 ** epoch for epoch in range(len(model))]
    
    for weight in model:
        #print('loading model')
        model_pytorch = NeuralNetFastAi(embedding_matrix, 6).cuda()
        temp_dict = torch.load(weight)
        del temp_dict['embedding.weight']
        temp_dict['embedding.weight'] = torch.tensor(embedding_matrix)
        
        model_pytorch.load_state_dict(temp_dict)
        model_pytorch.eval()
        
        test_preds = np.zeros(len(x_test))
        for i, x_batch in enumerate(test_loader):
            #x_batch = x_batch.cuda()
            with torch.no_grad():
                y_pred = model_pytorch(*x_batch)
            test_preds[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:,0]
        # Сохраняем предсказание
        checkpoint_predictions.append(test_preds.flatten())
        print('предсказание pytorch ок')
        del model_pytorch
        gc.collect()
    
    predictions_pytorch.append(np.average(checkpoint_predictions, weights=checkpoint_weights, axis=0))  

torch.cuda.empty_cache()

try:
    del temp_dict
except:
    pass


#########################################################
predictions_pytorch = np.average(predictions_pytorch, axis=0)
#########################################################

####################################################################
#KERAS PART
####################################################################


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
batch_size = 128
epochs = 4


def build_model(embedding_matrix, num_aux_targets):
    sequence_input = Input(shape=(maxlen, ))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable = False)(sequence_input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x) #CuDNNLSTM
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x) #CuDNNLSTM
    x = AttentionWithContext()(x)
    x = Dense(DENSE_HIDDEN_UNITS, activation="sigmoid")(x) 
    x = Dropout(0.3)(x)
    x = Dense(int(DENSE_HIDDEN_UNITS/2), activation="sigmoid")(x) 
    preds = Dense(1, activation="sigmoid")(x)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(x)
    model = Model(sequence_input, outputs=[preds, aux_result])
    model.compile(loss='binary_crossentropy', 
    optimizer=Adam(lr=1e-3, clipvalue=0.5),metrics=['accuracy'])

    return model
    
WEIGHTS_KERAS = [
                #[#'../input/keraslstm/model 1 epoch 1.h5',
                #'../input/keraslstm/model 1 epoch 2.h5',
                #'../input/keraslstm/model 1 epoch 3.h5',
                #'../input/keraslstm/model 1 epoch 4.h5',
                #'../input/keraslstmp2/model 1 epoch 5.h5'
                #],
                ['../input/keraslstmmodel2/model 1 epoch 1-2.h5',
                '../input/keraslstmmodel2/model 1 epoch 2-2.h5',
                '../input/keraslstmmodel2/model 1 epoch 3-2.h5',
                '../input/keraslstmmodel2/model 1 epoch 4-2.h5']
                    ] # добавить пути к весам финальных моделей

predictions_keras = []

for model in WEIGHTS_KERAS:
    checkpoint_predictions = []
    checkpoint_weights = [2 ** epoch for epoch in range(len(model))]
    
    for weight in model:
        model_keras = build_model(embedding_matrix, 6)
        model1epoch1 = load_model(weight,custom_objects={'AttentionWithContext':AttentionWithContext})
        
        model_keras.layers[0].set_weights(model1epoch1.layers[0].get_weights())
        model_keras.layers[2].set_weights(model1epoch1.layers[2].get_weights())
        model_keras.layers[3].set_weights(model1epoch1.layers[3].get_weights())
        model_keras.layers[4].set_weights(model1epoch1.layers[4].get_weights())
        model_keras.layers[5].set_weights(model1epoch1.layers[5].get_weights())
        model_keras.layers[6].set_weights(model1epoch1.layers[6].get_weights())
        model_keras.layers[7].set_weights(model1epoch1.layers[7].get_weights())
        model_keras.layers[8].set_weights(model1epoch1.layers[8].get_weights())
        model_keras.layers[9].set_weights(model1epoch1.layers[9].get_weights())
        model_keras.layers[10].set_weights(model1epoch1.layers[10].get_weights())
        
        del model1epoch1
        
        checkpoint_predictions.append(model_keras.predict(x_test, batch_size=2048)[0].flatten())
        print('предсказание keras ok')
        del model_keras
    
    predictions_keras.append(np.average(checkpoint_predictions, weights=checkpoint_weights, axis=0))

#########################################################
predictions_keras = np.average(predictions_keras, axis=0)
#########################################################
from keras import backend as K
import gc
K.clear_session()
gc.collect()


#########################################################
#FINAL PREDICTION
#########################################################

FINAL_PREDICTION = np.average([predictions_pytorch,predictions_keras], axis=0)
#FINAL_PREDICTION = np.average([predictions_pytorch], axis=0)

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
print('пробуем сохранить предсказание')
submission["prediction"] = FINAL_PREDICTION
submission.to_csv('submission.csv', index=False)