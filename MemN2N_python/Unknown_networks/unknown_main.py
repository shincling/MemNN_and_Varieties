# -*- coding: utf8 -*-
from __future__ import division
import argparse
import glob
import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer,label_binarize

class SimpleAttentionLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, vocab, embedding_size,enable_time, W_h, W_q,W_o, nonlinearity=lasagne.nonlinearities.tanh, **kwargs):
        super(SimpleAttentionLayer, self).__init__(incomings, **kwargs) #？？？不知道这个super到底做什么的，会引入input_layers和input_shapes这些属性
        if len(incomings) != 2:
            raise NotImplementedError
        batch_size, max_sentlen ,embedding_size = self.input_shapes[0]
        self.batch_size,self.max_sentlen,self.embedding_size=batch_size,max_sentlen,embedding_size
        self.W_h=self.add_param(W_h,(embedding_size,embedding_size), name='Attention_layer_W_h')
        self.W_q=self.add_param(W_q,(embedding_size,embedding_size), name='Attention_layer_W_q')
        self.W_o=self.add_param(W_o,(embedding_size,), name='Attention_layer_W_o')
        self.nonliearity=nonlinearity
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        # self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A,self.C]])

    def get_output_shape_for(self, input_shapes):
        return (self.batch_size,self.embedding_size)
    def get_output_for(self, inputs, **kwargs):
        #input[0]:(BS,max_senlen,emb_size),input[1]:(BS,1,emb_size)
        activation=self.nonliearity(T.dot(inputs[0],self.W_h)+T.dot(inputs[1],self.W_q))
        final=T.dot(activation,self.W_o) #(BS,max_sentlen)
        alpha=lasagne.nonlinearities.softmax(final) #(BS,max_sentlen)
        final=T.batched_dot(alpha,inputs[0])#(BS,max_sentlen)*(BS,max_sentlen,emb_size)--(BS,emb_size)
        return final
    def reset_zero(self):
        self.set_zero(self.zero_vec)


class Model:
    def __init__(self, train_file, test_file, batch_size=32, embedding_size=20, max_norm=40, lr=0.01, num_hops=3, adj_weight_tying=True, linear_start=True, enable_time=False,**kwargs):
        train_lines, test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([train_lines, test_lines], axis=0) #直接头尾拼接
        vocab, word_to_idx, idx_to_word, max_sentlen = self.get_vocab(lines)
        #C是document的列表，Q是定位问题序列的列表，Y是答案组成的列表，目前为知都是字符形式的，没有向量化#
        self.data = {'train': {}, 'test': {}}  #各是一个字典
        S_train, self.data['train']['Q'], self.data['train']['Y'] = self.process_dataset(train_lines, word_to_idx, max_sentlen, offset=0)
        S_test, self.data['test']['Q'], self.data['test']['Y'] = self.process_dataset(test_lines, word_to_idx, max_sentlen)
        S = np.concatenate([np.zeros((1, max_sentlen), dtype=np.int32), S_train, S_test], axis=0)
        self.data['train']['S'],self.data['test']['S']=S_train,S_test
        for i in range(min(10,len(self.data['test']['Y']))):
            for k in ['S', 'Q', 'Y']:
                print k, self.data['test'][k][i]
        print 'batch_size:', batch_size, 'max_sentlen:', max_sentlen
        print 'sentences:', S.shape
        print 'vocab size:', len(vocab)

        for d in ['train', 'test']:
            print d,
            for k in ['S', 'Q', 'Y']:
                print k, self.data[d][k].shape,
            print ''

        vocab=[]
        for i in range(len(idx_to_word)):
            vocab.append(idx_to_word[i+1])

        lb = LabelBinarizer()

        self.enable_time=enable_time
        self.batch_size = batch_size
        self.max_sentlen = max_sentlen if not enable_time else max_sentlen+1
        self.embedding_size = embedding_size
        self.num_classes = len(vocab) + 1
        self.vocab = vocab
        self.lb = lb
        self.init_lr = lr
        self.lr = self.init_lr
        self.max_norm = max_norm
        self.S = S
        self.idx_to_word = idx_to_word
        self.nonlinearity = None if linear_start else lasagne.nonlinearities.softmax
        self.word_to_idx=word_to_idx

        self.build_network()

    def build_network(self):
        batch_size, max_sentlen, embedding_size, vocab, enable_time = self.batch_size, self.max_sentlen, self.embedding_size, self.vocab,self.enable_time

        s = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        # c_pe = T.tensor4()
        # q_pe = T.tensor4()
        self.s_shared = theano.shared(np.zeros((batch_size, max_sentlen), dtype=np.int32), borrow=True)
        self.q_shared = theano.shared(np.zeros((batch_size, ), dtype=np.int32), borrow=True)
        '''最后的softmax层的参数'''
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)
        S_shared = theano.shared(self.S, borrow=True)#这个S把train test放到了一起来干事情#

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_sentlen))
        l_question_in = lasagne.layers.InputLayer(shape=(batch_size,1))

        w_emb=lasagne.init.Normal(std=0.1)
        l_context_emb = lasagne.layers.EmbeddingLayer(l_context_in,len(vocab)+1,embedding_size,W=w_emb,name='sentence_embedding') #(BS,max_sentlen,emb_size)
        l_question_emb= lasagne.layers.EmbeddingLayer(l_question_in,len(vocab)+1,embedding_size,W=l_context_emb.W,name='question_embedding') #(BS,1,d)

        l_context_rnn=lasagne.layers.LSTMLayer(l_context_emb,embedding_size,name='context_lstm') #(BS,max_sentlen,emb_size)

        w_h,w_q,w_o=lasagne.init.Normal(std=0.1),lasagne.init.Normal(std=0.1),lasagne.init.Normal(std=0.1)
        #下面这个层是用来利用question做attention，得到文档在当前q下的最后一个表示,输出一个(BS,emb_size)的东西
        #得到一个(BS,emb_size)的加权平均后的向量
        l_context_attention=SimpleAttentionLayer((l_context_rnn,l_question_emb),vocab, embedding_size,enable_time, W_h=w_h, W_q=w_q,W_o=w_o, nonlinearity=lasagne.nonlinearities.tanh)
        # l_merge=TmpMergeLayer((l_context_attention,l_question_emb),W_merge_r=w_merge_r,W_merge_q=w_merge_q, nonlinearity=lasagne.nonlinearities.tanh))
        context_attention_h=lasagne.layers.helper.get_output(l_context_attention,{l_context_in:s,l_question_in:q})

        return




    def get_lines(self, fname):
        lines = [] #每个元素是个字典看来
        for i, line in enumerate(open(fname)):
            ll=line.split('\t')
            id = int(ll[0])
            sentence=ll[1]
            question=ll[2][0:ll[2].find(' ')]
            answer=ll[3].strip()
            lines.append({'id':id,'sentence':sentence,'question':question,'target':answer})
        return np.array(lines)

    def get_vocab(self, lines): #这个函数相当于预处理的函数
        vocab = set()
        max_sentlen = 0
        for i, line in enumerate(lines):
            #words = nltk.word_tokenize(line['text'])
            words=line['sentence'].split(' ')  #这里做了一个修改，替换掉了nltk的工具
            max_sentlen = max(max_sentlen, len(words))
            for w in words:
                vocab.add(w)
            vocab.add(line['question'])
            vocab.add(line['target'])

        word_to_idx = {}
        for w in vocab:
            word_to_idx[w] = len(word_to_idx) + 1

        idx_to_word = {}
        for w, idx in word_to_idx.iteritems():
            idx_to_word[idx] = w

        return vocab, word_to_idx, idx_to_word, max_sentlen


    def process_dataset(self,lines, word_to_idx, max_sentlen, offset=0):
        S,Q,Y=[],[],[]
        for i, line in enumerate(lines):
            word_indices = [word_to_idx[w] for w in line['sentence'].split(' ')]
            word_indices += [0] * (max_sentlen - len(word_indices)) #这是补零，把句子填充到max_sentLen
            S.append(word_indices)
            Q.append(word_to_idx[line['question']])
            Y.append(word_to_idx[line['target']])
        return np.array(S),np.array(Q),np.array(Y)



def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--task', type=int, default=35, help='Task#')
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--back_method', type=str, default='sgd', help='Train Method to bp')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=100, help='Embedding size')
    parser.add_argument('--max_norm', type=float, default=40.0, help='Max norm')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_hops', type=int, default=3, help='Num hops')
    parser.add_argument('--linear_start', type='bool', default=True, help='Whether to start with linear activations')
    parser.add_argument('--shuffle_batch', type='bool', default=True, help='Whether to shuffle minibatches')
    parser.add_argument('--n_epochs', type=int, default=500, help='Num epochs')
    parser.add_argument('--enable_time', type=bool, default=False, help='time word embedding')
    args = parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80

    if args.train_file == '' or args.test_file == '':
        args.train_file = glob.glob('*toy_train.txt' )[0]
        args.test_file = glob.glob('*_toy_test.txt' )[0]
        # args.train_file = '/home/shin/DeepLearning/MemoryNetwork/MemN2N_python/MemN2N-master/data/en/qqq_train.txt'
        # args.test_file ='/home/shin/DeepLearning/MemoryNetwork/MemN2N_python/MemN2N-master/data/en/qqq_test.txt'

    model = Model(**args.__dict__)
    # model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
    main()