# -*- coding: utf8 -*-
from __future__ import division
import argparse
import glob
import re
import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
import nltk
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
        self.nonlinearity=nonlinearity
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        # self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A,self.C]])

    def get_output_shape_for(self, input_shapes):
        return (self.batch_size,self.embedding_size)
    def get_output_for(self, inputs, **kwargs):
        #input[0]:(BS,max_senlen,emb_size),input[1]:(BS,1,emb_size)
        activation0=(T.dot(inputs[0],self.W_h))
        activation1=T.dot(inputs[1],self.W_q).reshape([self.batch_size,self.embedding_size]).dimshuffle(0,'x',1)
        activation=self.nonlinearity(activation0+activation1)#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        final=T.dot(activation,self.W_o) #(BS,max_sentlen)
        alpha=lasagne.nonlinearities.softmax(final) #(BS,max_sentlen)
        final=T.batched_dot(alpha,inputs[0])#(BS,max_sentlen)*(BS,max_sentlen,emb_size)--(BS,emb_size)
        return final
    # TODO:think about the set_zero
    def reset_zero(self):
        self.set_zero(self.zero_vec)


class SimplePointerLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, vocab, embedding_size,enable_time, W_h, W_q,W_o, nonlinearity=lasagne.nonlinearities.tanh,**kwargs):
        super(SimplePointerLayer, self).__init__(incomings, **kwargs) #？？？不知道这个super到底做什么的，会引入input_layers和input_shapes这些属性
        if len(incomings) != 3:
            raise NotImplementedError
        # if mask_input is not None:
        #     incomings.append(mask_input)
        batch_size, max_sentlen ,embedding_size = self.input_shapes[0]
        self.batch_size,self.max_sentlen,self.embedding_size=batch_size,max_sentlen,embedding_size
        self.W_h=self.add_param(W_h,(embedding_size,embedding_size), name='Pointer_layer_W_h')
        self.W_q=self.add_param(W_q,(embedding_size,embedding_size), name='Pointer_layer_W_q')
        self.W_o=self.add_param(W_o,(embedding_size,), name='Pointer_layer_W_o')
        self.nonlinearity=nonlinearity
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        # self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A,self.C]])

    def get_output_shape_for(self, input_shapes):
        return (self.batch_size,self.max_sentlen)
    def get_output_for(self, inputs, **kwargs):
        #input[0]:(BS,max_senlen,emb_size),input[1]:(BS,1,emb_size),input[2]:(BS,max_sentlen)
        activation0=(T.dot(inputs[0],self.W_h))
        activation1=T.dot(inputs[1],self.W_q).reshape([self.batch_size,self.embedding_size]).dimshuffle(0,'x',1)
        activation=self.nonlinearity(activation0+activation1)#.dimshuffle(0,'x',2)#.repeat(self.max_sentlen,axis=1)
        final=T.dot(activation,self.W_o) #(BS,max_sentlen)
        if inputs[2] is not None:
            final=inputs[2]*final-(1-inputs[2])*1000000
        alpha=lasagne.nonlinearities.softmax(final) #(BS,max_sentlen)
        # final=T.batched_dot(alpha,inputs[0])#(BS,max_sentlen)*(BS,max_sentlen,emb_size)--(BS,emb_size)
        return alpha
    # TODO:think about the set_zero
    def reset_zero(self):
        self.set_zero(self.zero_vec)
class TmpMergeLayer(lasagne.layers.MergeLayer):
    def __init__(self,incomings,W_merge_r,W_merge_q, nonlinearity=lasagne.nonlinearities.tanh,**kwargs):
        super(TmpMergeLayer, self).__init__(incomings, **kwargs) #？？？不知道这个super到底做什么的，会引入input_layers和input_shapes这些属性
        if len(incomings) != 2:
            raise NotImplementedError
        batch_size,embedding_size=self.input_shapes[0]
        self.W_merge_r=self.add_param(W_merge_r,(embedding_size,embedding_size),name='MergeLayer_w_r')
        self.W_merge_q=self.add_param(W_merge_q,(embedding_size,embedding_size),name='MergeLayer_w_q')
        self.batch_size,self.embedding_size=batch_size,embedding_size
        self.nonlinearity=nonlinearity
    def get_output_shape_for(self, input_shapes):
        return self.input_shapes[0]
    def get_output_for(self, inputs, **kwargs):
        h_r,h_q=inputs[0],inputs[1] # h_r:(BS,emb_size),h_q:(BS,1,emb_size)
        # result=T.dot(self.W_merge_r,h_r)+T.dot(self.W_merge_q,h_q).reshape((self.batch_size,self.embedding_size))
        result=T.dot(h_r,self.W_merge_r)+T.dot(h_q,self.W_merge_q).reshape((self.batch_size,self.embedding_size))
        return result

class TransposedDenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units,embedding_size,vocab_size, W_final_softmax=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming,num_units, name='softmax_layer_w',**kwargs)
        # self.W_final_softmax=self.add_param(W_final_softmax,(embedding_size,num_units),name='softmax_layer_w')
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)




class Model:
    def __init__(self, train_file, test_file, batch_size=32, embedding_size=20, max_norm=40, lr=0.01, num_hops=3, adj_weight_tying=True, linear_start=True, enable_time=False,pointer_nn=False,optimizer='sgd',enable_mask=True,std_rate=0.1,choice_num=4,**kwargs):
        max_sentlen, max_storylen, vocab= 0, 0, set()
        max_sentlen, max_storylen, vocab, total_train = self.get_stories(train_file,max_sentlen,max_storylen,vocab)
        max_sentlen, max_storylen, vocab, total_test = self.get_stories(test_file,max_sentlen,max_storylen,vocab)

        word_to_idx = {}
        for w in vocab:
            word_to_idx[w] = len(word_to_idx) + 1

        idx_to_word = {}
        for w, idx in word_to_idx.iteritems():
            idx_to_word[idx] = w

        #C是document的列表，Q是定位问题序列的列表，Y是候选答案，T是正确答案，目前为知都是字符形式的，没有向量化#
        self.data = {'train': {}, 'test': {}}  #各是一个字典
        train_file_ans='mc160.train.ans'
        test_file_ans='mc160.test.ans'
        self.data['train']['S'], self.data['train']['Q'], self.data['train']['Y'],self.data['train']['T'], self.data['train']['Mask']= self.process_dataset(train_file_ans,total_train, word_to_idx, max_storylen,max_sentlen)
        self.data['test']['S'], self.data['test']['Q'], self.data['test']['Y'],self.data['test']['T'], self.data['test']['Mask'] = self.process_dataset(test_file_ans,total_test, word_to_idx, max_storylen,max_sentlen)

        # for i in range(min(10,len(self.data['test']['Y']))):
        #     for k in ['S', 'Q', 'Y']:
        #         print k, self.data['test'][k][i]
        print 'batch_size:', batch_size, 'max_storylen:', max_storylen ,'max_sentlen:', max_sentlen
        print 'vocab size:', len(vocab)

        for d in ['train', 'test']:
            print d,
            for k in ['S', 'Q', 'Y','T','Mask']:
                print k, self.data[d][k].shape,
            print ''

        vocab=[]
        for i in range(len(idx_to_word)):
            vocab.append(idx_to_word[i+1])
        idx_to_word[0]='#'
        word_to_idx['#']=0

        lb = LabelBinarizer()

        self.enable_time=enable_time
        self.optimizer=optimizer
        self.batch_size = batch_size
        self.max_sentlen = max_sentlen if not enable_time else max_sentlen+1
        self.embedding_size = embedding_size
        self.num_classes = len(vocab) + 1
        self.vocab = vocab
        self.lb = lb
        self.init_lr = lr
        self.lr = self.init_lr
        self.max_norm = max_norm
        self.idx_to_word = idx_to_word
        self.nonlinearity = None if linear_start else lasagne.nonlinearities.softmax
        self.word_to_idx=word_to_idx
        self.pointer_nn=pointer_nn
        self.std=std_rate
        self.enable_mask=enable_mask
        # self.build_network()

    def build_network(self):
        batch_size, max_sentlen, embedding_size, vocab, enable_time = self.batch_size, self.max_sentlen, self.embedding_size, self.vocab,self.enable_time

        s = T.imatrix()
        s = T.imatrix()
        # q = T.ivector()
        q = T.imatrix()
        y = T.imatrix()
        mask= T.imatrix()# if self.enable_mask else None
        # c_pe = T.tensor4()
        # q_pe = T.tensor4()
        self.s_shared = theano.shared(np.zeros((batch_size, max_sentlen), dtype=np.int32), borrow=True)
        self.mask_shared = theano.shared(np.zeros((batch_size, max_sentlen), dtype=np.int32), borrow=True)
        self.q_shared = theano.shared(np.zeros((batch_size, 1), dtype=np.int32), borrow=True)
        '''最后的softmax层的参数'''
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)
        # S_shared = theano.shared(self.S, borrow=True)#这个S把train test放到了一起来干事情#

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_sentlen))
        l_mask_in = lasagne.layers.InputLayer(shape=(batch_size, max_sentlen))
        l_question_in = lasagne.layers.InputLayer(shape=(batch_size,1))

        w_emb=lasagne.init.Normal(std=self.std)
        l_context_emb = lasagne.layers.EmbeddingLayer(l_context_in,self.num_classes,embedding_size,W=w_emb,name='sentence_embedding') #(BS,max_sentlen,emb_size)
        l_question_emb= lasagne.layers.EmbeddingLayer(l_question_in,self.num_classes,embedding_size,W=l_context_emb.W,name='question_embedding') #(BS,1,d)

        # w_emb_query=lasagne.init.Normal(std=self.std)
        # l_question_emb= lasagne.layers.EmbeddingLayer(l_question_in,self.num_classes,embedding_size,W=w_emb_query,name='question_embedding') #(BS,1,d)

        l_context_rnn_f=lasagne.layers.LSTMLayer(l_context_emb,embedding_size,name='contexut_lstm',mask_input=l_mask_in,backwards=False) #(BS,max_sentlen,emb_size)
        l_context_rnn_b=lasagne.layers.LSTMLayer(l_context_emb,embedding_size,name='context_lstm',mask_input=l_mask_in,backwards=True) #(BS,max_sentlen,emb_size)
        # l_context_rnn_f=lasagne.layers.GRULayer(l_context_emb,embedding_size,name='context_lstm',mask_input=l_mask_in,backwards=False) #(BS,max_sentlen,emb_size)
        # l_context_rnn_b=lasagne.layers.GRULayer(l_context_emb,embedding_size,name='context_lstm',mask_input=l_mask_in,backwards=True) #(BS,max_sentlen,emb_size)
        l_context_rnn=lasagne.layers.ElemwiseSumLayer((l_context_rnn_f,l_context_rnn_b))
        w_h,w_q,w_o=lasagne.init.Normal(std=self.std),lasagne.init.Normal(std=self.std),lasagne.init.Normal(std=self.std)
        #下面这个层是用来利用question做attention，得到文档在当前q下的最后一个表示,输出一个(BS,emb_size)的东西
        #得到一个(BS,emb_size)的加权平均后的向量
        if not self.pointer_nn:
            l_context_attention=SimpleAttentionLayer((l_context_rnn,l_question_emb),vocab, embedding_size,enable_time, W_h=w_h, W_q=w_q,W_o=w_o, nonlinearity=lasagne.nonlinearities.tanh)
            w_merge_r,w_merge_q=lasagne.init.Normal(std=self.std),lasagne.init.Normal(std=self.std)
            l_merge=TmpMergeLayer((l_context_attention,l_question_emb),W_merge_r=w_merge_r,W_merge_q=w_merge_q, nonlinearity=lasagne.nonlinearities.tanh)

            w_final_softmax=lasagne.init.Normal(std=self.std)
            # l_pred = TransposedDenseLayer(l_merge, self.num_classes,embedding_size=embedding_size,vocab_size=self.num_classes,W_final_softmax=w_final_softmax, b=None, nonlinearity=lasagne.nonlinearities.softmax)
            l_pred = lasagne.layers.DenseLayer(l_merge, self.num_classes, W=w_final_softmax, b=None, nonlinearity=lasagne.nonlinearities.softmax,name='l_final')

            probas=lasagne.layers.helper.get_output(l_pred,{l_context_in:s,l_question_in:q,l_mask_in:mask})
            probas = T.clip(probas, 1e-7, 1.0-1e-7)

            pred = T.argmax(probas, axis=1)

            cost = T.nnet.binary_crossentropy(probas, y).sum()
        else :
            l_context_pointer=SimplePointerLayer((l_context_rnn,l_question_emb,l_mask_in),vocab, embedding_size,enable_time, W_h=w_h, W_q=w_q,W_o=w_o, nonlinearity=lasagne.nonlinearities.tanh)
            l_pred=l_context_pointer
            probas=lasagne.layers.helper.get_output(l_pred,{l_context_in:s,l_question_in:q,l_mask_in:mask})
            probas = T.clip(probas, 1e-7, 1.0-1e-7)
            pred = T.argmax(probas, axis=1)

            cost = T.nnet.categorical_crossentropy(probas, y).sum()
            # cost = cost*batch_size/mask.sum()*10
            pass
        params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
        print 'params:', params
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
        if self.optimizer=='sgd':
            updates = lasagne.updates.sgd(scaled_grads, params, learning_rate=self.lr)
        elif self.optimizer=='adagrad':
            updates = lasagne.updates.adagrad(scaled_grads, params, learning_rate=self.lr)
        else:
            updates = lasagne.updates.adadelta(scaled_grads, params, learning_rate=self.lr)


        givens = {
            s: self.s_shared,
            q: self.q_shared,
            y: self.a_shared,
            mask: self.mask_shared
        }

        # test_output=lasagne.layers.helper.get_output(l_context_attention,{l_context_in:s,l_question_in:q}).flatten().sum()
        # self.train_model1 = theano.function([],test_output, givens=givens,on_unused_input='warn' )
        self.train_model = theano.function([], cost, givens=givens, updates=updates)
        self.compute_pred = theano.function([], outputs= [probas,pred], givens=givens, on_unused_input='ignore')

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [l_context_emb.W]])




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

    def get_stories(self,fname,max_sentlen,max_storylen,vocab):
        total=[]
        all=open(fname).read()
        story_list=all.split('***************************************************')[1:]
        for i,story in enumerate(story_list):
            one_story_dict={}

            part_index=story.find('\r\n\r\n1: ')
            contents=story[(story.find('\r\n\r\n')+2):part_index]
            sent_detector = nltk.data.load('/home/shin/nltk_data/tokenizers/punkt/english.pickle')
            sentences=sent_detector.tokenize(contents)
            sentences=[sen.replace('\r\n','') for sen in sentences]
            one_story_dict['story']=sentences
            max_storylen=len(sentences) if len(sentences)>max_storylen else max_storylen
            print 'The story{} covers:{} sentences'.format(i,len(sentences))

            questions=re.findall('[1-4]:[\s\S]*?\r\n\r\n',story[part_index:])
            one_story_dict['question1']={'q':re.findall('\d:.*?: (.*?)\r\n',questions[0])[0],
                                         'A':re.findall('A\) (.*?)\r\n',questions[0])[0],
                                         'B':re.findall('B\) (.*?)\r\n',questions[0])[0],
                                         'C':re.findall('C\) (.*?)\r\n',questions[0])[0],
                                         'D':re.findall('D\) (.*?)\r\n',questions[0])[0],}
                                         # 'target':re.findall('\*([A-D])',questions[0])[0]}
            one_story_dict['question2']={'q':re.findall('\d:.*?: (.*?)\r\n',questions[1])[0],
                                         'A':re.findall('A\) (.*?)\r\n',questions[1])[0],
                                         'B':re.findall('B\) (.*?)\r\n',questions[1])[0],
                                         'C':re.findall('C\) (.*?)\r\n',questions[1])[0],
                                         'D':re.findall('D\) (.*?)\r\n',questions[1])[0],}
                                         # 'target':re.findall('\*([A-D])',questions[1])[0]}
            one_story_dict['question3']={'q':re.findall('\d:.*?: (.*?)\r\n',questions[2])[0],
                                         'A':re.findall('A\) (.*?)\r\n',questions[2])[0],
                                         'B':re.findall('B\) (.*?)\r\n',questions[2])[0],
                                         'C':re.findall('C\) (.*?)\r\n',questions[2])[0],
                                         'D':re.findall('D\) (.*?)\r\n',questions[2])[0],}
                                         # 'target':re.findall('\*([A-D])',questions[2])[0]}
            one_story_dict['question4']={'q':re.findall('\d:.*?: (.*?)\r\n',questions[3])[0],
                                         'A':re.findall('A\) (.*?)\r\n',questions[3])[0],
                                         'B':re.findall('B\) (.*?)\r\n',questions[3])[0],
                                         'C':re.findall('C\) (.*?)\r\n',questions[3])[0],
                                         'D':re.findall('D\) (.*?)\r\n',questions[3])[0],}
                                         # 'target':re.findall('\*([A-D])',questions[3])[0]}
            total.append(one_story_dict)


            for i,line in enumerate(sentences):
                words=nltk.word_tokenize(line)
                max_sentlen=max(max_sentlen,len(words))
                for w in words:
                    vocab.add(w)

            for i in range(1,5):
                for j in ['q','A','B','C','D']:
                    words=nltk.word_tokenize(one_story_dict['question%d'%i][j])
                    max_sentlen=max(max_sentlen,len(words))
                    for w in words:
                        vocab.add(w)


        return max_sentlen,max_storylen,vocab,total




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


    def process_dataset(self,train_or_test,total, word_to_idx, max_storylen,max_sentlen, offset=0):
        S,Q,Y,T,Mask=[],[],[],[],[]
        for one_passage in total:
            s=np.zeros([max_storylen,max_sentlen],dtype=np.int32)
            mask_story=np.zeros(max_storylen,dtype=np.int32)
            mask_story[:len(one_passage['story'])]=1 # 一篇阅读的story的mask
            for idx,sen in enumerate(one_passage['story']):
                one_sen=np.zeros(max_sentlen,dtype=np.int32)
                words=nltk.word_tokenize(sen)
                one_sen[:len(words)]=[word_to_idx[w] for w in words]
                s[idx,:]=one_sen

            for i in range(1,5): #对于每篇文章里的四个问题
                S.append(s)
                Mask.append(mask_story)
                one_question=np.zeros(max_sentlen,dtype=np.int32)
                q=one_passage['question{}'.format(i)]
                q_words=nltk.word_tokenize(q['q'])
                one_question[:len(q_words)]=[word_to_idx[w] for w in q_words]
                Q.append(one_question)

                y=np.zeros([4,max_sentlen],dtype=np.int32)
                for jdx,j in enumerate(['A','B','C','D']):
                    one_answer=np.zeros(max_sentlen,dtype=np.int32)
                    ans_words=nltk.word_tokenize(one_passage['question{}'.format(i)][j])
                    one_answer[:len(ans_words)]=[word_to_idx[w] for w in ans_words]
                    y[jdx,:]=one_answer
                Y.append(y)

        target=open(train_or_test).read()
        target_list=re.findall('[A-D]',target)
        assert len(target_list)==len(Y)
        T=[label_binarize([t],['A','B','C','D']) for t in target_list]

        return np.array(S),np.array(Q),np.array(Y),np.array(T),np.array(Mask)

    def set_shared_variables(self, dataset, index,enable_time):
        c = np.zeros((self.batch_size, self.max_sentlen), dtype=np.int32)
        # mask = np.zeros((self.batch_size, self.max_sentlen), dtype=np.int32)
        q = np.zeros((self.batch_size, 1), dtype=np.int32)
        y = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)
        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i, row in enumerate(dataset['S'][indices]):
            row = row[:self.max_sentlen]
            c[i, :len(row)] = row
        mask=np.int32(c!=0) #if self.enable_mask else None

        q[:len(indices),:] = dataset['Q'][indices] #问题的行数组成的列表
        '''底下这个整个循环是得到一个batch对应的那个调整的矩阵'''
        # y[:len(indices), 1:self.num_classes] = self.lb.transform(dataset['Y'][indices])#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        y[:len(indices), 1:self.num_classes] = label_binarize([self.idx_to_word[i] for i in dataset['Y'][indices]],self.vocab)#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        # y[:len(indices), 1:self.embedding_size] = self.mem_layers[0].A[[self.word_to_idx(i) for i in list(dataset['Y'][indices])]]#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        self.s_shared.set_value(c)
        self.mask_shared.set_value(mask)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)

    def set_shared_variables_pointer(self, dataset, index,enable_time):
        c = np.zeros((self.batch_size, self.max_sentlen), dtype=np.int32)
        q = np.zeros((self.batch_size, 1), dtype=np.int32)
        y = np.zeros((self.batch_size, self.max_sentlen), dtype=np.int32)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i, row in enumerate(dataset['S'][indices]):
            row = row[:self.max_sentlen]
            c[i, :len(row)] = row
        mask=np.int32(c!=0) #if self.enable_mask else None
        q[:len(indices),:] = dataset['Q'][indices] #问题的行数组成的列表
        '''底下这个整个循环是得到一个batch对应的那个调整的矩阵'''
        # y[:len(indices), 1:self.num_classes] = self.lb.transform(dataset['Y'][indices])#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        # y[:len(indices), 1:self.max_sentlen] = label_binarize([self.idx_to_word[i] for i in dataset['Y'][indices]],self.vocab)#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        # y[:len(indices), 1:self.max_sentlen] = [label_binarize([dataset['Y'][indices]],)#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        for i in range(len(indices)):
            one_hot=label_binarize([dataset['Y'][i]],dataset['S'][i])
            y[i,:]=one_hot

        # y[:len(indices), 1:self.embedding_size] = self.mem_layers[0].A[[self.word_to_idx(i) for i in list(dataset['Y'][indices])]]#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        self.s_shared.set_value(c)
        self.mask_shared.set_value(mask)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)


    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        n_train_batches = len(self.data['train']['Y']) // self.batch_size
        self.lr = self.init_lr
        prev_train_f1 = None

        while (epoch < n_epochs):
            epoch += 1
            if epoch % 50 == 0: #每隔25个epoch，则速率减半
                self.lr /= 2.0

            indices = range(n_train_batches)
            if shuffle_batch:
                self.shuffle_sync(self.data['train'])#保持对应性地一次性shuffle了C Q Y

            total_cost = 0
            start_time = time.time()
            # print 'TRAIN', '=' * 40
            # train_f1, train_errors = self.compute_f1(self.data['train'])
            # print 'TRAIN_ERROR:', (1-train_f1)*100
            for minibatch_index in indices:#一次进入一个batch的数据
                if not self.pointer_nn:
                    self.set_shared_variables(self.data['train'], minibatch_index,self.enable_time)#这里的函数总算把数据传给了模型里面初始化的变量
                else:
                    self.set_shared_variables_pointer(self.data['train'], minibatch_index,self.enable_time)#这里的函数总算把数据传给了模型里面初始化的变量
                total_cost += self.train_model()
                # print self.train_model1()
                self.set_zero(self.zero_vec)  #reset是把A,C的第一行（也就是第一个元素，对应字典了的第一个词）reset了一次，变成了0
            end_time = time.time()
            print '\n' * 3, '*' * 80
            print 'epoch:', epoch, 'cost:', (total_cost / len(indices)), ' took: %d(s)' % (end_time - start_time)

            print 'TRAIN', '=' * 40
            if not self.pointer_nn:
                train_f1, train_errors = self.compute_f1(self.data['train'])
            else:
                train_f1, train_errors = self.compute_f1_pointer(self.data['train'])
            print 'TRAIN_ERROR:', (1-train_f1)*100
            if False:
                for i, pred in train_errors[:10]:
                    print 'context: ', self.to_words(self.data['train']['S'][i])
                    print 'question: ', self.to_words([self.data['train']['Q'][i]])
                    print 'correct answer: ', self.data['train']['Y'][i]
                    print 'predicted answer: ', pred
                    print '---' * 20
            '''这块负责了linearity和softmanx的切换'''
            if False and prev_train_f1 is not None and train_f1 < prev_train_f1 and self.nonlinearity is None:
                print 'The linearity ends.××××××××××××××××××\n\n'
                prev_weights = lasagne.layers.helper.get_all_param_values(self.network)
                self.build_network(nonlinearity=lasagne.nonlinearities.softmax)
                lasagne.layers.helper.set_all_param_values(self.network, prev_weights)
            else:
                print 'TEST', '=' * 40
                if not self.pointer_nn:
                    test_f1, test_errors = self.compute_f1(self.data['test']) #有点奇怪这里的f1和test_error怎么好像不对应的？
                else:
                    test_f1, test_errors = self.compute_f1_pointer(self.data['test']) #有点奇怪这里的f1和test_error怎么好像不对应的？

                print 'test_f1,test_errors:',test_f1,len(test_errors)
                print '*** TEST_ERROR:', (1-test_f1)*100
                if 1 and (50<epoch) :
                    for i, pred in test_errors[:10]:
                        print 'context: ', self.to_words(self.data['test']['S'][i],'S')
                        print 'question: ', self.to_words([self.data['test']['Q'][i]],'Q')
                        print 'correct answer: ', self.to_words(self.data['test']['Y'][i],'Y')
                        print 'predicted answer: ', self.idx_to_word[self.data['test']['S'][i][pred]]
                        print '---' * 20

            prev_train_f1 = train_f1
    def predict(self, dataset, index):
        if not self.set_shared_variables_pointer:
            self.set_shared_variables(dataset, index,self.enable_time)
        else:
            self.set_shared_variables_pointer(dataset, index,self.enable_time)
        result=self.compute_pred()
        # print 'probas:{}\n'.format(index)
        # print result[0]
        return result[1]

    def compute_f1(self, dataset):
        n_batches = len(dataset['Y']) // self.batch_size
        # TODO: find out why not -1
        y_pred = np.concatenate([self.predict(dataset, i) for i in xrange(n_batches)]).astype(np.int32) #- 1
        # y_true = [self.vocab.index(y) for y in dataset['Y'][:len(y_pred)]]
        y_true = dataset['Y'][:len(y_pred)]
        # print metrics.confusion_matrix(y_true, y_pred)
        # print metrics.classification_report(y_true, y_pred)
        errors = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                # errors.append((i, self.lb.classes_[p]))
                errors.append((i, self.vocab[p]))
                pass
        return metrics.f1_score(y_true, y_pred, average='weighted', pos_label=None), errors

    def compute_f1_pointer(self, dataset):
        n_batches = len(dataset['Y']) // self.batch_size
        # TODO: find out why not -1
        y_pred = np.concatenate([self.predict(dataset, i) for i in xrange(n_batches)]).astype(np.int32) #- 1
        # y_true = [self.vocab.index(y) for y in dataset['Y'][:len(y_pred)]]
        # y_true = dataset['Y'][:len(y_pred)]
        y_true=[]
        for i in range(len(y_pred)):
            y_true.append(list(dataset['S'][i]).index(dataset['Y'][i]))


        # print metrics.confusion_matrix(y_true, y_pred)
        # print metrics.classification_report(y_true, y_pred)
        errors = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                # print t,p
                # errors.append((i, self.lb.classes_[p]))
                # errors.append((i, self.vocab[p]))
                errors.append((i, p))
                pass
        return metrics.f1_score(y_true, y_pred, average='weighted', pos_label=None), errors



    def shuffle_sync(self, dataset):
        p = np.random.permutation(len(dataset['Y']))
        for k in ['S', 'Q', 'Y']:
            dataset[k] = dataset[k][p]

    def to_words(self, indices,ty):
        # sents = []
        # for idx in indices:
        #     words = ' '.join([self.idx_to_word[idx] for idx in self.S[idx] if idx > 0])
        #     words = ' '.join([self.idx_to_word[idx] for i in idx)
        #     sents.append(words)
        # return ' '.join(sents)
        sent = ''
        if ty =='S':
            for idx in indices:
                sent+=self.idx_to_word[idx]
                sent+=' '
        elif ty =='Q':
            for idx in indices[0]:
                sent+=self.idx_to_word[idx]
                sent+=' '
        elif ty =='Y':
            sent=self.idx_to_word[indices]
        return sent

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--task', type=int, default=35, help='Task#')
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--back_method', type=str, default='sgd', help='Train Method to bp')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=100, help='Embedding size')
    parser.add_argument('--max_norm', type=float, default=40.0, help='Max norm')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--num_hops', type=int, default=3, help='Num hops')
    parser.add_argument('--linear_start', type='bool', default=True, help='Whether to start with linear activations')
    parser.add_argument('--shuffle_batch', type='bool', default=True, help='Whether to shuffle minibatches')
    parser.add_argument('--n_epochs', type=int, default=500, help='Num epochs')
    parser.add_argument('--enable_time', type=bool, default=False, help='time word embedding')
    parser.add_argument('--pointer_nn',type=bool,default=True,help='Whether to use the pointer networks')
    parser.add_argument('--enable_mask',type=bool,default=True,help='Whether to use the mask')
    parser.add_argument('--std_rate',type=float,default=0.5,help='The std number for the Noraml init')
    args = parser.parse_args()

    if args.train_file == '' or args.test_file == '':
        args.train_file = glob.glob('mc160.train.txt' )[0]
        args.test_file = glob.glob('mc160.test.txt' )[0]
        # args.train_file = glob.glob('*_onlyName_train.txt' )[0]
        # args.test_file = glob.glob('*_onlyName_test.txt' )[0]
        # args.train_file = glob.glob('*_sent_train.txt' )[0]
        # args.test_file = glob.glob('*_sent_test.txt' )[0]
        # args.train_file = glob.glob('*_toy_train.txt' )[0]
        # args.test_file = glob.glob('*_toy_test.txt' )[0]
        # args.train_file = '/home/shin/DeepLearning/MemoryNetwork/MemN2N_python/MemN2N-master/data/en/qqq_train.txt'
        # args.test_file ='/home/shin/DeepLearning/MemoryNetwork/MemN2N_python/MemN2N-master/data/en/qqq_test.txt'

    print '*' * 80
    print 'args:', args
    print '*' * 80
    model = Model(**args.__dict__)
    model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
    main()