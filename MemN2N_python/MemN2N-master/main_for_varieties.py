# -*- coding: utf8 -*-
from __future__ import division
import argparse
import re
import glob
import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer,label_binarize
import port_list


import warnings
warnings.filterwarnings('ignore', '.*topo.*')

def count_logic(test_predict,n_status=8,n_q=7):
    correct_status=0
    correct_logic_reply=0
    correct_answer=0

    y_true=[i[2] for i in test_predict]
    story_slots=y_true.index('0000000',1)
    story_total=int(len(test_predict)/story_slots)
    for idx in range(int(story_total)):
        story=test_predict[idx*story_slots:(idx+1)*story_slots]
        for j in range(n_status):
            status_target=story[j*2][2]
            status_predict=story[j*2][1]
            language_predict=story[j*2+1][1]
            if status_target==status_predict:
                correct_status+=1

            '''开始判断logic_reply'''
            rest_list=['已经为您预订完毕。']
            if status_target[0]=='0':
                rest_list.extend(port_list.namelist)
            if status_target[1]=='0':
                rest_list.extend(port_list.countlist)
            if status_target[2]=='0':
                rest_list.extend(port_list.departurelist)
            if status_target[3]=='0':
                rest_list.extend(port_list.destinationlist)
            if status_target[4]=='0':
                rest_list.extend(port_list.timelist)
            if status_target[5]=='0':
                rest_list.extend(port_list.idnumberlist)
            if status_target[6]=='0':
                rest_list.extend(port_list.phonelist)

            if language_predict in rest_list:
                correct_logic_reply+=1

        for j in range(n_q):
            answer_predict,answer_target=story[-(j+1)][1],story[-(j+1)][2]
            if answer_predict==answer_target:
                correct_answer+=1

    print 'number of total story',story_total
    print 'number of correct status:',correct_status,correct_status/(story_total*n_status)
    print 'number of correct logic reply:',correct_logic_reply,correct_logic_reply/(story_total*n_status)
    print 'number of correct answer:',correct_answer,correct_answer/(story_total*n_q)

    pass

class InnerProductLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, nonlinearity=None, **kwargs):
        super(InnerProductLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        M = inputs[0]
        u = inputs[1]
        output = T.batched_dot(M, u)
        if self.nonlinearity is not None:
            output = self.nonlinearity(output)
        return output


class BatchedDotLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])


class SumLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axis, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis+1:]

    def get_output_for(self, input, **kwargs):
        return T.sum(input, axis=self.axis)


class TemporalEncodingLayer(lasagne.layers.Layer):
    '''对应了论文里的Temporal Encoding部分，引入了T_A或者T_C参数来学习'''
    def __init__(self, incoming, T=lasagne.init.Normal(std=0.1), **kwargs):
        super(TemporalEncodingLayer, self).__init__(incoming, **kwargs)
        self.T = self.add_param(T, self.input_shape[-2:], name="T")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input + self.T


class TransposedDenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        activation = T.dot(input, self.W.T)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class MemoryNetworkLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, vocab, embedding_size,enable_time, A, A_T, C, C_T, nonlinearity=lasagne.nonlinearities.softmax, **kwargs):
        super(MemoryNetworkLayer, self).__init__(incomings, **kwargs) #？？？不知道这个super到底做什么的，会引入input_layers和input_shapes这些属性
        if len(incomings) != 3:
            raise NotImplementedError

        batch_size, max_seqlen, max_sentlen = self.input_shapes[0]

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))
        l_B_embedding = lasagne.layers.InputLayer(shape=(batch_size, embedding_size))
        l_context_pe_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen, embedding_size))

        l_context_in = lasagne.layers.ReshapeLayer(l_context_in, shape=(batch_size * max_seqlen * max_sentlen, )) #reshape，拼成了一行
        l_A_embedding = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=A)
        self.A = l_A_embedding.W
        l_A_embedding = lasagne.layers.ReshapeLayer(l_A_embedding, shape=(batch_size, max_seqlen, max_sentlen, embedding_size)) #reshape把一行的东西再转回来
        l_A_embedding = lasagne.layers.ElemwiseMergeLayer((l_A_embedding, l_context_pe_in), merge_function=T.mul)
        l_A_embedding = SumLayer(l_A_embedding, axis=2) #同样地，把一个句子里按所有词加起来
        if not enable_time:
            l_A_embedding = TemporalEncodingLayer(l_A_embedding, T=A_T)
            self.A_T = l_A_embedding.T

        l_C_embedding = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=C)
        # l_C_embedding = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=self.A)
        self.C = l_C_embedding.W
        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        l_C_embedding = lasagne.layers.ElemwiseMergeLayer((l_C_embedding, l_context_pe_in), merge_function=T.mul)
        l_C_embedding = SumLayer(l_C_embedding, axis=2)
        if not enable_time:
            l_C_embedding = TemporalEncodingLayer(l_C_embedding, T=C_T)
            # l_C_embedding = TemporalEncodingLayer(l_C_embedding, T=self.A_T)
            self.C_T = l_C_embedding.T
        '''注意这底下的几个层都是暂时直接实例化，至进行了init，具体的操作需要用到各个类里面的函数来计算'''
        l_prob = InnerProductLayer((l_A_embedding, l_B_embedding), nonlinearity=nonlinearity) #32*10*20 和 32*20*1结合，成32*10
        l_weighted_output = BatchedDotLayer((l_prob, l_C_embedding))

        l_sum = lasagne.layers.ElemwiseSumLayer((l_weighted_output, l_B_embedding))

        self.l_context_in = l_context_in
        self.l_B_embedding = l_B_embedding
        self.l_context_pe_in = l_context_pe_in
        self.network = l_sum
        '''注册了一层的四个参数进去，分别是A,A_T,C,C_T ？？？有个问题是，不知道A_T C_T什么时候初始化了'''
        params = lasagne.layers.helper.get_all_params(self.network, trainable=True)
        values = lasagne.layers.helper.get_all_param_values(self.network, trainable=True)
        for p, v in zip(params, values):
            self.add_param(p, v.shape, name=p.name)

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A,self.C]])

    def get_output_shape_for(self, input_shapes):
        return lasagne.layers.helper.get_output_shape(self.network)

    def get_output_for(self, inputs, **kwargs):
        return lasagne.layers.helper.get_output(self.network, {self.l_context_in: inputs[0], self.l_B_embedding: inputs[1], self.l_context_pe_in: inputs[2]})

    def reset_zero(self):
        self.set_zero(self.zero_vec)


class Model:
    def __init__(self, train_file, test_file, batch_size=32, embedding_size=20, max_norm=40, lr=0.01, num_hops=3, adj_weight_tying=True, linear_start=True, enable_time=False,**kwargs):
        train_lines, test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([train_lines, test_lines], axis=0) #直接头尾拼接
        vocab, word_to_idx, idx_to_word, max_seqlen, max_sentlen = self.get_vocab(lines)
        #C是document的列表，Q是定位问题序列的列表，Y是答案组成的列表，目前为知都是字符形式的，没有向量化#
        self.data = {'train': {}, 'test': {}}  #各是一个字典
        S_train, self.data['train']['C'], self.data['train']['Q'], self.data['train']['Y'] = self.process_dataset(train_lines, word_to_idx, max_sentlen, offset=0)
        S_test, self.data['test']['C'], self.data['test']['Q'], self.data['test']['Y'] = self.process_dataset(test_lines, word_to_idx, max_sentlen, offset=len(S_train))
        S = np.concatenate([np.zeros((1, max_sentlen), dtype=np.int32), S_train, S_test], axis=0)
        for i in range(min(10,len(self.data['test']['C']))):
            for k in ['C', 'Q', 'Y']:
                print k, self.data['test'][k][i]
        print 'batch_size:', batch_size, 'max_seqlen:', max_seqlen, 'max_sentlen:', max_sentlen
        print 'sentences:', S.shape
        print 'vocab size:', len(vocab)

        for d in ['train', 'test']:
            print d,
            for k in ['C', 'Q', 'Y']:
                print k, self.data[d][k].shape,
            print ''

        vocab=[]
        for i in range(len(idx_to_word)):
            vocab.append(idx_to_word[i+1])


        lb = LabelBinarizer()
        # lb.fit(list(vocab))
        # vocab = lb.classes_.tolist()


        self.enable_time=enable_time
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.max_sentlen = max_sentlen if not enable_time else max_sentlen+1
        self.embedding_size = embedding_size
        self.num_classes = len(vocab) + 1
        self.vocab = vocab
        self.adj_weight_tying = adj_weight_tying
        self.num_hops = num_hops
        self.lb = lb
        self.init_lr = lr
        self.lr = self.init_lr
        self.max_norm = max_norm
        self.S = S
        self.idx_to_word = idx_to_word
        self.nonlinearity = None if linear_start else lasagne.nonlinearities.softmax
        self.word_to_idx=word_to_idx

        # self.build_network(self.nonlinearity)
        self.build_other_network(self.nonlinearity)

    def build_network(self, nonlinearity):
        batch_size, max_seqlen, max_sentlen, embedding_size, vocab,enable_time = self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size, self.vocab,self.enable_time

        c = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        c_pe = T.tensor4()
        q_pe = T.tensor4()
        self.c_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
        self.q_shared = theano.shared(np.zeros((batch_size, ), dtype=np.int32), borrow=True)
        '''最后的softmax层的参数'''
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)
        self.c_pe_shared = theano.shared(np.zeros((batch_size, max_seqlen, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        self.q_pe_shared = theano.shared(np.zeros((batch_size, 1, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        S_shared = theano.shared(self.S, borrow=True)#这个S把train test放到了一起来干事情#

        if enable_time:
            pass

        cc = S_shared[c.flatten()].reshape((batch_size, max_seqlen, max_sentlen))
        qq = S_shared[q.flatten()].reshape((batch_size, max_sentlen))

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))
        l_question_in = lasagne.layers.InputLayer(shape=(batch_size, max_sentlen))
        #大概可以判断，这个pe就是原文里面的合成一个用到的分布矩阵#
        l_context_pe_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        l_question_pe_in = lasagne.layers.InputLayer(shape=(batch_size, 1, max_sentlen, embedding_size))
        '''底下这几部分是在初始化映射矩阵'''
        # A, C = lasagne.init.Normal(std=0.1).sample((len(vocab)+1, embedding_size)), lasagne.init.Normal(std=0.1)
        # B, C = lasagne.init.Normal(std=0.1).sample((len(vocab)+1, embedding_size)), lasagne.init.Normal(std=0.1)
        B, C = lasagne.init.Normal(std=0.1), lasagne.init.Normal(std=0.1)
        A_T, C_T = lasagne.init.Normal(std=0.1), lasagne.init.Normal(std=0.1)
        W = B if self.adj_weight_tying else lasagne.init.Normal(std=0.1) #这里决定了原文里的 A与B两个映射矩阵相同

        l_question_in = lasagne.layers.ReshapeLayer(l_question_in, shape=(batch_size * max_sentlen, ))
        l_B_embedding = lasagne.layers.EmbeddingLayer(l_question_in, len(vocab)+1, embedding_size, W=W) #到这变成了224*20
        B = l_B_embedding.W #B就是上一行初始化用到的W,是(len(vocab)+1, embedding_size)这种size
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding, shape=(batch_size, 1, max_sentlen, embedding_size)) #reshape变成了32*1*7*20
        l_B_embedding = lasagne.layers.ElemwiseMergeLayer((l_B_embedding, l_question_pe_in), merge_function=T.mul)
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding, shape=(batch_size, max_sentlen, embedding_size))
        l_B_embedding = SumLayer(l_B_embedding, axis=1)
        #这是个初始化第一层，后面的层在循环里动态连接了#
        self.mem_layers = [MemoryNetworkLayer((l_context_in, l_B_embedding, l_context_pe_in), vocab, embedding_size,enable_time, A=B, A_T=A_T, C=C, C_T=C_T, nonlinearity=nonlinearity)]
        for _ in range(1, self.num_hops):
            if self.adj_weight_tying:
                A, C = self.mem_layers[-1].C, lasagne.init.Normal(std=0.1)
                if not enable_time:
                    A_T, C_T = self.mem_layers[-1].C_T, lasagne.init.Normal(std=0.1)
                # A, C = self.mem_layers[-1].C,self.mem_layers[-1].C
                # if not enable_time:
                #     A_T, C_T = self.mem_layers[-1].C_T, self.mem_layers[-1].C_T
                # A=lasagne.init.Normal(std=0.1)
                # C=A
                # if not enable_time:
                #     A_T=lasagne.init.Normal(std=0.1)
                #     C_T=A_T
            else:  # RNN style
                A, C = self.mem_layers[-1].A, self.mem_layers[-1].C
                if not enable_time:
                    A_T, C_T = self.mem_layers[-1].A_T, self.mem_layers[-1].C_T
            self.mem_layers += [MemoryNetworkLayer((l_context_in, self.mem_layers[-1], l_context_pe_in), vocab, embedding_size, enable_time,A=A, A_T=A_T, C=C, C_T=C_T, nonlinearity=nonlinearity)]

        if True and self.adj_weight_tying:
            # l_pred = TransposedDenseLayer(self.mem_layers[-1], self.num_classes, W=self.mem_layers[-1].C, b=None, nonlinearity=lasagne.nonlinearities.softmax)
            l_pred = TransposedDenseLayer(self.mem_layers[-1], 1, W=self.mem_layers[-1].C, b=None, nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_pred = lasagne.layers.DenseLayer(self.mem_layers[-1], self.num_classes, W=lasagne.init.Normal(std=0.1), b=None, nonlinearity=lasagne.nonlinearities.softmax)

        c_emb = lasagne.layers.helper.get_output(self.mem_layers[-1],{l_context_in: cc, l_question_in: qq, l_context_pe_in: c_pe, l_question_pe_in: q_pe})
        probas = lasagne.layers.helper.get_output(l_pred, {l_context_in: cc, l_question_in: qq, l_context_pe_in: c_pe, l_question_pe_in: q_pe})
        # probas = lasagne.layers.helper.get_output(l_pred,None)
        probas = T.clip(probas, 1e-7, 1.0-1e-7)

        pred = T.argmax(probas, axis=1)

        cost = T.nnet.categorical_crossentropy(probas, y).sum()

        params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
        print 'params:', params
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
        updates = lasagne.updates.sgd(scaled_grads, params, learning_rate=self.lr)

        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared,
            c_pe: self.c_pe_shared,
            q_pe: self.q_pe_shared
        }

        self.train_model = theano.function([], cost, givens=givens, updates=updates)
        self.compute_pred = theano.function([], outputs= [pred,probas,c_emb], givens=givens, on_unused_input='ignore')

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [B]])

        self.nonlinearity = nonlinearity
        self.network = l_pred

    def build_other_network(self, nonlinearity):
        batch_size, max_seqlen, max_sentlen, embedding_size, vocab,enable_time = self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size, self.vocab,self.enable_time

        c = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        c_pe = T.tensor4()
        q_pe = T.tensor4()
        self.c_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
        self.q_shared = theano.shared(np.zeros((batch_size, ), dtype=np.int32), borrow=True)
        '''最后的softmax层的参数'''
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)
        self.c_pe_shared = theano.shared(np.zeros((batch_size, max_seqlen, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        self.q_pe_shared = theano.shared(np.zeros((batch_size, 1, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        S_shared = theano.shared(self.S, borrow=True)#这个S把train test放到了一起来干事情#

        if enable_time:
            pass

        cc = S_shared[c.flatten()].reshape((batch_size, max_seqlen, max_sentlen))
        qq = S_shared[q.flatten()].reshape((batch_size, max_sentlen))

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))
        W, C = lasagne.init.Normal(std=0.1), lasagne.init.Normal(std=0.1)

        l_context_in = lasagne.layers.ReshapeLayer(l_context_in, shape=(batch_size* max_seqlen*max_sentlen,))
        l_context_embedding_0 = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=W) #到这变成了224*20
        l_context_embedding = lasagne.layers.ReshapeLayer(l_context_embedding_0,(batch_size,max_sentlen*max_seqlen,embedding_size))
        l_context_lstm=lasagne.layers.LSTMLayer(l_context_embedding,embedding_size)
        l_context_layer=lasagne.layers.SliceLayer(l_context_lstm,-1,1)

        l_question_in = lasagne.layers.InputLayer(shape=(batch_size, max_sentlen))
        l_question_in = lasagne.layers.ReshapeLayer(l_question_in,shape=(batch_size*max_sentlen,))
        l_question_embedding = lasagne.layers.EmbeddingLayer(l_question_in, len(vocab)+1, embedding_size,W=l_context_embedding_0.W) #reshape变成了32*1*7*20
        l_question_embedding = lasagne.layers.ReshapeLayer(l_question_embedding, shape=(batch_size, max_sentlen, embedding_size))
        l_question_layer=lasagne.layers.LSTMLayer(l_question_embedding,embedding_size)
        l_question_layer=lasagne.layers.SliceLayer(l_question_layer,-1,1)


        l_pred=lasagne.layers.ElemwiseMergeLayer((l_context_layer,l_question_layer),T.mul)
        # l_context_layer = lasagne.layers.ReshapeLayer(l_context_layer,(batch_size*embedding_size,))
        # l_question_layer = lasagne.layers.ReshapeLayer(l_question_layer,(batch_size*embedding_size,))
        # l_pred=lasagne.layers.ElemwiseMergeLayer((l_context_layer,l_question_layer),T.sum)
        # l_pred = lasagne.layers.ReshapeLayer(l_pred,(batch_size,embedding_size))


        probas = lasagne.layers.helper.get_output(l_pred, {l_context_in: cc, l_question_in: qq })
        # probas = lasagne.layers.helper.get_output(l_pred,None)
        probas = T.clip(probas, 1e-7, 1.0-1e-7)

        pred = T.argmax(probas, axis=1)

        cost = T.nnet.categorical_crossentropy(probas, y).sum()

        params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
        print 'params:', params
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
        updates = lasagne.updates.sgd(scaled_grads, params, learning_rate=self.lr)

        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared,
        }

        self.train_model = theano.function([], cost, givens=givens, updates=updates)
        self.compute_pred = theano.function([], outputs= [pred,probas], givens=givens, on_unused_input='ignore')

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [B]])

        self.nonlinearity = nonlinearity
        self.network = l_pred



    def reset_zero(self):
        self.set_zero(self.zero_vec)
        for l in self.mem_layers:
            l.reset_zero()

    def predict(self, dataset, index):
        self.set_shared_variables(dataset, index,self.enable_time)
        result=self.compute_pred()
        # print 'probas:\n'
        # print result[1]
        return result[0]

    def compute_f1(self, dataset):
        n_batches = len(dataset['Y']) // self.batch_size
        y_pred = np.concatenate([self.predict(dataset, i) for i in xrange(n_batches)]).astype(np.int32) - 1
        '''为什么上面要减1呢，因为predict的时候，用到的都是len(vocab)+1的那个东西，而vocab.index是从没加补位符开始的
        也就是说如果predict到了词典的第一个词，那么y_pred应该是1，-1之后才是vocab.index(0)对应的结果。'''

        y_true = [self.vocab.index(y) for y in dataset['Y'][:len(y_pred)]]
        # print metrics.confusion_matrix(y_true, y_pred)
        # print metrics.classification_report(y_true, y_pred)
        errors = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                # errors.append((i, self.lb.classes_[p]))
                errors.append((i, self.vocab[p]))
                pass
        return metrics.f1_score(y_true, y_pred, average='weighted', pos_label=None), errors, [(idx,self.vocab[y[0]],self.vocab[y[1]]) for idx,y in enumerate(zip(y_pred,y_true))]

    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        n_train_batches = len(self.data['train']['Y']) // self.batch_size
        self.lr = self.init_lr
        prev_train_f1 = None

        while (epoch < n_epochs):
            epoch += 1
            # if epoch and epoch%20==0:
            #     tmp=lasagne.layers.helper.get_all_param_values(self.network)
            #     for idx,i in enumerate(tmp):
            #         print '*'*50
            #         print idx
            #         print i



            if epoch % 25 == 0: #每隔25个epoch，则速率减半
                self.lr /= 2.0

            indices = range(n_train_batches)
            if shuffle_batch:
                self.shuffle_sync(self.data['train'])#保持对应性地一次性shuffle了C Q Y

            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:#一次进入一个batch的数据
                self.set_shared_variables(self.data['train'], minibatch_index,self.enable_time)#这里的函数总算把数据传给了模型里面初始化的变量
                total_cost += self.train_model()
                self.reset_zero()  #reset是把A,C的第一行（也就是第一个元素，对应字典了的第一个词）reset了一次，变成了0
            end_time = time.time()
            print '\n' * 3, '*' * 80
            print 'epoch:', epoch, 'cost:', (total_cost / len(indices)), ' took: %d(s)' % (end_time - start_time)

            print 'TRAIN', '=' * 40
            train_f1, train_errors, _ = self.compute_f1(self.data['train'])
            print 'TRAIN_ERROR:', (1-train_f1)*100
            if False:
                for i, pred in train_errors[:10]:
                    print 'context: ', self.to_words(self.data['train']['C'][i])
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
                test_f1, test_errors, test_predict = self.compute_f1(self.data['test']) #有点奇怪这里的f1和test_error怎么好像不对应的？
                print 'test_f1,test_errors:',test_f1,len(test_errors)
                print '*** TEST_ERROR:', (1-test_f1)*100
                if 0 :
                    for i, pred in test_errors[:10]:
                        print 'context: ', self.to_words(self.data['test']['C'][i])
                        print 'question: ', self.to_words([self.data['test']['Q'][i]])
                        print 'correct answer: ', self.data['test']['Y'][i]
                        print 'predicted answer: ', pred
                        print '---' * 20
                if 1 :
                    count_logic(test_predict)

            prev_train_f1 = train_f1

    def to_words(self, indices):
        sents = []
        for idx in indices:
            words = ' '.join([self.idx_to_word[idx] for idx in self.S[idx] if idx > 0])
            sents.append(words)
        return ' '.join(sents)

    def shuffle_sync(self, dataset):
        p = np.random.permutation(len(dataset['Y']))
        for k in ['C', 'Q', 'Y']:
            dataset[k] = dataset[k][p]

    def set_shared_variables(self, dataset, index,enable_time):
        c = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
        q = np.zeros((self.batch_size, ), dtype=np.int32)
        y = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)
        c_pe = np.zeros((self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)
        q_pe = np.zeros((self.batch_size, 1, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)
        # c_pe = np.ones((self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)
        # q_pe = np.ones((self.batch_size, 1, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i, row in enumerate(dataset['C'][indices]):
            row = row[:self.max_seqlen]
            c[i, :len(row)] = row

        q[:len(indices)] = dataset['Q'][indices] #问题的行数组成的列表
        '''底下这个整个循环是得到一个batch对应的那个调整的矩阵'''
        for key, mask in [('C', c_pe), ('Q', q_pe)]:
            for i, row in enumerate(dataset[key][indices]):
                sentences = self.S[row].reshape((-1, self.max_sentlen)) #这句相当于把每一句，从标号变成具体的词，并补0
                for ii, word_idxs in enumerate(sentences):
                    J = np.count_nonzero(word_idxs)
                    for j in np.arange(J):
                        mask[i, ii, j, :] = (1 - (j+1)/J) - ((np.arange(self.embedding_size)+1)/self.embedding_size)*(1 - 2*(j+1)/J)

        # c_pe=np.not_equal(c_pe,0)
        # q_pe=np.not_equal(q_pe,0)

        # y[:len(indices), 1:self.num_classes] = self.lb.transform(dataset['Y'][indices])#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        y[:len(indices), 1:self.num_classes] = label_binarize(dataset['Y'][indices],self.vocab)#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        # y[:len(indices), 1:self.embedding_size] = self.mem_layers[0].A[[self.word_to_idx(i) for i in list(dataset['Y'][indices])]]#竟然是把y变成了而之花的one=hot向量都，每个是字典大小这么长
        self.c_shared.set_value(c)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)
        self.c_pe_shared.set_value(c_pe)
        self.q_pe_shared.set_value(q_pe)

    def get_vocab(self, lines): #这个函数相当于预处理的函数
        vocab = set()
        max_sentlen = 0
        for i, line in enumerate(lines):
            #words = nltk.word_tokenize(line['text'])
            words=line['text'].split(' ')  #这里做了一个修改，替换掉了nltk的工具
            max_sentlen = max(max_sentlen, len(words))
            for w in words:
                vocab.add(w)
            if line['type'] == 'q':
                vocab.add(line['answer'])

        word_to_idx = {}
        for w in vocab:
            word_to_idx[w] = len(word_to_idx) + 1

        idx_to_word = {}
        for w, idx in word_to_idx.iteritems():
            idx_to_word[idx] = w

        max_seqlen = 0
        for i, line in enumerate(lines):
            if line['type'] == 'q':
                id = line['id']-1
                indices = [idx for idx in range(i-id, i) if lines[idx]['type'] == 's'][::-1][:50]
                #上面这个表达式倒是挺优美的，就是算出了一个问题对应的从句子开始到它的序号处所能得到的非问句的长度
                max_seqlen = max(len(indices), max_seqlen)

        return vocab, word_to_idx, idx_to_word, max_seqlen, max_sentlen

    def process_dataset(self, lines, word_to_idx, max_sentlen, offset):
        S, C, Q, Y = [], [], [], []
        '''S是每句序号化之后组成的列表的列表，
        C是从后往前排列的每一问题的story，
        Q是问题的行数组成的列表，
        Y是答案组成的列表'''
        for i, line in enumerate(lines):
            word_indices = [word_to_idx[w] for w in line['text'].split(' ')]
            word_indices += [0] * (max_sentlen - len(word_indices)) #这是补零，把句子填充到max_sentLen
            S.append(word_indices)
            if line['type'] == 'q':
                id = line['id']-1
                indices = [offset+idx+1 for idx in range(i-id, i) if lines[idx]['type'] == 's'][::-1][:50]
                line['refs'] = [indices.index(offset+i+1-id+ref) for ref in line['refs']]
                C.append(indices)
                Q.append(offset+i+1)
                Y.append(line['answer'])
        return np.array(S, dtype=np.int32), np.array(C), np.array(Q, dtype=np.int32), np.array(Y)

    def get_lines(self, fname):
        lines = [] #每个元素是个字典看来
        for i, line in enumerate(open(fname)):
            id = int(line[0:line.find(' ')]) #找到每行的对应的故事里的序号
            line = line.strip()
            line = line[line.find(' ')+1:] #去掉序号开始到末尾

            if line.find('?') == -1: #如果不是问句
                lines.append({'type': 's', 'text': line})
            else: #如果是问句
                idx = line.find('?')
                tmp = line[idx+1:].split('\t')
                # lines.append({'id': id, 'type': 'q', 'text': line[:idx], 'answer': tmp[1].strip(), 'refs': [int(x)-1 for x in tmp[2:][0].split(' ')]}) #这里为什么int(x)要减去1？？？
                lines.append({'id': id, 'type': 'q', 'text': line[:idx], 'answer': tmp[1].strip(), 'refs': [0]}) #这里为什么int(x)要减去1？？？
            if False and i > 1000:
                break
        return np.array(lines)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')




def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--task', type=int, default=331, help='Task#')
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=100, help='Embedding size')
    parser.add_argument('--max_norm', type=float, default=40.0, help='Max norm')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_hops', type=int, default=3, help='Num hops')
    parser.add_argument('--adj_weight_tying', type='bool', default=True, help='Whether to use adjacent weight tying')
    parser.add_argument('--linear_start', type='bool', default=True, help='Whether to start with linear activations')
    parser.add_argument('--shuffle_batch', type='bool', default=True, help='Whether to shuffle minibatches')
    parser.add_argument('--n_epochs', type=int, default=500, help='Num epochs')
    parser.add_argument('--enable_time', type=bool, default=False, help='time word embedding')
    args = parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80

    if args.train_file == '' or args.test_file == '':
        args.train_file = glob.glob('data/en/qa%d_*train.txt' % args.task)[0]
        args.test_file = glob.glob('data/en/qa%d_*test.txt' % args.task)[0]
        # args.train_file = '/home/shin/DeepLearning/MemoryNetwork/MemN2N_python/MemN2N-master/data/en/qqq_train.txt'
        # args.test_file ='/home/shin/DeepLearning/MemoryNetwork/MemN2N_python/MemN2N-master/data/en/qqq_test.txt'

    model = Model(**args.__dict__)
    model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
    main()
