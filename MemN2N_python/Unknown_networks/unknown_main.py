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



def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--task', type=int, default=35, help='Task#')
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