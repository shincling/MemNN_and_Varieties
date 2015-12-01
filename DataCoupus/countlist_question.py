# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

countlist_question=[]
countlist_question.append('您要订几张？')
countlist_question.append('您需要几张？')
countlist_question.append('您要多少张？')
countlist_question.append('您买几张？')
countlist_question.append('您要买多少张？')
countlist_question.append('您想买几张？')
countlist_question.append('您想要买多少张？')
countlist_question.append('您买几张票？')
countlist_question.append('先生买几张票？')
countlist_question.append('小姐买几张票？')
countlist_question.append('您需要订购多少张票？')
countlist_question.append('要买几张飞机票？')
countlist_question.append('几张？')
countlist_question.append('多少张？')
countlist_question.append('几张票？')
countlist_question.append('多少张票？')
countlist_question.append('需要票的数量是多少？')
countlist_question.append('您订购的数量？')
countlist_question.append('机票数量？')
countlist_question.append('您订购的数目是？')
countlist_question.append('飞机票数目？')
countlist_question.append('买几张？')
countlist_question.append('订几张？')
countlist_question.append('订购数量？')
countlist_question.append('您要订几张？谢谢。')
countlist_question.append('好的，请告诉我您要订几张？')
countlist_question.append('ok，没问题，订几张？')
countlist_question.append('您需要几张飞机票？')
countlist_question.append('请告诉我您需要多少张飞机票呢？')
countlist_question.append('您想要几张飞机票呢？')
countlist_question.append('您需要多少？谢谢。')
countlist_question.append('麻烦您说下买几张。')
countlist_question.append('知道了，买几张呀？')
countlist_question.append('明白了，麻烦说下买几张？')
countlist_question.append('好的，没问题，想要多少张呀？')
countlist_question.append('订机票吗，订几张呀？')
countlist_question.append('您要买多少张呢，谢谢。')

countlist_question_cut=[]
for ans in countlist_question:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    countlist_question_cut.append(w_sent)
pass