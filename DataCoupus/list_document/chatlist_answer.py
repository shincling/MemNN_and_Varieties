# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

chatlist_answer=[]
chatlist_answer.append('稍等，让我想想。')
chatlist_answer.append('让我考虑一下。')
chatlist_answer.append('等会儿，我想想。')
'''
chatlist_answer.append('一张')
chatlist_answer.append('一张票')
chatlist_answer.append('一张机票')
chatlist_answer.append('我要买一张票')
chatlist_answer.append('我要买一张机票')
chatlist_answer.append('帮我订一张')
chatlist_answer.append('预订一张')
chatlist_answer.append('请帮我预订一张')
'''
chatlist_answer_cut=[]
for ans in chatlist_answer:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    w_sent=w_sent.replace('一张'.decode('utf8'),'[slot_chat]')
    chatlist_answer_cut.append(w_sent)
pass