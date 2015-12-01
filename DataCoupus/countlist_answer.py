# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

countlist_answer=[]
'''
count=<count_Answer>
count=<count_Answer>就够了。
count=我要<count_Answer>机票。
count=我想订<count_Answer>。
count=买<count_Answer>票就可以。
count=机票数量是<count_Answer>。
count=想订购<count_Answer>机票。
count=<count_Answer>即可。
count=可能需要<count_Answer>。

[槽_count]。
[槽_count]张就可以了。
帮我预定[槽_count]张。
[槽_count]张。
[槽_count]张机票。
'''
countlist_answer.append('一张')
countlist_answer.append('一张。')
countlist_answer.append('一张就够了。')
countlist_answer.append('我要一张机票。')
countlist_answer.append('我想订一张。')
countlist_answer.append('买一张票就可以。')
countlist_answer.append('机票数量是一张。')
countlist_answer.append('想订购一张机票。')
countlist_answer.append('一张即可。')
countlist_answer.append('可能需要一张。')
countlist_answer.append('一张就可以了。')
countlist_answer.append('帮我预定一张。')
countlist_answer.append('一张机票。')
countlist_answer_cut=[]
for ans in countlist_answer:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    w_sent=w_sent.replace('一张'.decode('utf8'),'[slot_count]')
    countlist_answer_cut.append(w_sent)
pass