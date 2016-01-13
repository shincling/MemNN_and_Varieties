# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

phonelist_question=[]
phonelist_question.append('请问您的电话号码是多少？')
phonelist_question.append('请问您的电话号是？')
phonelist_question.append('请告诉我您的电话号码？')
phonelist_question.append('您好，我需要输入您的电话号码。')
phonelist_question.append('请您告诉我您的电话号码。')
phonelist_question.append('请提供乘客的电话号码。')
phonelist_question.append('乘客的电话是？')
phonelist_question.append('您的电话是多少啊？')
phonelist_question.append('请您告诉我您的电话号码。')
phonelist_question.append('先生的电话号码是多少呢？')
phonelist_question.append('您电话多少？')
phonelist_question.append('您电话号码多少？')
phonelist_question.append('我们需要知道您的电话号。')
phonelist_question.append('电话？')
phonelist_question.append('电话号？')
phonelist_question.append('电话号码？')
phonelist_question.append('电话？')
phonelist_question.append('电话号码？')
phonelist_question.append('您电话的号码？')
phonelist_question.append('请告诉我您的电话号码。')
phonelist_question.append('小姐的电话号码是多少呢？')
phonelist_question.append('还需要提供电话，谢谢。')
phonelist_question.append('电话号？谢谢。')
phonelist_question.append('麻烦您说下电话号。')
phonelist_question.append('请问您电话，谢谢。')
phonelist_question.append('好的，请告诉我您的电话号码，非常感谢。')
phonelist_question.append('没问题，您电话号码？')
phonelist_question.append('知道了，您电话的号码？')
phonelist_question.append('明白了，请告诉我您的电话号码。')
phonelist_question.append('好的，小姐的电话号码是多少呢？')
phonelist_question.append('了解了，还需要提供电话，谢谢。')
phonelist_question.append('嗯嗯，电话号？谢谢。')

phonelist_question.append('请问您的电话号码是多少？')
phonelist_question.append('请告诉我您的联系方式？')
phonelist_question.append('请问您的联系电话是？')
phonelist_question.append('请告诉我您的联系电话。')
phonelist_question.append('您好，我需要输入您的联系电话。')

phonelist_question_cut=[]
for ans in phonelist_question:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    phonelist_question_cut.append(w_sent)
pass