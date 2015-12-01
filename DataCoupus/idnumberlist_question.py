# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

idnumberlist_question=[]
idnumberlist_question.append('请问您的身份证号码是多少？')
idnumberlist_question.append('请问您的身份证号是？')
idnumberlist_question.append('请告诉我您的身份证号码？')
idnumberlist_question.append('您好，我需要输入您的身份证号码。')
idnumberlist_question.append('请您告诉我您的身份证号码。')
idnumberlist_question.append('请提供乘客的身份证号码。')
idnumberlist_question.append('乘客的身份证是？')
idnumberlist_question.append('您的身份证是多少啊？')
idnumberlist_question.append('请您告诉我您的身份证号码。')
idnumberlist_question.append('先生的身份证号码是多少呢？')
idnumberlist_question.append('您身份证多少？')
idnumberlist_question.append('您身份证号码多少？')
idnumberlist_question.append('我们需要知道您的身份证号。')
idnumberlist_question.append('身份证？')
idnumberlist_question.append('身份证号？')
idnumberlist_question.append('身份证号码？')
idnumberlist_question.append('身份证件？')
idnumberlist_question.append('身份证件号码？')
idnumberlist_question.append('您身份证的号码？')
idnumberlist_question.append('请告诉我您的身份证号码。')
idnumberlist_question.append('小姐的身份证号码是多少呢？')
idnumberlist_question.append('还需要提供身份证件，谢谢。')
idnumberlist_question.append('身份证号？谢谢。')
idnumberlist_question.append('麻烦您说下身份证号。')
idnumberlist_question.append('请问您身份证，谢谢。')
idnumberlist_question.append('好的，请告诉我您的身份证号码，非常感谢。')
idnumberlist_question.append('没问题，您身份证件号码？')
idnumberlist_question.append('知道了，您身份证的号码？')
idnumberlist_question.append('明白了，请告诉我您的身份证号码。')
idnumberlist_question.append('好的，小姐的身份证号码是多少呢？')
idnumberlist_question.append('了解了，还需要提供身份证件，谢谢。')
idnumberlist_question.append('嗯嗯，身份证号？谢谢。')

idnumberlist_question_cut=[]
for ans in idnumberlist_question:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    idnumberlist_question_cut.append(w_sent)
pass