# -*- coding: utf8 -*-
import random
def main():
    f=open(output_path,'w')
    for i in range(sent_num):
        num=i+1
        slot_num=random.randint(1,10)
        name=chr(random.randint(97, 122))+chr(random.randint(97, 122))+chr(random.randint(97, 122))+chr(random.randint(97, 122))+chr(random.randint(97, 122))
        phone=random.randint(10000000,99999999)
        question=random.choice(['phone','name'])
        ans=slot_num+3 if question=='name' else slot_num+7
        line=str(num)+'\t'
        line+='哈 '*slot_num
        line+='我的 名字 叫 {} ,'.format(name)
        line+=' 电话 是 {} 。\t'.format(phone)
        line+='{} ?\t{}\n'.format(question,ans)
        f.writelines(line)


if __name__=='__main__':
    output_path='unk_sent_test.txt'
    sent_num=1000
    main()