中文说明：
这个项目是自己一直在Memory Network的工作记录,大部分在仿照Facebook的bAbI数据集的格式的中文对话语料上的一些任务。
由于是工作记录的性质，所以确实比较凌乱，这里说一下一些可用的东西。

##1.数据集
主要背景是一个垂直领域下的预订机票的问题，主要集中在/DataCorpus目录下，通常.txt文件都是，以qaxx开始的是比较规范的语料。
语料设计过程中添加了很多的随机化的策略。语料生成代码也有，有心人可以找到。不过大部分情况下直接找语料应该可以有想要的形式。

##2.Matlab源码
在/MemN2N-babi-matlab目录下，主要是在原来MemN2N公开的代码上的一些修改和测试，对应的是中文语料。

##3.Python版本MemN2N的移植和拓展
在/MemN2N_python/目录下，主要是main.py文件。这是这个项目最大工作的部分。
python版本源码来源于　https://github.com/npow/MemN2N ，但是其中有一个非常非常隐秘的bug。
在标准数据集(babi)上正常使用没有问题，但是在面对随机化更严重的数据时，会有问题。



# Memory Networks

This project contains implementations of memory networks.
This includes code in the following subdirectories:


* [MemN2N-lang-model](MemN2N-lang-model): This code trains MemN2N model for language modeling, see Section 5 of the paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)". This code is implemented in [Torch7](http://torch.ch/) (written in Lua); more documentation is given in the README in that subdirectory.
 

* [MemN2N-babi-matlab](MemN2N-babi-matlab): The code for the MemN2N bAbI task experiments of Section 4 of the paper:

     [S. Sukhbaatar, A. Szlam, J. Weston, R. Fergus. End-To-End Memory Networks. arXiv:1503.08895](http://arxiv.org/abs/1503.08895).
 
  This code is implemented in Matlab; more documentation is given in the README in that subdirectory.

# Modification of mine (我的一些改动）
1. 添加了未登录词的处理方案
   在字典建立过程中，只用Train的预料，加入了10个 unknown词位，用来替换测试过程中引入的未登录词。
2. 训练函数，一直用不加softmax的来跑（这个很关键，否则根本没法跑）
3. 制作多样性的预料
   facebook Q20的语料中，词典数目实在太少了，不能体现真正的逻辑功能。没有说服力。我加入了很多完全随机的数字来替换某些中文字。并且语式也维持了多样。事实证明效果依然不错，肯定了模型的能力。
4. 还有一些trick只能在代码里体现了。

