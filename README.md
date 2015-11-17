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

