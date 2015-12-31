# -*- coding: utf8 -*-
__author__ = 'shin'

import matlab.engine

Matlabpath='/home/shin/DeepLearning/MemoryNetwork/MemNN/MemN2N-babi-matlab'

eng = matlab.engine.start_matlab()
eng.eval('cd %s;' % (Matlabpath), nargout=0)

for i in range(1,1001):




