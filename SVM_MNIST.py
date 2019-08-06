from svmutil import *
import numpy as np


# MINIST实验
y, x = svm_read_problem('Data/mnist.scale')
yt, xt = svm_read_problem('Data/mnist.scale.t')
model = svm_train(y,x)
p_label, p_acc, p_val = svm_predict(yt[0:117], xt[0:117], model)
# 保存结果
mydir = 'result/mnist_label_scale.txt'
with open(mydir, 'w+') as f:
    print(p_label,file=f)
mydir = 'result/mnist_acc_scale.txt'
with open(mydir,'w+') as f:
    print(p_acc,file=f)
mydir = 'result/mnist_val_scale.txt'
with open(mydir,'w+') as f:
    print(p_val,file=f)