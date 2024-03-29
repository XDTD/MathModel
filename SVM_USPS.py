from svmutil import *
import numpy as np


# usps实验
y, x = svm_read_problem('Data/usps')
yt, xt = svm_read_problem('Data/usps.t')
model = svm_train(y,x)
p_label, p_acc, p_val = svm_predict(yt[0:117], xt[0:117], model)
# 保存结果
mydir = 'result/usps_label.txt'
with open(mydir, 'w+') as f:
    print(p_label,file=f)
mydir = 'result/usps_acc.txt'
with open(mydir,'w+') as f:
    print(p_acc,file=f)
mydir = 'result/usps_val.txt'
with open(mydir,'w+') as f:
    print(p_val,file=f)