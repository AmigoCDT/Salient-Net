# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np


log_file = "./se_weight_salientnet50-imagenet.log"
train_acc = []
train_5_acc = []
val_acc = []
val_5_acc = []
with open(log_file, "r") as f:
    for line_cont in f.readlines():
        if "10000/10010" in line_cont:
            # train_acc.append("")
            train_cont = line_cont.split("Acc@1")[-1].split("(")
            # print(train_cont)
            _train_acc = float(train_cont[1].split(")")[0].replace(" ", ""))
            _train_5_acc = float(train_cont[2].replace(")", "").replace(" ", ""))
            train_acc.append(_train_acc)
            train_5_acc.append(_train_5_acc)
        elif "*" in line_cont:
            #  * Acc@1 66.466 Acc@5 87.782
            _val_cont = line_cont.split(" Acc")
            _val_acc = float(_val_cont[1].split(" ")[-1])
            _val_5_acc = float(_val_cont[-1].split(" ")[-1])
            val_acc.append(_val_acc)
            val_5_acc.append(_val_5_acc)
        else:
            pass

assert len(train_acc)==len(val_acc)
epochs = list(range(len(val_acc)))

print(train_acc)
print(val_acc)

plt.title("SSANet-Imagenet") 
plt.xlabel("epochs") 
plt.ylabel("acc@1") 
plt.plot(epochs, train_acc, "r-", epochs, val_acc, "r--")
plt.show()
