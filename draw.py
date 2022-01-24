import matplotlib.pyplot as plt
import os
import torch
#draw a fig from txt
f=open("eval_lovsr_8+4_80.txt")
lines=f.readlines()
epoch=[]
psnr1=[]
psnr2=[]
for line in lines:
    epoch.append(int(line.split(",")[0].split(":")[-1]))
    psnr1.append(float(line.split(",")[1].split("[")[-1]))
    psnr2.append(float(line.split(",")[-1].split("]")[0]))
fig = plt.figure()
plt.title("psnr graph")
plt.ylim(28,31)
plt.plot(epoch,psnr1,label="psnr1")
plt.plot(epoch,psnr2,label="psnr2")
plt.xlabel('PSNR')
plt.ylabel("epoch")
plt.savefig(os.path.join('./out/eval.png'))
plt.close(fig)


