from PIL import Image
import numpy as np
#将训练数据转化的工具
global iner
iner= []
def test():
        #将test.png转化成数组
        img = np.array(Image.open('test.png').convert("L"))
        x,y = img.shape
        for i in range(x):
                for j in range(y):
                        r = img[i,j]
                        iner.append(r)
def train(num,data):
        #用于生产训练用数据
        img = np.array(Image.open("train_data//"+str(num+1)+'.png').convert("L"))
        f = open("data//{0}.txt".format(num+1), 'w+')  
        x,y = img.shape
        #flag = int(input("num?\n"))
        flag = data
        print(flag,file=f,end=",")
        for i in range(x):
                for j in range(y):
                        r = img[i,j]
                        iner.append(r)
                        print(r,file=f,end=",") if i!=9 or j!=9 else print(r,file=f,end="")
        f.close()
global das
das = [1,2,3,4,5,6,7,8,9,0,
       6,2,4,9,3,1,0,8,5,9,
       4,4,3,3,6,7,0,1
       ]
def maker(num):
        for i in range(len(num)):
                train(i,num[i])
