#%%
import torch
import numpy as np
import math
#%%
v = np.array([1,2])
# 計算長度
length = np.linalg.norm(v)
# torch裡面也有，只是參數要擺torch的array(不是np的)
# length = torch.linalg.norm(torch.FloatTensor(v))
# %%
# 計算方向(透過tan的反函數求角度)
# 先求出tan
tan = v[1] / v[0]
# 再用math.atan求出角度(單位弳度)
atan = math.atan(tan)
print(atan)
# %%
# 四則運算
# 如果是和常數的話，對"所有"元素都做運算
# 如果是和張量的話，對"同位置"元素做運算
# %%
# 內積 以@作為運算符號
# 這個要用在兩個np array物件上面
a = np.array([2,1])
b = np.array([-3,2])
print(a@b)
# %%
# 計算夾角
# 先算內積在除以兩個的長度得到cos(sita)
# 再用acos得到角度 弳度
# %%
# 取得轉置矩陣
A = np.array([[1,2,3],[4,5,6]])
print(A.T)
# 或是使用
print(np.transpose(A))
# %%
# 反矩陣
A = np.array([[1,2],[3,4]])
print(np.linalg.inv(A))
# %%
torch.cuda.is_available()
# %%
