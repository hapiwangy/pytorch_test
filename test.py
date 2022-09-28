# # 下面的結果顯示了目前torch有提供簡單又快速的方法讓我們可以去做計算
# import numpy as np
# from sklearn.datasets import load_boston
# import torch
# X, y = load_boston(return_X_y = True)
# X_tensor = torch.from_numpy(X)
# # b = b * 1
# one = torch.ones((X.shape[0], 1))
# # 將x和one合併
# X = torch.cat((X_tensor, one), axis=1)
# w = torch.linalg.inv(X.T @ X) @ X.T @ y
# print(w)

import numpy as np
import matplotlib.pyplot as plt
# 定義損失函數和其導數。損失函數也就是目標函數
def fun(x): return 2 * x ** 4 - 3 * x ** 2 + 2 * x - 20
def dfun(x): return 8 * x ** 3 - 6 * x + 2

# 參數分別代表
# (X的起點, X的1階導數, 執行週期, 學習率)
def GD(x_start, df, epochs, lr):
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = dfun(x)
        # 更新x值
        x += - dx * lr
        xs[i+1] = x
    return xs
x_start = 5
epochs = 100000
learn_rate = 0.001

# 梯度下降法
w = GD(x_start, dfun, epochs, learn_rate)
print(np.round(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, fun(t), c='b')
plt.plot(w, fun(w), marker='o')
plt.show()
