# 這裡的架構可以熟悉，以後用在其他地方也很好用
# 載入套件
import numpy as np
from regex import B
import torch
import matplotlib.pyplot as plt

# 定義訓練函數
def train(X, y, epochs=100, lr = 0.001):
    loss_list, w_list, b_list=[], [], []

    # 設定w, b為常態分布之隨機亂數
    w = torch.randn(1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    for epoch in range(epochs):
        y_pred = w * X + b

        # 計算loss(這裡用loss_function用MSE)
        ## 可以嘗試其他的loss function，其他地方照舊即可
        MSE = torch.mean(torch.square(y_pred - y))
        MSE.backward()

        # 設定不參與梯度下降，w, b才能進行運算
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            # detach:與運算圖分離，numpy()轉成numpy陣列
            # w.detach().numpy()
            w_list.append(w.item())
            b_list.append(b.item())
            loss_list.append(MSE.item())

            # 梯度重製
            w.grad.zero_()
            b.grad.zero_()

    return w_list, b_list, loss_list

# 產生隨機線性資料
n = 100
X = np.linspace(0, 50, n)
y = np.linspace(0, 50, n)

# 加入一點雜訊
X += np.random.uniform(-10, 10, n)
y += np.random.uniform(-10, 10, n)

# 進行訓練
w_list, b_list, loss_list = train(torch.tensor(X), torch.tensor(y), epochs = 10000)
# 取得w、b的最佳解
print(f"w={w_list[-1]} b={b_list[-1]}")

# numpy驗證
coef = np.polyfit(X, y, deg=1)
print(f"w = {coef[0]}, b = {coef[1]}")

# 訓練的點和回歸線畫圖
# plt.scatter(X, y)
# plt.plot(X, w_list[-1] * X + b_list[-1])
# plt.show()

# 損失函數繪圖
plt.plot(loss_list)
plt.show()