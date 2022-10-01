import matplotlib.pyplot as plt
import numpy as np
import torch

# 訓練資料
n = 100
X = np.linspace(0, 50, n)
y = np.linspace(0, 50, n)
# 加入雜訊
X += np.random.uniform(-10, 10, n)
y += np.random.uniform(-10, 10, n)

# 定義模型函數
def create_model(input_feature, output_feature):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_feature, output_feature),
        torch.nn.Flatten(0, -1) # 所有維度轉成一維
    )
    return model

# 定義訓練函數
## 神經網路只有使用一層FC_layer，輸入/出僅有一個神經元(X/y)
## bias的預設維True，也就是求y=ax+b
def train(X, y, epochs=2000, lr=1e-8):
    ##這裡用mseloss來取代mse
    model = create_model(1,1)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    ## 進行正向/反向訓練
    loss_list, w_list, b_list = [], [], []
    for epoch in range(epochs):
        y_pred = model(X)
        # 計算loss_function value
        MSE = loss_fn(y_pred, y)
        # 梯度重製
        model.zero_grad()
        # 反向傳導
        MSE.backward()
        # 權重更新
        ## 損失函數用loss_fn(y_pred, y)
        ## 梯度重製用model.zero_grad()來代替w、b
        ## 這裡用model.parameters來代替w、b
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        linear_layer = model[0]
        if (epoch + 1) % 1000 == 0 or epoch < 1000:
            w_list.append(linear_layer.weight[:, 0].item()) # 轉成常數
            b_list.append(linear_layer.bias.item())
            loss_list.append(MSE.item())

    return w_list, b_list, loss_list

# 執行訓練
X2, y2 = torch.FloatTensor(X.reshape(X.shape[0], 1)), torch.FloatTensor(y)
w_list, b_list, loss_list = train(X2, y2, epochs = 10 ** 5, lr = 1e-4)
print(f"w={w_list}, b={b_list}")

coef = np.polyfit(X, y, deg=1)
print(f"w={coef[0]}, b={coef[0]}")
plt.plot(loss_list)
plt.show()