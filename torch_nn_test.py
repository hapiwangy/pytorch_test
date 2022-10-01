# full_connect_layer
import torch
# 產生隨機輸入之二維陣列
input = torch.randn(128, 20)

# 建立Linear神經層
## 參數依序為
## 輸入神經元, 輸出神經元, 是否產生bias, 裝置(None, cpu, GPU)
## linear的轉換為 y = x * A + b
## 這裡寫的都是矩陣乘法，並且忽略transpose的符號
layer1 = torch.nn.Linear(20, 30)

# 神經層計算
## 未訓練linear_layer是進行矩陣內積
output = layer1(input)

# 建立Bilinear神經層
## Bilinear的轉換為y=x1*A*x2 + b
## 這裡寫的都是矩陣乘法，並且忽略transpose的符號
layer2 = torch.nn.Bilinear(20, 30, 40)
## 因為一個*左邊，一個*右邊，所以維度不一樣
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
# 神經層計算
## 沒有訓練一樣進行矩陣內積
output = layer2(input1, input2)
