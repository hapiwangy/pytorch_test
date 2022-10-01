import torch
# tensor相關運算 

# 可以宣告不同型態的tensor
# tensor = tensor/(Int/Float/Long)Tensor(list_var)
# 四則運算:直接tensor +/-/*//(除)/@(內積) tensor
# 如果要顯示數值的話可以轉換為numpy陣列 tensor.numpy()
# 也可以把numpy陣列轉換成tensor。tensor = torch.from_numpy(np_array_var)
# 沿著某一個方向來做加總tenosr.sum(axis=多少)

# pytorch要自己把變數搬到cpu/gpu做運算，如果兩個變數不在同一個地方的話會出現錯誤
# 透過.to('cpu/cuda')或.cuda()/.cpu()來搬到cpu/gpu
## 雖然比較麻煩，但可以避免out of memory錯誤
# tensors = torch.tensor([[1,2,3]])
# print(tensors)

# tensor_gpu = tensors.cuda()
# print(tensor_gpu) # 給到gpu

# tensor_gpu_2 = tensors.to('cuda:1') #若有多個gpu可以給他們編號
# print(tensor_gpu_2)

# tensor_cpu = tensors.cpu() #給回CPU
# print(tensor_cpu)

# device = "cuda" if torch.cuda.is_available() else "cpu":
# tensor_gpu(device) # 先設定device=gpu/cpu，之後再對變數設定

# 轉換成稀疏矩陣
# a = torch.LongTensor([[0,1,1],[2,0,2]]) # 定義稀疏矩陣有值得行/列
# v = torch.FloatTensor([3,4,5]) # 定義稀疏矩陣的值
# 下面的Size來定義稀疏矩陣的大小
# print(torch.sparse.FloatTensor(a, v, torch.Size([2,3])).to_dense())
## 可以印出上面的結果來看稀疏矩陣之間的關係

# 稀疏矩陣的運算
# a = torch.sparse.FloatTensor(a, v, torch.Size([2,3])) + \
#     torch.sparse.FloatTensor(a, v, torch.Size([2,3]))
# a.to_dense()

# 自動微分
## 設定自動微分
# x = torch.tensor(4.0, requires_grad=True)
# y = x ** 2
# print(y)
# print(y.grad_fn) # 取得y的梯度函數
# y.backward()# 進行反向傳導(對y進行偏微分)
# print(x.grad)
## 上面的執行結果是 x^2 對x進行微分，(2*x)->(x=4帶入) = tensor(8.)(這個就是梯度)

# 自動顯示相關屬性
# x = torch.tensor(1.0, requires_grad=True)
# y = torch.tensor(2.0)
# z = x * y
# for i, name in zip([x,y,z], "xyz"):
#     print(f"{name}\ndata: {i.data}\nrequire_grad: {i.requires_grad}\ngrad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf {i.is_leaf}\n")

# example : 計算cross entropy
# x = torch.ones(5)
# y = torch.zeros(3)
# w = torch.randn(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)
# z = torch.matmul(x, w) + b
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# print(f"z的梯度函數:{z.grad_fn}")
# print(f"loss的梯度函數:{loss.grad_fn}")
# 因為z是w、b函數，loss是z的函數，所以對loss進行反向傳導就好了
# loss.backward()
# print(w.grad)
# print(b.grad)

## 之前會使用Variable和tensor混用，現在就都用tensor就好了

# 進行過一次的.backward的時候無法再進行下一次(運算圖會被銷毀)，除非先用retain_graph=True保留運算圖
# 梯度會不斷地進行累加，所以執行y.backward後要重製梯度，使用x.grad.zero_()

## 重製/不重製梯度的範例
# x = torch.tensor(5.0, requires_grad=True)
# y = x ** 3
# # 把x.grad.zero_()拿掉的話就是沒有重製梯度
# y.backward(retain_graph=True) # 梯度下降
# print(f"第一下降{x.grad}")
# x.grad.zero_()
# y.backward(retain_graph=True) # 梯度下降
# print(f"第二下降{x.grad}")
# x.grad.zero_()
# y.backward(retain_graph=True) # 梯度下降
# print(f"第三下降{x.grad}")

## 多變數的梯度下降
# x = torch.tensor(5.0, requires_grad=True)
# y = x ** 2
# z = y ** 3
# z.backward()
# print(x.grad)