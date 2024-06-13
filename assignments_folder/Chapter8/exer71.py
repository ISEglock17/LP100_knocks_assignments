"""
言語処理100本ノック 第8章課題

71

単層ニューラルネットワークの組み方はこちら参照
https://qiita.com/y_fujisawa/items/94d3d2b0e5362510e042
https://free.kikagaku.ai/tutorial/basic_of_deep_learning/learn/pytorch_beginners

"""
import torch
from torch import nn

class SLNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)    # 300次元ベクトルから4つのカテゴリーのベクトルへと線形変換する 全結合層を定義
    
    def forward(self, x):
        logits = self.fc(x)     # ソフトマックス関数を適用する前のスコアを出力
        return logits
    
# データのロードと型変換
x_train = torch.load("./assignments_folder/Chapter8/x_train.pt").float()
x_valid = torch.load("./assignments_folder/Chapter8/x_valid.pt").float()
x_test = torch.load("./assignments_folder/Chapter8/x_test.pt").float()

y_train = torch.load("./assignments_folder/Chapter8/y_train.pt").long()
y_valid = torch.load("./assignments_folder/Chapter8/y_valid.pt").long()
y_test = torch.load("./assignments_folder/Chapter8/y_test.pt").long()
    
model = SLNet(300, 4)
print(model)

logits = model(x_train[:4])
y_hat = nn.Softmax(dim=1)(logits)
print(logits)
print(y_hat)

"""
＊出力結果＊
tensor([[ 0.1098,  0.0762,  0.0714, -0.1422],
        [ 0.0629,  0.1139,  0.0164, -0.1026],
        [ 0.0455,  0.0924, -0.0487, -0.0802],
        [ 0.0888, -0.0953, -0.0856, -0.0860]], grad_fn=<AddmmBackward0>)
tensor([[0.2698, 0.2609, 0.2596, 0.2097],                       # 最初の入力が各カテゴリに属する確率を示す。
        [0.2595, 0.2730, 0.2477, 0.2199],
        [0.2604, 0.2729, 0.2370, 0.2296],
        [0.2848, 0.2369, 0.2392, 0.2391]], grad_fn=<SoftmaxBackward0>)
        
"""