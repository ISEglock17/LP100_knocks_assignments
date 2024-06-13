"""
言語処理100本ノック 第8章課題

72

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

criterion = nn.CrossEntropyLoss()   # クロスエントロピー損失
logits = model(x_train[:4])
loss = criterion(logits, y_train[:4])   # logitsとy_train[0](正解ラベル)の間の損失

print("損失: ", loss.item())

model.zero_grad()
loss.backward()
print("勾配: ")
print(model.fc.weight.grad)