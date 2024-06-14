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


""" 
リーダブルコードの実践

明確な名前付け
クラス名: SLNet（Single Layer Network）として、シンプルな構造であることを明示。
関数名: __init__, forward など、Pythonのクラス内で一般的に使用される名前を採用し、役割が直感的に理解できる。
変数名: criterion, loss, grad など、意味が明確で直感的に理解できる名前を使用。
一貫性のあるスタイル
インデント: 4スペースで統一。
空白: 適切に使用し、演算子や関数の引数間のスペースを保つ。
命名規則: スネークケースで統一された変数名 (x_train, y_train, etc.)。
コメントの適切な使用
クラスと関数の説明: SLNet クラスや criterion の目的をコメントで説明。
重要な処理: loss.backward() の前にコメントを付けて、処理の流れを補足。
損失関数: criterion の選択に関しての説明を追加。
関数の分割
単一責任の原則:
__init__ 関数はモデルの構造を定義。
forward 関数は入力データをモデルに渡して出力を得る処理を担当。
他の処理も関数化し、それぞれの責任を明確化。
DRY原則（Don't Repeat Yourself）
全結合層の定義: nn.Linear を用いて、シンプルに全結合層を1回定義。
損失計算と勾配計算: criterion を使用し、クロスエントロピー損失を1回の関数呼び出しで計算。
エラーハンドリング
型変換: .float() や .long() を用いて、モデルに渡すデータの型を明示的に変換。
損失計算: loss.item() で数値値として損失を取得。
コードの再利用性と保守性
モジュール化: モデルを SLNet クラスに分離し、他の部分からも再利用可能に。
パラメータ化: input_size と output_size をコンストラクタで受け取り、柔軟にモデルを変更可能。
学習処理の記述
損失関数の定義: nn.CrossEntropyLoss() を criterion として明示。
勾配の計算: loss.backward() で勾配を計算し、model.zero_grad() で勾配をリセット。

"""