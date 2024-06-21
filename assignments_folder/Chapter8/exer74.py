"""
言語処理100本ノック 第8章課題

74 正解率の計測
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ．

参考にしたサイト
https://qiita.com/yulily/items/a98ef90fe6cbc7f268c0

"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------------------
#   クラス，関数定義
# -----------------------------------------

# ニュースデータセットクラス
class NewsDataset(Dataset):
    def __init__(self, x, y, phase="train"):
        self.x = x
        self.y = y
        self.phase = phase

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# 単純なニューラルネットワーク
class SLNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # 全結合層を定義

    def forward(self, x):
        return self.fc(x)  # ロジットを出力


def calc_acc(net, dataloader):
    """
        Calculate_Accuracy
        正解率(accuracy)を計算するメソッド
        accuracy の説明 https://atmarkit.itmedia.co.jp/ait/articles/2209/15/news040.html
    """
    net.eval()  # モデルを評価モードにする
    correct_num = 0
    with torch.no_grad():   # 勾配計算をオフにする
        for feature_vals, labels in dataloader:       # データローダーで，特徴量Xと正解のラベルyを取り出す
            outputs = net(feature_vals)     # 特徴量を学習済ニューラルネットワークモデルに掛けて各カテゴリーの出現率を出力
            _, preds = torch.max(outputs, 1) # 各カテゴリーの出現率のテンソルから，最大値を取るインデックスをカテゴリ名として出力 torch.max(outputs, 1)の第2引数は次元を示す
            correct_num += torch.sum(preds == labels.data)  # 正解数の集計
    return correct_num / len(dataloader.dataset)    # 正解率の計算


# -----------------------------------------
#   データセットのロードとモデル読み込み
# -----------------------------------------

# データのロードと型変換
x_train = torch.load("./assignments_folder/Chapter8/x_train.pt").float()
x_valid = torch.load("./assignments_folder/Chapter8/x_valid.pt").float()
x_test = torch.load("./assignments_folder/Chapter8/x_test.pt").float()

y_train = torch.load("./assignments_folder/Chapter8/y_train.pt").long()
y_valid = torch.load("./assignments_folder/Chapter8/y_valid.pt").long()
y_test = torch.load("./assignments_folder/Chapter8/y_test.pt").long()

# データセットとデータローダーの準備
train_dataset = NewsDataset(x_train, y_train, phase="train")
valid_dataset = NewsDataset(x_valid, y_valid, phase="val")
test_dataset = NewsDataset(x_test, y_test, phase="test")

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# モデルの読み込み
model = torch.jit.load("./assignments_folder/Chapter8/model_scripted.pth", map_location="cpu")


# -----------------------------------------
#   メインルーチン
# -----------------------------------------

train_acc = calc_acc(model, train_dataloader)
valid_acc = calc_acc(model, valid_dataloader)
test_acc = calc_acc(model, test_dataloader)

print("学習データの正解率(accuracy): {:.4f}".format(train_acc))
print("検証データの正解率(accuracy): {:.4f}".format(valid_acc))
print("テストデータの正解率(accuracy): {:.4f}".format(test_acc))




# -----------------------------------------
#   補足情報
# -----------------------------------------

"""
出力結果

学習データの正解率(accuracy): 0.8966
検証データの正解率(accuracy): 0.8883
テストデータの正解率(accuracy): 0.8876
"""

"""
リーダブルコードの実践内容

モジュールのインポート：

必要なライブラリを最初にまとめてインポートしています。これにより、コードの依存関係が一目でわかります。
クラス定義の分離：

NewsDatasetクラスとSLNetクラスを定義し、それぞれの役割を明確にしています。これにより、データセットの処理とモデルの定義が明確に分離され、コードの再利用性が向上します。
データのロードと型変換の明示：

torch.loadでデータをロードし、必要な型に変換しています。データの準備部分が明確に分かれており、データがどこから来て、どのように処理されるかがわかりやすいです。
データセットとデータローダーの準備の分離：

データセットとデータローダーの準備を分けて記述しています。これにより、データのロードと前処理が一貫しており、後続の処理が簡単になります。
モデルのロードと評価関数の定義の分離：

モデルのロード部分と精度を計算する関数を分けて定義しています。これにより、各部分の役割が明確になり、コードの理解が容易になります。
関数の利用：

calc_accという関数を定義して、モデルの精度を計算する処理をまとめています。これにより、同じ処理を繰り返す必要がなくなり、コードが簡潔になります。
コメントの追加：

各セクションにコメントを追加して、コードの目的や動作を説明しています。これにより、コードの意図が明確になり、他の人が理解しやすくなります。
変数名の工夫：

変数名や関数名をわかりやすく命名しています。例えば、train_dataloaderやcalc_accなど、何をするための変数や関数かが名前から推測できます。
テンプレート化された構造：

一般的なPyTorchのプログラム構造（データセットの定義、データローダーの作成、モデルの定義、トレーニング・評価関数の実装）に従っています。これにより、PyTorchに慣れたユーザーが理解しやすくなっています。
余分な処理の排除：

必要な処理だけを記述し、余分な処理を含めないようにしています。これにより、コードがシンプルで読みやすくなります。
"""