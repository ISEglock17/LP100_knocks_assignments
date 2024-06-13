import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ニュースデータセットクラス
class NewsDataset(Dataset):
    def __init__(self, x, y, phase="train"):
        self.x = x
        self.y = y
        self.phase = phase

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 単純なニューラルネットワーク
class SLNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # 全結合層を定義

    def forward(self, x):
        return self.fc(x)  # ロジットを出力

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

dataloaders_dict = {"train": train_dataloader, "val": valid_dataloader, "test": test_dataloader}

# モデル、損失関数、最適化手法の定義
model = SLNet(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学習用関数の定義
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("--------------------------------------------")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 訓練モード
            else:
                model.eval()  # 検証モード

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()  # 勾配の初期化

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)  # 損失の計算
                    _, preds = torch.max(outputs, 1)  # 予測ラベルの取得

                    if phase == "train":
                        loss.backward()  # 勾配の計算
                        optimizer.step()  # 重みの更新

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss /= len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

# 学習の実行
num_epochs = 10
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs)
