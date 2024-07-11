import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from transformers import BertTokenizer, BertModel  # Trainer, TrainingArgumentsは不要なので削除
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from matplotlib import pyplot as plt

# -----------------------------------------
#   データ準備
# -----------------------------------------
# newsCorporaから記事をDataFrame形式で読み取る
file = "./assignments_folder/Chapter6/news+aggregator/newsCorpora.csv"
data = pd.read_csv(file, encoding="utf-8", header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])

# 特定のpublisherのみを抽出する
publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
data = data.loc[data["PUBLISHER"].isin(publishers), ["TITLE", "CATEGORY"]].reset_index(drop=True)

# dataの前処理を行う
for i in range(len(data["TITLE"])):
    text = data["TITLE"][i]
    text_clean = re.sub(r"[\"\".,:;\(\)#\|\*\+\!\?#$%&/\]\[\{\}]", "", text)
    text_clean = re.sub("[0-9]+", "0", text_clean)
    text_clean = re.sub("\s-\s", " ", text_clean)
    data.at[i, "TITLE"] = text_clean

# 学習用(Train)，検証用(Valid)，評価用(Test)に分割する
train, valid_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=15, stratify=data["CATEGORY"])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=15, stratify=valid_test["CATEGORY"])

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

# カテゴリを数値にマッピング
category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.from_numpy(train['CATEGORY'].map(category_dict).values)
y_valid = torch.from_numpy(valid['CATEGORY'].map(category_dict).values)
y_test = torch.from_numpy(test['CATEGORY'].map(category_dict).values)

# -----------------------------------------
#   BERT DataSet
# ----------------------------------------- 
class BERTDataSet(Dataset):
    """ 
    sentence: 入力テキスト
    add_special_tokens=True: [CLS] と [SEP] トークンを追加
    max_length=128: 最大長
    padding='max_length': 長さが足りない場合にパディングを追加
    truncation=True: 長さが超過する場合にカット
    return_attention_mask=True: 注意マスクを返す
    return_tensors='pt': PyTorchのテンソルとして返す
    
    input_ids: テキストのIDリスト
    attention_mask: テキストの各トークンが実際のトークンかパディングかを示すマスク
    labels: ラベルのテンソル
    """
    def __init__(self, X, y, tokenizer, max_length=128):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        sentence = self.X[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        label = torch.tensor(self.y[idx], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

# データセット作成
train_dataset = BERTDataSet(train, y_train, phase='train')
valid_dataset = BERTDataSet(valid, y_valid, phase='val')
test_dataset = BERTDataSet(test, y_test, phase='val')

# データローダーの作成
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {'train': train_dataloader, 'val': valid_dataloader, 'test': test_dataloader}


# -----------------------------------------
#   BERT 分類モデルの定義
# ----------------------------------------- 
class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate, hidden_size, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Sequential(nn.Linear(768, hidden_size),
                                nn.ReLU(),
                                nn.BatchNorm1d(hidden_size),
                                nn.Linear(hidden_size, output_size)
                                )

    def forward(self, ids, mask):
        output = self.bert(ids, attention_mask=mask)[-1]
        output = self.fc(self.drop(output))
        return output


# -----------------------------------------
#   学習用関数の定義
# ----------------------------------------- 
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    """
    # GPU確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())
    print("使用デバイス:", device)
    
    net.to(device)
    """
    
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # 訓練モード
            else:
                net.eval() # 検証モード
            
            epoch_loss = 0.0 # epochの損失和
            epoch_corrects = 0 # epochの正解数
            
            # データローダーからミニバッチを取り出すループ
            for data in dataloaders_dict[phase]:
                ids = data['ids']
                mask = data['mask']
                labels = data['labels']
                optimizer.zero_grad() # optimizerを初期化
                """
                ids = data['ids'].to(device)
                mask = data['mask'].to(device)
                labels = data['labels'].to(device)
                optimizer.zero_grad() # optimizerを初期化
                """
                
                # 順伝播計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(ids, mask)
                    loss = criterion(outputs, labels) # 損失を計算
                    _, preds = torch.max(outputs, 1) # ラベルを予想する
                    
                    # 訓練時は逆伝播にする
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # lossを更新
                    epoch_loss += loss.item() * ids.size(0)
                    # 正解数を更新
                    epoch_corrects += torch.sum(preds == labels.data)
            
            # epochごとのlossと正解率の表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.cpu())
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc.cpu())
            
        print('Epoch {} / {} (train) Loss: {:.4f}, Acc: {:.4f}, (val) Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, num_epochs, train_loss[-1], train_acc[-1], valid_loss[-1], valid_acc[-1]))
    return train_loss, train_acc, valid_loss, valid_acc


# パラメータの設定
DROP_RATE = 0.2
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5

# BERTモデルの定義
net = BERTClass(DROP_RATE, HIDDEN_SIZE, OUTPUT_SIZE)
net.train()

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
optimizer = torch.optim.AdamW(params=net.parameters(), lr=LEARNING_RATE)

train_loss, train_acc, valid_loss, valid_acc = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=NUM_EPOCHS)


# モデルの性能評価
# テストデータでの評価
net.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for data in test_dataloader:
        ids = data['input_ids']
        mask = data['attention_mask']
        labels = data['labels']

        outputs = net(ids, mask)
        _, preds = torch.max(outputs, 1)
        
        test_predictions.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# 精度と分類レポートの出力
print(f'Accuracy: {accuracy_score(test_labels, test_predictions)}')
print(classification_report(test_labels, test_predictions, target_names=list(category_dict.keys())))

plt.plot(train_loss, label='Train Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.plot(train_acc, label='Train Accuracy')
plt.plot(valid_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
