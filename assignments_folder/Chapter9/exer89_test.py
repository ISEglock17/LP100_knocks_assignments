import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

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
train['LABEL'] = train['CATEGORY'].map(category_dict)
valid['LABEL'] = valid['CATEGORY'].map(category_dict)
test['LABEL'] = test['CATEGORY'].map(category_dict)

# -----------------------------------------
#   BERT DataSet
# ----------------------------------------- 
class BERTDataSet(Dataset):
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

# トークナイザーのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# データセット作成
train_dataset = BERTDataSet(train['TITLE'].values, train['LABEL'].values, tokenizer)
valid_dataset = BERTDataSet(valid['TITLE'].values, valid['LABEL'].values, tokenizer)
test_dataset = BERTDataSet(test['TITLE'].values, test['LABEL'].values, tokenizer)

# データローダーの作成
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------
#   BERT モデル
# -----------------------------------------
# モデルの準備
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# トレーニング設定
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    save_steps=10,
    eval_steps=10,
    save_total_limit=1
)

# Trainerの設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(axis=1)),
        'report': classification_report(p.label_ids, p.predictions.argmax(axis=1), target_names=['Business', 'Entertainment', 'Politics', 'Sport'])
    }
)

# モデルのトレーニング
trainer.train()

# 検証データでの評価
eval_results = trainer.evaluate()
print(f"Validation Results: {eval_results}")

# テストデータでの評価
test_results = trainer.predict(test_dataset)
print(f"Test Results: {test_results.metrics}")
print(test_results.metrics['test_accuracy'])
print(test_results.metrics['test_report'])

# 混同行列を表示
import seaborn as sns
import matplotlib.pyplot as plt

# 混同行列の計算
from sklearn.metrics import confusion_matrix

y_pred = test_results.predictions.argmax(axis=1)
y_true = test['LABEL'].values

conf_matrix = confusion_matrix(y_true, y_pred)

# 混同行列の可視化
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Business', 'Entertainment', 'Politics', 'Sport'], yticklabels=['Business', 'Entertainment', 'Politics', 'Sport'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
