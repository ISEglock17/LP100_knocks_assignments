"""
言語処理100本ノック 第6章課題

問58 正則化パラメータの変更
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# データの読み込み
train_df = pd.read_csv('./assignments_folder/Chapter6/train.txt', delimiter='\t', header=None, names=['category', 'headline'])
valid_df = pd.read_csv('./assignments_folder/Chapter6/valid.txt', delimiter='\t', header=None, names=['category', 'headline'])
test_df = pd.read_csv('./assignments_folder/Chapter6/test.txt', delimiter='\t', header=None, names=['category', 'headline'])

# TF-IDFベクトルの計算
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['headline'])
X_valid_tfidf = tfidf_vectorizer.transform(valid_df['headline'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['headline'])

# 正則化パラメータの範囲
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

# 正則化パラメータごとに正解率を格納するリスト
train_accuracies = []
valid_accuracies = []
test_accuracies = []

# 正則化パラメータごとにモデルを学習し、正解率を求める
for C in C_values:
    # ロジスティック回帰モデルの学習
    logistic_model = LogisticRegression(C=C, max_iter=1000)
    logistic_model.fit(X_train_tfidf, train_df['category'])
    
    # 学習データでの正解率
    train_predictions = logistic_model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(train_df['category'], train_predictions)
    train_accuracies.append(train_accuracy)
    
    # 検証データでの正解率
    valid_predictions = logistic_model.predict(X_valid_tfidf)
    valid_accuracy = accuracy_score(valid_df['category'], valid_predictions)
    valid_accuracies.append(valid_accuracy)
    
    # テストデータでの正解率
    test_predictions = logistic_model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(test_df['category'], test_predictions)
    test_accuracies.append(test_accuracy)
        
    print(f"【正則化パラメーター: {C}】\n")
    print(f"train精度: {train_accuracies[-1]}")
    print(f"valid精度: {valid_accuracies[-1]}")
    print(f"test精度: {test_accuracies[-1]}\n")

# グラフのプロット
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(C_values, valid_accuracies, marker='s', label='Validation Accuracy')
plt.plot(C_values, test_accuracies, marker='^', label='Test Accuracy')
plt.xscale('log')  # 正則化パラメータを対数スケールで表示
plt.xlabel('Regularization Parameter(C)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Regularization Parameter')
plt.legend()
plt.grid(True)
plt.show()


# 出力結果
"""
【正則化パラメーター: 0.001】

train精度: 0.43744728703964014
valid精度: 0.43670411985018726
test精度: 0.46741573033707867

【正則化パラメーター: 0.01】

train精度: 0.7888670227719988
valid精度: 0.7752808988764045
test精度: 0.8

【正則化パラメーター: 0.1】

train精度: 0.7979570799362758
valid精度: 0.7820224719101123
test精度: 0.8104868913857678

【正則化パラメーター: 1】

train精度: 0.9510823727860557
valid精度: 0.898876404494382
test精度: 0.9093632958801499

【正則化パラメーター: 10】

train精度: 0.9983131852684847
valid精度: 0.9295880149812734
test精度: 0.9265917602996254

【正則化パラメーター: 100】

train精度: 0.9994377284228282
valid精度: 0.9280898876404494
test精度: 0.9265917602996254
"""


"""

―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。

適切な変数名とコメント: 変数名やコメントが明確であり、コードの理解を助けています。例えば、textやsentenceといった変数名は適切であり、コメントも関数の役割を説明しています。
可読性のためのコードの分割: プログラム全体が1つの長い関数ではなく、論理的なブロックに分割されています。これにより、各部分の機能を理解しやすくなります。
条件分岐の明確化: 条件分岐が必要な箇所で条件式が明確になっており、可読性が向上しています。例えば、if len(line) == 2といった条件はシンプルで理解しやすいです。
適切なデータ構造の使用: 形態素解析結果を格納するために辞書やリストといった適切なデータ構造が使用されています。これにより、データの扱いや操作が容易になります。
適切な関数の使用: ファイルの読み込みやデータの解析に適切な関数が使用されており、コードが効率的であり、可読性が高まっています。
一貫性のあるスタイル: インデントやスペースの使用が一貫しており、コードのスタイルが統一されています。

"""