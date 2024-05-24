"""
言語処理100本ノック 第6章課題

問57 特徴量の重みの確認
52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．
"""

# https://qiita.com/tsugar/items/0391c9a45842f9d9ae69　
# に従い，TF-IDFの指標を抽出量として調べる。

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# データの読み込み
train_df = pd.read_csv('./assignments_folder/Chapter6/train.txt', delimiter='\t', header=None, names=['category', 'headline'])
valid_df = pd.read_csv('./assignments_folder/Chapter6/valid.txt', delimiter='\t', header=None, names=['category', 'headline'])
test_df = pd.read_csv('./assignments_folder/Chapter6/test.txt', delimiter='\t', header=None, names=['category', 'headline'])


# TF-IDFベクトルの計算
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# 学習データでTF-IDFをフィットし、各データセットに変換を適用
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['headline'])
X_valid_tfidf = tfidf_vectorizer.transform(valid_df['headline'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['headline'])

# 各単語のTF-IDF値を確認する
feature_names = tfidf_vectorizer.get_feature_names_out()
df_tfidf_train = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
df_tfidf_valid = pd.DataFrame(X_valid_tfidf.toarray(), columns=feature_names)
df_tfidf_test = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)

"""
# TF-IDFの結果を表示（先頭の5行を表示）
print("Training set TF-IDF sample:")
print(df_tfidf_train.head())
print("\nValidation set TF-IDF sample:")
print(df_tfidf_valid.head())
print("\nTest set TF-IDF sample:")
print(df_tfidf_test.head())

# TF-IDFの結果をファイルに保存
df_tfidf_train.to_csv('./assignments_folder/Chapter6/train.feature.txt', sep='\t', index=False)
df_tfidf_valid.to_csv('./assignments_folder/Chapter6/valid.feature.txt', sep='\t', index=False)
df_tfidf_test.to_csv('./assignments_folder/Chapter6/test.feature.txt', sep='\t', index=False)
"""

# 課題52
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Y_train = train_df['category']
Y_valid = valid_df['category']
Y_test = test_df['category']

# ロジスティック回帰モデルの学習
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, Y_train)


# 課題53
# print(f"カテゴリ順: {logistic_model.classes_}\n")
 
Y_pred = logistic_model.predict(X_valid_tfidf)
# print(f"各記事のカテゴリ(ラベル): {Y_valid.values}")
# print(f"各記事のカテゴリ予測: {Y_pred}\n")
 
Y_pred = logistic_model.predict_proba(X_valid_tfidf)
# print(f"カテゴリの予測確率: \n{Y_pred}")


# 課題54
from sklearn.metrics import accuracy_score
 
Y_pred_train = logistic_model.predict(X_train_tfidf)
Y_pred_test = logistic_model.predict(X_test_tfidf)
 
# print(f"trainの精度: {accuracy_score(Y_train, Y_pred_train)}")
# print(f"testの精度: {accuracy_score(Y_test, Y_pred_test)}")


# 課題55
from sklearn.metrics import confusion_matrix
 
# print(f"学習データの混同行列: \n{confusion_matrix(Y_train, Y_pred_train)}\n")
# print(f"評価データの混同行列: \n{confusion_matrix(Y_test, Y_pred_test)}")


# 課題56
from sklearn.metrics import precision_score, recall_score, f1_score
 
def metrics(y_data, y_pred, ave=None):
  precision_sco = precision_score(y_data, y_pred, average=ave)
  recall_sco = recall_score(y_data, y_pred, average=ave)
  f1_sco = f1_score(y_data, y_pred, average=ave)
  form = "適合率: {}\n再現率: {}\nF1: {}\n".format(precision_sco, recall_sco, f1_sco)
  return form
 
# print(f"カテゴリ順:{logistic_model.classes_}\n\n{metrics(Y_test, Y_pred_test)}")
# print("マクロ平均:\n", metrics(Y_test, Y_pred_test, "macro"))
# print("マイクロ平均:\n", metrics(Y_test, Y_pred_test, "micro"))


# 課題57
import numpy as np

features = np.array(tfidf_vectorizer.get_feature_names_out())
for c, coef in zip(logistic_model.classes_, logistic_model.coef_):
    top_10 = pd.DataFrame(features[np.argsort(-coef)[:10]], columns=[f"Top 10 Features with High Weight (Class: {c})"], index=[i for i in range(1, 11)])
    worst_10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=[f"Top 10 Features with Low Weight (Class: {c})"], index=[i for i in range(1, 11)])
    print(top_10, "\n")
    print(worst_10, "\n", "-" * 70)


# 出力結果
"""
   Top 10 Features with High Weight (Class: エンターテインメント)
1                                          kardashian
2                                               chris
3                                                 kim
4                                               miley
5                                               cyrus
6                                                star
7                                               movie
8                                                film
9                                                 jay
10                                            thrones   

   Top 10 Features with Low Weight (Class: エンターテインメント)
1                                              update
2                                              google
3                                               china
4                                                says
5                                            facebook
6                                               study
7                                                  gm
8                                               apple
9                                                 ceo
10                                            billion
 ----------------------------------------------------------------------
   Top 10 Features with High Weight (Class: ビジネス)
1                                             fed
2                                           china
3                                          stocks
4                                            bank
5                                             ecb
6                                          update
7                                            euro
8                                         ukraine
9                                             oil
10                                         profit

   Top 10 Features with Low Weight (Class: ビジネス)
1                                          ebola
2                                         google
3                                          video
4                                     kardashian
5                                      microsoft
6                                           star
7                                       facebook
8                                          study
9                                          virus
10                                         apple
 ----------------------------------------------------------------------
   Top 10 Features with High Weight (Class: 健康)
1                                         ebola
2                                        cancer
3                                         study
4                                           fda
5                                          drug
6                                          mers
7                                         cases
8                                           cdc
9                                         virus
10                                       health

   Top 10 Features with Low Weight (Class: 健康)
1                                           gm
2                                     facebook
3                                      twitter
4                                        apple
5                                       google
6                                      climate
7                                          ceo
8                                        china
9                                         deal
10                                      profit
 ----------------------------------------------------------------------
   Top 10 Features with High Weight (Class: 科学技術)
1                                          google
2                                        facebook
3                                           apple
4                                       microsoft
5                                         climate
6                                              gm
7                                           tesla
8                                            nasa
9                                             fcc
10                                     heartbleed

   Top 10 Features with Low Weight (Class: 科学技術)
1                                         stocks
2                                            fed
3                                           drug
4                                         shares
5                                       american
6                                            ecb
7                                     kardashian
8                                         cancer
9                                        ukraine
10                                         ebola
 ----------------------------------------------------------------------
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