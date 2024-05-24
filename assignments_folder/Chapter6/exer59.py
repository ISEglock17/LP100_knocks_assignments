"""
言語処理100本ノック 第6章課題

問59 ハイパーパラメータの探索
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
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

"""
# 課題57
import numpy as np

features = np.array(tfidf_vectorizer.get_feature_names_out())
for c, coef in zip(logistic_model.classes_, logistic_model.coef_):
    top_10 = pd.DataFrame(features[np.argsort(-coef)[:10]], columns=[f"Top 10 Features with High Weight (Class: {c})"], index=[i for i in range(1, 11)])
    worst_10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=[f"Top 10 Features with Low Weight (Class: {c})"], index=[i for i in range(1, 11)])
    print(top_10, "\n")
    print(worst_10, "\n", "-" * 70)
"""

# 課題59
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 学習アルゴリズムとパラメーターの組み合わせを定義
models_params = [
    (LogisticRegression(), {'C': [0.001, 0.01, 0.1, 1, 10]}),
    (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 50]}),
    (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
]

best_accuracy = 0
best_model = None
best_params = None

# 各組み合わせで検証データ上の正解率を計算し、最良のモデルとパラメーターを見つける
for model, params in models_params:
    grid_search = GridSearchCV(model, params, cv=3)
    grid_search.fit(X_train_tfidf, train_df['category'])
    valid_predictions = grid_search.predict(X_valid_tfidf)
    valid_accuracy = accuracy_score(valid_df['category'], valid_predictions)
    print(f"検証データ正解率（{model.__class__.__name__}）: {valid_accuracy}")

    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_model = model.__class__.__name__
        best_params = grid_search.best_params_


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# 最良のモデルとパラメーターを用いて評価データ上の正解率を計算
if best_model == 'LogisticRegression':
    best_model_instance = LogisticRegression(**best_params)
elif best_model == 'RandomForestClassifier':
    best_model_instance = RandomForestClassifier(**best_params)
elif best_model == 'SVC':
    best_model_instance = SVC(**best_params)

best_model_instance.fit(X_train_tfidf, train_df['category'])
test_predictions = best_model_instance.predict(X_test_tfidf)
test_accuracy = accuracy_score(test_df['category'], test_predictions)

print(f"\n最良のモデルとパラメーター: {best_model}, {best_params}")
print(f"評価データ正解率: {test_accuracy}")



# 出力結果
"""
検証データ正解率（LogisticRegression）: 0.9295880149812734
検証データ正解率（RandomForestClassifier）: 0.8681647940074907
検証データ正解率（SVC）: 0.9220973782771535


最良のモデルとパラメーター: LogisticRegression, {'C': 10}
評価データ正解率: 0.9265917602996254
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