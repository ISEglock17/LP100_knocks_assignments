"""
言語処理100本ノック 第6章課題

51. 特徴量抽出
学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう
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
Training set TF-IDF sample:
    00   07   08   09  0ff  0ut   10  100  1000  10000  100000  100k  100th  101  103  106  ...  zip  zloty  zmapp  zoe  zombie  zombies  zone  zooey  zuckerberg  zynga   zâ  œlousyâ  œpiece  œwaist  エンターテインメント
  ビジネス
0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
3  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    

[5 rows x 12512 columns]

Validation set TF-IDF sample:
    00   07   08   09  0ff  0ut   10  100  1000  10000  100000  100k  100th  101  103  106  ...  zip  zloty  zmapp  zoe  zombie  zombies  zone  zooey  zuckerberg  zynga   zâ  œlousyâ  œpiece  œwaist  エンターテインメント
  ビジネス
0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
3  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    

[5 rows x 12512 columns]

Test set TF-IDF sample:
    00   07   08   09  0ff  0ut   10  100  1000  10000  100000  100k  100th  101  103  106  ...  zip  zloty  zmapp  zoe  zombie  zombies  zone  zooey  zuckerberg  zynga   zâ  œlousyâ  œpiece  œwaist  エンターテインメント
  ビジネス
0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
3  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    
4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0    0.0     0.0   0.0    0.0  0.0  0.0  0.0  ...  0.0    0.0    0.0  0.0     0.0      0.0   0.0    0.0         0.0    0.0  0.0      0.0     0.0     0.0         0.0   0.0    

[5 rows x 12512 columns]
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