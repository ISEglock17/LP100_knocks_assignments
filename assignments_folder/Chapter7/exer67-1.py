"""
言語処理100本ノック 第7章課題

67. k-meansクラスタリング
国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

国名に関する単語ベクトル生成をexer67-1.py，k-meansクラスタリングをexer67-2.pyで行う。

"""
import re
from tqdm import tqdm   # 進捗率表示用
import csv
import gensim

# 入力ファイルと出力ファイルのパス
inputfile = './assignments_folder/Chapter7/questions-words_similarity.txt'
outputfile = './assignments_folder/Chapter7/country_word2vec.csv'
model_file = './assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz'

# Word2Vecモデルのロード
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

# 変数定義
categorie_list1 = ['capital-common-countries', 'capital-world']
categorie_list2 = ['currency', 'gram6-nationality-adjective']
country_list = set()

# 国名抽出
with open(inputfile, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        if line[0] == ':':  # 行がカテゴリ名を示している場合
            category = re.sub(r'[ :\n]', '', line)
        else:
            word_list = line.strip().split('\t')[0].split()
            if category in categorie_list1:
                country_list.add(word_list[1])
            elif category in categorie_list2:
                country_list.add(word_list[0])
            else:
                continue

# 進捗率計算用処理
total = len(country_list)

# CSVファイルへの書き込み
with open(outputfile, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['Country', 'Word2Vec'])  # ヘッダーの書き込み
    for country in tqdm(country_list, total=total):
        country_vec = model[country]
        country_vec_str = ' '.join(map(str, country_vec))  # ベクトルを文字列に変換
        writer.writerow([country, country_vec_str])
        
# 出力
print(len(country_list))
print(country_list)


# 出力結果
"""
129
{'Oman', 'Hungary', 'Belgium', 'Gabon', 'Mauritania', 'Greece', 'Belize', 'Slovenia', 'Iraq', 'Greenland', 'Portugal', 'Turkmenistan', 'Cambodia', 'India', 'Tunisia', 'Austria', 'Niger', 'Montenegro', 'Netherlands', 'Samoa', 'Bulgaria', 'Germany', 'Suriname', 'Kyrgyzstan', 'Mali', 'Kenya', 'Rwanda', 'Liberia', 'Morocco', 'Laos', 'Fiji', 'Tajikistan', 'France', 'Eritrea', 'Malaysia', 'Estonia', 'Denmark', 'Bahamas', 'Slovakia', 'Ukraine', 'Ireland', 'Philippines', 'Europe', 'Belarus', 'Lebanon', 'Uganda', 'Israel', 'Burundi', 'Algeria', 'Gambia', 'Qatar', 'Macedonia', 'Dominica', 'Norway', 'Tuvalu', 'Italy', 'Moldova', 'Lithuania', 'Venezuela', 'Albania', 'Switzerland', 'Uruguay', 'Nigeria', 'Japan', 'Pakistan', 'Jamaica', 'Nicaragua', 'Sudan', 'Turkey', 
'Uzbekistan', 'Mexico', 'Croatia', 'Senegal', 'Nepal', 'Cuba', 'Afghanistan', 'Mozambique', 'Peru', 'Bangladesh', 'Liechtenstein', 'Zimbabwe', 'Cyprus', 'Vietnam', 'Spain', 'Colombia', 'Australia', 'Somalia', 'Russia', 'Malawi', 'Sweden', 'Guinea', 'Jordan', 'Angola', 'Romania', 'Georgia', 'Botswana', 'USA', 'Bhutan', 'China', 'Azerbaijan', 'Poland', 'Indonesia', 'Serbia', 'Iran', 'Honduras', 'Syria', 'Zambia', 'Argentina', 'Brazil', 'Armenia', 'Madagascar', 'Guyana', 'Malta', 'Canada', 'Libya', 'Taiwan', 'Korea', 'Namibia', 'Chile', 'Kazakhstan', 'Latvia', 'Bahrain', 'Iceland', 'Ghana', 'Finland', 'Egypt', 'Ecuador', 'Thailand', 'England'}
"""

"""
― 参考になるサイト
https://qiita.com/kuroitu/items/f18acf87269f4267e8c1
上記のサイトで記述されているtqdmを用いることで，進捗率をプログレスバーとして表示することができる。

リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く
コメントの追加: コードの目的や各行の動作を説明するコメントが追加されています。
適切な変数名: 変数名は意味がありわかりやすいものになっています。たとえば、categorie_list1、categorie_list2、country_listなどが挙げられます。
冗長なコメントの削除: 自己説明的なコードであるため、冗長なコメントは削除されています。
適切なファイルパス: ファイルの場所を指定する際に、適切な相対パスが使用されています。
進捗率の表示: 処理の進捗率をプログレスバーとして表示するためにtqdmが使用されています。

"""