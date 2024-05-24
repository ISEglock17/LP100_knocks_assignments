"""
言語処理100本ノック 第6章課題

50. データの入手・整形
News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
抽出された事例をランダムに並び替える．
抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．




input_file = "./assignments_folder/Chapter6/news+aggregator/newsCorpora.csv"    より，
情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．

抽出したファイルは，
output_file = "./assignments_folder/Chapter6/specified_publishers_articles.csv"
に出力する。

input_file = "./assignments_folder/Chapter6/specified_publishers_articles.csv"より，    
抽出された事例をランダムに並び替える．
抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．


FORMAT: ID ︙TITLE ︙URL ︙PUBLISHER ︙CATEGORY ︙STORY ︙HOSTNAME ︙TIMESTAMP

＊詳細内容＊
ID 数値ID
TITLE ニュースのタイトル 
URL URL
PUBLISHER 出版社名
CATEGORY ニュースのカテゴリー（b = ビジネス、t = 科学技術、e = エンターテインメント、m = 健康）
STORY 同じストーリーに関するニュースを含むクラスターの英数字ID
HOSTNAME URLホスト名
TIMESTAMP 1970年1月1日00:00:00 GMTからのミリ秒数で表した、ニュースが発表されたおおよその時間。
"""

import csv
import random

# 指定された情報源
specified_publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]

# 元のファイルと新しいファイルのパス
input_file = "./assignments_folder/Chapter6/news+aggregator/newsCorpora.csv"
specified_publishers_output_file = "./assignments_folder/Chapter6/specified_publishers_articles.csv"
train_file = "./assignments_folder/Chapter6/train.txt"
valid_file = "./assignments_folder/Chapter6/valid.txt"
test_file = "./assignments_folder/Chapter6/test.txt"

# 指定された情報源の記事を抽出して新しいファイルに書き込む
with open(input_file, 'r', encoding='utf-8') as csv_file, open(specified_publishers_output_file, 'w', newline='', encoding='utf-8') as output_csv:
    reader = csv.reader(csv_file, delimiter='\t')
    writer = csv.writer(output_csv, delimiter='\t')

    for row in reader:
        publisher = row[3]  # 出版社名の列
        if publisher in specified_publishers:
            writer.writerow(row)

print("指定された情報源の記事を抽出し、新しいファイルに書き込みました。")



# ファイルを読み込み、ランダムに並び替える
with open(specified_publishers_output_file, 'r', encoding='utf-8') as file:
    instances = file.readlines()

random.shuffle(instances)

# データの数を取得
total_instances = len(instances)
train_size = int(0.8 * total_instances)
valid_size = int(0.1 * total_instances)

# データを学習データ、検証データ、評価データに分割
train_data = instances[:train_size]
valid_data = instances[train_size:train_size+valid_size]
test_data = instances[train_size+valid_size:]



# カテゴリ名の変換辞書
category_dict = {
    "b": "ビジネス",
    "t": "科学技術",
    "e": "エンターテインメント",
    "m": "健康"
}

def convert_category(category):
    """
        カテゴリ名を変換する関数
    """
    return category_dict.get(category, "Unknown")

def write_data(filename, data):
    """
        ファイルにデータを書き込む
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for instance in data:
            article_data_list = instance.strip().split('\t')  # CATEGORYとTITLEのみを取得
            title = article_data_list[1]
            category = convert_category(article_data_list[4])  # カテゴリ名を変換
            file.write(f"{category}\t{title}\n")  # タブ区切り形式で書き込み

# 学習データ、検証データ、評価データの書き込み
write_data(train_file, train_data)
write_data(valid_file, valid_data)
write_data(test_file, test_data)

print("データを学習データ、検証データ、評価データに分割し、ファイルに保存しました。")



def count_category_instances(data: str) -> dict:
    """
        各カテゴリにおける事例数をカウントする関数
        data: 事例
        -> category_counts: 各カテゴリに対するカウントを格納した辞書型
    """
    category_counts = {"ビジネス": 0, "科学技術": 0, "エンターテインメント": 0, "健康": 0}
    for instance in data:
        category = convert_category(instance.strip().split('\t')[4])  # カテゴリ名を取得し、変換する
        category_counts[category] += 1
    return category_counts

def print_category_counts(data_name: str, category_counts: int):
    """
        カテゴリごとの事例数を表示する関数
        data_name: カテゴリ名
        category_counts: カテゴリのカウント
    """
    print(f"{data_name}のカテゴリごとの事例数:")
    for category, count in category_counts.items():
        print(f"{category}: {count}件")
    print()

# カテゴリごとの事例数をカウント
train_category_counts = count_category_instances(train_data)
valid_category_counts = count_category_instances(valid_data)
test_category_counts = count_category_instances(test_data)

# 各データセットのカテゴリごとの事例数を表示
print_category_counts("学習データ", train_category_counts)
print_category_counts("検証データ", valid_category_counts)
print_category_counts("評価データ", test_category_counts)


# 出力結果
"""
# train.txt
エンターテインメント	Save the Met #weareopera
健康	Study Suggests Health Insurance Saves Lives. The Hill Wonders If That's A  ...
ビジネス	Teva Gets US High Court Hearing on Generic Copaxone Delay (4)
ビジネス	PRECIOUS-Gold struggles below $1300, hovers near six-week low
エンターテインメント	The Fault in the Film
...

# valid.txt
エンターテインメント	Scarlett Johansson Talks Balancing Motherhood And A Career
エンターテインメント	Easter Week for Stoics: Why I Love Jesus But I'm Kind of 'Meh' About Easter
エンターテインメント	Aus fans welcome Harry Potter update
エンターテインメント	Rose Byrne keeps comedy streak going
エンターテインメント	Nick Carter Gets Married, Retroactively Destroys Every '90s Girl's Dreams
...

# test.txt
エンターテインメント	First official look at new Superman
エンターテインメント	Avril Lavigne shows off her slim figure in plunging leather bodice in new  ...
エンターテインメント	"Time For The ""Happily Ever After"": After ""Bachelorette"" Finale, Andi And Josh  ..."
ビジネス	Spanish stocks - Factors to watch on Monday
エンターテインメント	Lana Wachowski - Jupiter Ascending pushed back to 2015
...

"""

"""
学習データのカテゴリごとの事例数:
ビジネス: 4467件
科学技術: 1226件
エンターテインメント: 4244件
健康: 743件

検証データのカテゴリごとの事例数:
ビジネス: 565件
科学技術: 162件
エンターテインメント: 523件
健康: 85件

評価データのカテゴリごとの事例数:
ビジネス: 595件
科学技術: 136件
エンターテインメント: 522件
健康: 82件
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