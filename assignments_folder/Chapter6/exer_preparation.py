"""
    ファイル内容説明

    input_file = "./assignments_folder/Chapter6/news+aggregator/newsCorpora.csv"    より，
    情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
    
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

# 指定された情報源
specified_publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]

# 元のファイルと新しいファイルのパス
input_file = "./assignments_folder/Chapter6/news+aggregator/newsCorpora.csv"
output_file = "./assignments_folder/Chapter6/specified_publishers_articles.csv"

# 指定された情報源の記事を抽出して新しいファイルに書き込む
with open(input_file, 'r', encoding='utf-8') as csv_file, open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
    reader = csv.reader(csv_file, delimiter='\t')
    writer = csv.writer(output_csv, delimiter='\t')

    for row in reader:
        publisher = row[3]  # 出版社名の列
        if publisher in specified_publishers:
            writer.writerow(row)

print("指定された情報源の記事を抽出し、新しいファイルに書き込みました。")
