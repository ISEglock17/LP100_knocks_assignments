"""
言語処理100本ノック 第2章課題

17. １列目の文字列の異なりPermalink
1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはcut, sort, uniqコマンドを用いよ．

"""
import subprocess
import pandas as pd

# ファイルパス指定
filepath = "./assignments_folder/Chapter2/popular-names.txt"    # ファイルパスを指定
df = pd.read_table(filepath, header=None)                     # pandasにおけるDataFrame形式に変更する

unique_names = set(df.iloc[:, 0])      # 1列目を抽出する

# 出力
print(unique_names)       

"""
＊ 出力結果 ＊
{'Noah', 'Bessie', 'Helen', 'Madison', 'Judith', 'Stephanie', 'Jennifer', 'Oliver', 'Ethel', 'Robert', 'Larry', 'Kathleen', 'Julie', 'Lauren', 'Jayden', 'Ronald', 'Donald', 'Ida', 'Brian', 'Crystal', 'Rebecca', 'George', 'Deborah', 'Harry', 'Debra', 'Doris', 'Laura', 'Kelly', 'Pamela', 'Mildred', 'Nicholas', 'Frances', 'Charles', 'Carolyn', 'Austin', 'Anthony', 'Jessica', 'Mason', 'Sarah', 'Mary', 'Lucas', 'Clara', 'Amelia', 'Evelyn', 'Dorothy', 'Angela', 'Heather', 'Liam', 'Logan', 'Cynthia', 'Tracy', 'William', 'Frank', 'Betty', 'Steven', 'Virginia', 'Margaret', 'Minnie', 'Lisa', 'Jacob', 'Patricia', 'Lori', 
'Alexis', 'Alexander', 'Brandon', 'Ethan', 'Isabella', 'Elizabeth', 'Aiden', 'Benjamin', 'David', 'Bertha', 'Justin', 'Melissa', 'Richard', 'Shirley', 'Thomas', 'Brittany', 'Barbara', 'Ashley', 'Tyler', 'Tammy', 'Susan', 'Gary', 'Andrew', 'Sharon', 'Charlotte', 'Taylor', 'John', 'Linda', 'Amy', 'Sandra', 'Alice', 'Ruth', 'Mia', 'Matthew', 'Lillian', 'Ava', 'Olivia', 'Joseph', 'Michael', 'Jason', 'Karen', 'Daniel', 'Amanda', 'Donna', 'Sophia', 'Megan', 'Emma', 'Anna', 'Christopher', 'Samantha', 'James', 'Walter', 'Carol', 'Joan', 'Joshua', 'Annie', 'Henry', 'Michelle', 'Hannah', 'Scott', 'Elijah', 'Abigail', 'Nancy', 'Kimberly', 'Florence', 'Mark', 'Harper', 'Edward', 'Emily', 'Chloe', 'Rachel', 'Jeffrey', 'Nicole', 'Marie'}

"""

print("UNIXコマンドを用いると，")
output = subprocess.check_output(["wsl", "cut", "-d", "\t", "-f", "1", filepath, "|", "sort", "|", "uniq"])
print(output.decode("utf-8"))

"""
＊ 出力結果 ＊
Abigail
Aiden
Alexander
Alexis
...

"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

適切なコメントの使用(p.76 6.6コードの意図を書く): 
コメントがコードの理解を助けるために使われています。例えば、どの部分が1列目の文字列を抽出しているのかを説明するコメントがあります。

変数名の意味の明確化(p.10 2.1 明確な単語を選ぶ): 
unique_names という変数名は、その内容を理解しやすくしています。1列目の文字列の異なる値を示すことが期待されるため、unique_names という名前は適切です。



"""