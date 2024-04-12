"""
言語処理100本ノック 第1章課題

06. 集合Permalink
“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

"""
# n-gramメソッド
def n_gram(str1: str, split_num: int, flag: bool = False) -> list:
    """
        n-gramを作るメソッド
        flagをTrueにすると，文字n-gram，
            　Falseにすると，単語n-gram
        が得られる。
    """
    n_gram_list = []
    
    if type(str1) is str:
        splited_word_list = str1.split()    # 単語に分割
        
        if flag is True:
            splited_word_list = ''.join(splited_word_list)
        
        for i in range(len(splited_word_list) - split_num + 1):
            n_gram_list.append(splited_word_list[i: i + split_num])
        
        return n_gram_list

# bi-gram集合の生成
str1 = "paraparaparadise"
str2 = "paragraph"

X = set(n_gram(str1, 2, True))
Y = set(n_gram(str2, 2, True))

# 出力
print("X")
print(X)
print("Y")
print(Y)
print("XとYの和集合")
print(X | Y)       
print("XとYの積集合")
print(X & Y)       
print("XとYの差集合 X - Y")
print(X - Y)       
print("XとYの差集合 Y - X")
print(Y - X)       

print("’se’というbi-gramがXに", end="")
print("含まれる" if "se" in X else "含まれない")
print("’se’というbi-gramがYに", end="")
print("含まれる" if "se" in Y else "含まれない")


# 出力結果
"""
X
{'se', 'ra', 'ad', 'is', 'ar', 'pa', 'ap', 'di'}
Y
{'ra', 'ag', 'ar', 'pa', 'gr', 'ph', 'ap'}
XとYの和集合
{'se', 'ra', 'ag', 'ad', 'is', 'ar', 'pa', 'gr', 'ph', 'ap', 'di'}
XとYの積集合
{'ra', 'ap', 'ar', 'pa'}
XとYの差集合 X - Y
{'di', 'se', 'ad', 'is'}
XとYの差集合 Y - X
{'gr', 'ag', 'ph'}
’se’というbi-gramがXに含まれる
’se’というbi-gramがYに含まれない
"""



"""
―リーダブルコードの内容で実践したこと―
・p.10の「2.1 明確な単語を選ぶ」で，
splited_word_listと分割した単語リストであることを示した。
split_numで分割数が引数であることを明確化した。
p.4「1.3 小さなことは絶対にいいこと?」に従い，
あえてn-gram生成を内包表記しなかった。

"""