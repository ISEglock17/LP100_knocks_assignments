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
        splited_word_list = str1.split()    #単語に分割
        
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
"""
―リーダブルコードの内容で実践したこと―


"""