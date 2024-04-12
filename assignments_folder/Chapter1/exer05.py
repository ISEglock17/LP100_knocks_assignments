"""
言語処理100本ノック 第1章課題

05. n-gramPermalink
与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

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
            splited_word_list = ''.join(splited_word_list)  # 文字でn-gramを作るため，単語の文字列を結合
        
        for i in range(len(splited_word_list) - split_num + 1):
            n_gram_list.append(splited_word_list[i: i + split_num])
        
        return n_gram_list

# テスト
print("単語bi-gramだと，")
print(n_gram("I am an NLPer", 2))
print("文字bi-gramだと")
print(n_gram("I am an NLPer", 2, True))
        
        

    
"""
―リーダブルコードの内容で実践したこと―
・p.10の「2.1 明確な単語を選ぶ」で，
splited_word_listと分割した単語リストであることを示した。
split_numで分割数が引数であることを明確化した。
p.4「1.3 小さなことは絶対にいいこと?」に従い，
あえてn-gram生成を内包表記しなかった。

n-gramとは何か，次のサイトを参考にするとよい。
自然言語処理に出てくるn-gramとはなに？: https://mojitoba.com/2019/09/23/what-is-ngram/

"""