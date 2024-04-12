"""
言語処理100本ノック 第1章課題

09. TypoglycemiaPermalink
スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
ただし，長さが４以下の単語は並び替えないこととする．適当な英語の文
（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え，
その実行結果を確認せよ．

"""
import random

# 関数定義
def word_magic(str1: str) -> str:
    """
        各単語の先頭と末尾以外をランダムに並び替える関数
    """ 
    message = ""
       
    if type(str1) is str:
        splited_word_list = str1.split()    # 単語に分割
    
    for word in splited_word_list:
        if len(word) > 4:
            middle_charactor = list(word[1:-1])
            random.shuffle(middle_charactor)
            message += str(word[0] + ''.join(middle_charactor)+ word[-1] + ' ')
        else:
            message += word + ' '
            
    return message

# 出力
print(word_magic("I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."))

# 出力結果
# I clundo’t bvlieee that I cuold atcually uedstrnand what I was randeig : the pmenohenal peowr of the hamun mind .

"""
―リーダブルコードの内容で実践したこと―
・p.10の「2.1 明確な単語を選ぶ」で，
splited_word_listと分割した単語リストであることを示した。

"""