"""
言語処理100本ノック 第1章課題

08. 暗号文Permalink
与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ

"""
import re

# 関数定義
def cipher(str1: str) -> str:
    """
        暗号化・複合化を行う関数
    """
    return "".join([chr(219 - ord(s)) if s.islower() else s for s in str1])

# テスト
print("\"heL値lo11sy1\"を暗号化します。")
print(cipher("heL値lo11sy1"))
print("これを複合化します。")
print(cipher(cipher("heL値lo11sy1")))

"""
―リーダブルコードの内容で実践したこと―


"""