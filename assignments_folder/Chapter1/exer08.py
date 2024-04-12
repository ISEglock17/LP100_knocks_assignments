"""
言語処理100本ノック 第1章課題

08. 暗号文Permalink
与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
・英小文字ならば(219 - 文字コード)の文字に置換
・その他の文字はそのまま出力
この関数を用い，英語のメッセージを暗号化・復号化せよ．

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
・p.171～p.173の「短いコードを書くこと」で，
内包表記とStringにおけるjoin()メソッドを用いて短くまとめた。

"""