import MeCab
import ipadic
mecab = MeCab.Tagger(ipadic.MECAB_ARGS)
print(mecab.parse("私はYahooプレミアム会員になりました。"))