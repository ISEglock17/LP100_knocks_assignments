"""
言語処理100本ノック 第3章課題

26. 強調マークアップの除去
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ（参考: マークアップ早見表）．

"""
import json

# 初期化
folderpath = "./assignments_folder/Chapter3/"
filename = "jawiki-country.json"

# JSONファイル読み込み
with open(folderpath + filename, "r", encoding="utf-8") as f:
    data_list = f.readlines()
    article_list = [json.loads(data) for data in data_list]

# イギリスに関する記事JSONファイルの抽出
    UK_article = str(list(filter(lambda x: x["title"] == "イギリス", article_list))[0])

# 課題26
import re
pattern = "基礎情報(.*?\<references/\>\\\\n)"    # イギリスの記事を見ると，referencesのところが最後となっているため，そこまでを抜き出す
basic_information = re.findall(pattern, UK_article)[0]
# print(basic_information) #デバッグ用出力

pattern = "(?<=\\\\n\|)(.*?) *= *(.*?)(?=\\\\n)"    # 後読みと先読みを活用して前後が\n| \nで囲まれていることを条件とし，"="の前後のテキストを抽出する
basic_information_set = re.findall(pattern, basic_information)
# print(basic_information_set) # デバッグ用出力

info_dic = {key: value for key, value in basic_information_set} # 辞書の生成

# 強調マークアップの削除
pattern = "(\\\'){2,5}"
info_dic2 = {key: re.sub(pattern , "", value) for key, value in info_dic.items()}    # re.subを利用して，\'が2～5回繰り返されている箇所を削除する　https://www.mediawiki.org/wiki/Help:Formatting/ja　参照
print(info_dic2)    # 出力


"""
＊　出力結果　＊
{'略名': 'イギリス', '日本語国名': 'グレートブリテン及び北アイルランド連合王国', '公式国名': '{{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />', '国旗画像': 'Flag of the United Kingdom.svg', '国章画像': '[[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]', '国章リンク': '（[[イギリスの国章|国章]]）', '標語': '{{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）', '国歌': "[[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />\\'\\'神よ女王を護り賜え\\'\\'<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}", '地図画像': 'Europe-UK.svg', '位置画像': 'United Kingdom (+overseas territories) in the World (+Antarctica claims).svg', '公用語': '[[英語]]', '首都': '[[ロンドン]]（事実上）', '最大都市': 'ロンドン', '元首等肩書': 
'[[イギリスの君主|女王]]', '元首等氏名': '[[エリザベス2世]]', '首相等肩書': '[[イギリスの首相|首相]]', '首相等氏名': '[[ボリス・ジョンソン]]', '他元首等肩書1': '[[貴族院 (イギリス)|貴族院議長]]', '他元首等氏名1': '[[:en:Norman Fowler, Baron Fowler|ノーマン・ファウラー]]', '他元首等肩書2': '[[庶民院 (イギリス)|庶民院議長]]', '他元首等氏名2': '{{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}', '他元首等肩書3': '[[連合王国最高裁判所|最高裁判
所長官]]', '他元首等氏名3': '[[:en:Brenda Hale, Baroness Hale of Richmond|ブレンダ・ヘイル]]', '面積順位': '76', '面積大きさ': '1 E11', '面積値': '244,820', '水面積率': '1.3%', '人口統計年': '2018', '人口順位': '22', '人
口大きさ': '1 E7', '人口値': '6643万5600<ref>{{Cite web|url=https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates|title=Population estimates - Office for National Statistics|accessdate=2019-06-26|date=2019-06-26}}</ref>', '人口密度値': '271', 'GDP統計年元': '2012', 'GDP値元': '1兆5478億<ref name="imf-statistics-gdp">[http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=IMF>Data and Statistics>World Economic Outlook Databases>By Countrise>United Kingdom]</ref>', 'GDP統計年MER': '2012', 'GDP順位MER': '6', 'GDP値MER': '2兆4337億<ref name="imf-statistics-gdp" />', 'GDP統計年': '2012', 'GDP順位': '6', 'GDP値': '2兆3162億<ref name="imf-statistics-gdp" />', 'GDP/人': '36,727<ref name="imf-statistics-gdp" />', '建国形態': '建国', '確立形態1': '[[イングランド王国]]／[[スコットランド王国]]<br />（両国とも[[合同法 (1707年)|1707年合同法]]まで）', '確立年月日1': '927年／843年', '確立形態2': '[[グレートブリテン王国]]成立<br />（1707年合同法）', '確立年月日2': '1707年{{0}}5月{{0}}1日', '確立形態3': '[[グレートブリテン及びアイルランド連合王国]]成立<br />（[[合同法 (1800年)|1800年合同法]]）', '確立年月日3': '1801年{{0}}1月{{0}}1日', '確立形態4': " 
現在の国号「\\'\\'\\'グレートブリテン及び北アイルランド連合王国\\'\\'\\'」に変更", '確立年月日4': '1927年{{0}}4月12日', '通貨': '[[スターリング・ポンド|UKポンド]] (£)', '通貨コード': 'GBP', '時間帯': '±0', '夏時間': '+1', 'ISO 3166-1': 'GB / GBR', 'ccTLD': '[[.uk]] / [[.gb]]<ref>使用は.ukに比べ圧倒的少数。</ref>', '国際電話番号': '44', '注記': '<references/>'}
PS C:\Users\ISE\Desktop\稲葉研究室\100本ノック 課題>
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記の利用: リスト内包表記が使われており、コンパクトでわかりやすいコードになっています。例えば、article_list の生成部分がリスト内包表記を用いています。
辞書の生成: info_dic の生成部分で、辞書内包表記が使われています。これにより、効率的でコンパクトな辞書の生成が行われています。

・p.10の「2.1 明確な単語を選ぶ」で，
適切な変数名: 変数名は適切に命名されており、理解しやすいです。例えば、folderpath や filename はそれぞれファイルのパスとファイル名を表しています。



コメントの活用: コメントが適切に使われており、コードの動作や意図が説明されています。例えば、# JSONファイル読み込み や # 課題26 のように、どの部分がどのような処理をしているかを説明しています。
デバッグ用のコメント: デバッグ用のコメントが残されており、必要に応じてデバッグやトラブルシューティングが行えるようになっています。
正規表現パターンの説明: 正規表現パターンに対してもコメントが付けられており、そのパターンが何を意味するかがわかりやすくなっています。

"""