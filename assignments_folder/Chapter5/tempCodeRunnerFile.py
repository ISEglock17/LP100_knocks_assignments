
from itertools import combinations
import re

# 文章の中から対象となる文を取得
sentence = sentences[2]

# 名詞を含む文節のインデックスを集める
nouns = [i for i, chunk in enumerate(sentence.chunks) if any(morph.pos == "名詞" for morph in chunk.morphs)]

# 名詞を含む文節のペアをすべて取得する
for i, j in combinations(nouns, 2):
    path_I = []  # 文節iから始まるパス
    path_J = []  # 文節jから始まるパス
    
    # 文節iから文節jへのパスを取得する
    while i != j:
        if i < j:  # 文節iの構文木経路上に文節jが存在する場合
            path_I.append(i)
            i = sentence.chunks[i].dst
        else:  # 文節iの構文木経路上に文節jが存在しない場合
            path_J.append(j)
            j = sentence.chunks[j].dst
    
    if len(path_J) == 0:  # 文節Iの構文木上に文節Jが存在する場合
        X = "X" + "".join(morph.surface for morph in sentence.chunks[path_I[0]].morphs if morph.pos != "名詞" and morph.pos != "記号") 
        Y = "Y" + "".join(morph.surface for morph in sentence.chunks[i].morphs if morph.pos != "名詞" and morph.pos != "記号")
        chunk_X = re.sub("X+", "X", X)
        chunk_Y = re.sub("Y+", "Y", Y)
        path_ItoJ = [chunk_X] + ["".join(morph.surface for n in path_I[1:] for morph in sentence.chunks[n].morphs if morph.pos != "記号")] + [chunk_Y]
        print(" -> ".join(path_ItoJ))  # パスを表示
    else:  # 文節Iの構文木上に文節Jが存在しない場合
        X = "X" + "".join(morph.surface for morph in sentence.chunks[path_I[0]].morphs if morph.pos != "名詞" and morph.pos != "記号") 
        Y = "Y" + "".join(morph.surface for morph in sentence.chunks[path_J[0]].morphs if morph.pos != "名詞" and morph.pos != "記号") 
        chunk_X = re.sub("X+", "X", X)
        chunk_Y = re.sub("Y+", "Y", Y)
        chunk_k = "".join(morph.surface for morph in sentence.chunks[i].morphs if morph.pos != "記号")
        path_X = [chunk_X] + ["".join(morph.surface for n in path_I[1:] for morph in sentence.chunks[n].morphs if morph.pos != "記号")]
        path_Y = [chunk_Y] + ["".join(morph.surface for n in path_J[1:] for morph in sentence.chunks[n].morphs if morph.pos != "記号")]
        print(" | ".join([" -> ".join(path_X), " -> ".join(path_Y), chunk_k]))  # パスを表示
