import javalang
import re 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import glob
import re
import pickle
import os
nltk.download("punkt")
#xu ly word cu phap vd: anhNguyenNhat -> anh nguyen nhat
def process_word(word):
    char = []
    ps=PorterStemmer()
    for i in range(0, len(word)):
        if word[i].islower() == True:
             char.append(word[i])
        else:
            char.append(' ')
            char.append(word[i].lower())
    words=[]
    for word in word_tokenize(''.join(char)):
        if(len(ps.stem(word))>1):
            words.append(ps.stem(word))
    return words

# function read folder
def openFolder(path, files, agr):
    files.extend(glob.glob(os.path.join(path, agr)))
    for file in os.listdir(path):
        fullpath = os.path.join(path, file)
        if os.path.isdir(fullpath) and not os.path.islink(fullpath):
            openFolder(fullpath,files,agr)
def preprocessing_source():
    name_files=[]
    name_sources=[]
    openFolder(r'/home/anhbm/Duc/Eclipse',name_files,"*.java")
    srs=[]
    for name_file in name_files:
        try:
            source_file=[]
            with open(name_file,'r') as f:
                source=f.readlines()
            f.close()
            if(len(source)==0):
                continue
            source=' '.join(source)
            try:
                tree=javalang.parse.parse(source)
            except javalang.parser.JavaSyntaxError:
                continue
            for x,y in tree.filter(javalang.tree.ClassDeclaration):
                source_file.extend(process_word(y.name))
            for a,b in tree.filter(javalang.tree.MethodDeclaration):
                source_file.extend(process_word(b.name))
            srs.append(source_file)
            name_sources.append(name_file)
        except UnicodeDecodeError:
            continue
    return srs,name_sources


