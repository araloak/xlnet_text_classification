import re
from langconv import *#繁体字转化为简体字

times = 1#使用第几次数据进行处理

def clean(path):
    data = pd.read_csv(path, sep='\t', error_bad_lines=False)
    text = list(data.iloc[:,1])
    label = list(data.iloc[:,0])
    cleaned_text = []
    #数据清理
    for string in text:
        #print("之前:",string)
        try:
            string = Traditional2Simplified(string)
        except:
            pass
        #string = re.sub("\?展开全文c","......",string)
        #string = re.sub("？[？]+","??",string)
        #string = re.sub("\?[\?]+","??",string)
        cleaned_text.append(string)
    write_tsv(path,clean_text,label)
    
def write_tsv(path,text,labels): 
    f=open(path,"w",encoding = "utf8")
    for (string,label) in zip(text,labels):
        f.write(str(label)+"	"+string+"\n")
    f.close()
    
if __name__ == "__main__":
    train_data_path = "../data/"+str(times)+"/train.tsv"
    test_data_path = "../data/"+str(times)+"/test.tsv"
    clean(train_data_path)
    clean(test_data_path)
