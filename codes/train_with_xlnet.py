import os
import argparse
#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'#使用服务器添加的指令

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.callbacks import Callback
from keras_xlnet.backend import keras
from keras.utils.np_utils import *
from get_xlnet_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--times', default=1, type=int, required=False, help='第几次训练')
parser.add_argument('--nclass', default=3, type=int, required=False, help='几分类')
parser.add_argument('--epoch', default=2, type=int, required=False, help='训练批次')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='训练批次')
parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch_size')
parser.add_argument('--dev_size', default=0.2, type=float, required=False, help='batch_size')
parser.add_argument('--maxlen', default=256, type=int, required=False, help='训练样本最大句子长度')
parser.add_argument('--pretrained_path', default="D:/codes/Xlnet/chinese_xlnet_mid_L-24_H-768_A-12/", type=str, required=False, help='预训练模型目录')
parser.add_argument('--submision_sample_path', default="../subs/submit_example.csv", type=str, required=False, help='提交模板')
parser.add_argument('--do_train', default=True,action='store_true', required=False, help='是否训练')
parser.add_argument('--do_predict', default=False, action='store_true', required=False, help='提交测试')

args = parser.parse_args()# 使用 parse_args() 解析添加的参数

args.train_path = "../data/"+str(args.times)+"/train.tsv"
args.test_path = "../data/"+str(args.times)+"/test.tsv"
args.sub_path = "../subs/"+str(args.times)+"/"
args.finetuned_path = "../models/"+str(args.times)+"/"

args.config_path = args.pretrained_path + 'xlnet_config.json'
args.model_path = args.pretrained_path + 'xlnet_model.ckpt'
args.vocab_path = args.pretrained_path + 'spiece.model'


if os.path.exists(args.sub_path)==False:
    os.mkdir(args.sub_path, mode=0o777)
if os.path.exists(args.finetuned_path)==False:
    os.mkdir(args.finetuned_path, mode=0o777)

if os.path.exists(args.train_path)==False:
    print("本次训练数据未准备")
    os.makedirs(args.train_path, mode=0o777)
    exit()
if os.path.exists(args.sub_path)==False:
    os.makedirs(args.sub_path, mode=0o777)
if os.path.exists(args.finetuned_path)==False:
    os.makedirs(args.finetuned_path, mode=0o777)    

# Read data
class DataSequence(keras.utils.Sequence):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return (len(self.y) + args.batch_size - 1) // args.batch_size

    def __getitem__(self, index):
        s = slice(index * args.batch_size, (index + 1) * args.batch_size)
        return [item[s] for item in self.x], self.y[s]

def generate_sequence(df):
    tokenizer = Tokenizer(args.vocab_path)
    tokens, classes = [], []
    
    for _, row in df.iterrows():
        try:
            text, cls = row[0], row[1]
            encoded = tokenizer.encode(text)[:args.maxlen - 1]
            encoded = [tokenizer.SYM_PAD] * (args.maxlen - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
            tokens.append(encoded)
            classes.append(str(cls))
            print(cls)
        except:
            pass
    tokens = np.array(tokens)
    classes = pd.get_dummies(pd.DataFrame(classes,columns = ["label"]),prefix_sep='_')
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    print(classes[0:3])
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes)

def train():
    model = build_xlnet(args)
    data = pd.read_csv(args.train_path, sep='\t', error_bad_lines=False)
    text_data = data.iloc[:,1] 
    label = data.iloc[:,0]
    
    x_train,x_dev, y_train, y_dev =train_test_split(text_data,label,test_size=args.dev_size, random_state=2020)
    
    train_df=pd.DataFrame(x_train).join(y_train)
    dev_df=pd.DataFrame(x_dev).join(y_dev)

    train_seq = generate_sequence(train_df)
    dev_seq = generate_sequence(dev_df)

    model = build_xlnet(args)
    model.fit_generator(
        generator=train_seq,
        validation_data=dev_seq,
        epochs=args.epoch,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                restore_best_weights=True,
                patience=1,),
        ],
    )

    model.save_weights(args.finetuned_path+"xlnet.h5")

def predict():
    model = build_xlnet(args)
    model.load_weights(args.finetuned_path+"xlnet.h5")

    test_seq = generate_sequence(args.vocab,args.test_path)
    predictions = model.predict_generator(test_seq, verbose=True)
    
    detailed_predictions(predictions)#记录softmax后预测概率原始值
    final_result = predictions.argmax(axis=-1)
    write_csv(final_result)
    
def write_csv(result):
    id = pd.read_csv(args.submision_sample_path)[["id"]]
    result = pd.DataFrame(result,columns = ["y"])
    result = id.join(result)
    result.to_csv(args.sub_path+"sub.csv",index = False)
def detailed_predictions(predictions):
    f = open("../subs/"+str(args.times)+"/predictions.txt","w",encoding = "utf8")
    for each in predictions:
        tem = [str(i) for i in each]
        f.write(" ".join(tem))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    if args.do_train == True:
        train()
    if args.do_predict == True:
        predict()


    