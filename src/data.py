import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel

class DataGenerator():
    def __init__(self, path, max_length):
        self.tokenizer  = AutoTokenizer.from_pretrained('bert-base-chinese') # 21128
        self.embedding  = AutoModel.from_pretrained('bert-base-chinese')
        self.vocab_size = 21128

        self.path       = path
        self.length     = max_length
    def get_data(self):
        path = self.path +  'train/'
        train_feature = []
        train_label   = []
        test_feature  = []
        test_label    = []


        for dirname, _ , filename in os.walk(path):
            for f in filename:
                f = open(path + f)
                text = ''
                for line in f:
                    text += line
                
                text    =  np.array(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)))
                tokens   = []
                for i in range(self.length, len(text)):
                    tokens.append(text[i-self.length:i])
                
                tokens = np.array(tokens)
                train_feature.extend(tokens[:,:-1])
                train_label.extend(tokens[:,-1])
        
        
        path = self.path +  'test/'
        for dirname, _ , filename in os.walk(path):
            for f in filename:
                f = open(path + f)
                text = ''
                for line in f:
                    text += line
                
                text    =  np.array(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)))
                tokens   = []
                for i in range(self.length, len(text)):
                    tokens.append(text[i-self.length:i])
                
                tokens = np.array(tokens)
                test_feature.extend(tokens[:,:-1])
                test_label.extend(tokens[:,-1])




        train_feature, train_label = np.array(train_feature), np.array(train_label)
        test_feature, test_label = np.array(test_feature), np.array(test_label)

        
        return train_feature, train_label, test_feature, test_label

    def Decode_Token(self, ids):
        return np.array(self.tokenizer.convert_ids_to_tokens(ids))

if __name__ == "__main__":
    data = DataGenerator('./../Data/',51)
    text = data.get_data(mode = 'train')
    

    