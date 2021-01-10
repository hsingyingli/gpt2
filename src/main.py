import argparse

from framework import *


  
def main(args):

    environment = Framework(args)
    environment.show_model()
    print("||   Start to load data ......")
    environment.get_data()
    print("||   %d Training data is Found"%(len(environment.train_loader.dataset)))
    print("||   %d Testing  data is Found"%(len(environment.test_loader.dataset)))
    print("||   Start Training ......")
    environment.train()
    print("|| Complete !!")
    environment.eval()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path'                      , type = str   , default = "./../data/")
    parser.add_argument('--epoch'                     , type = int   , default = 1000000)
    parser.add_argument('--batch_size'                , type = int   , default = 64)
    parser.add_argument('--lr'                        , type = float , default = 4e-4)
    parser.add_argument('--blocks'                    , type = int   , default = 8)
    parser.add_argument('--em_dim'                    , type = int   , default = 64)
    parser.add_argument('--max_length'                , type = int   , default = 51)
    parser.add_argument('--n_head'                    , type = int   , default = 2)
    parser.add_argument('--exp_id'                    , type = str   , default = '1')
    
    args = parser.parse_args()
    print(args)
    main(args)
