import torch
import torch.utils.data as Data
import tqdm


from tensorboardX import SummaryWriter
from data import *
from model import *


class Framework():
    def __init__(self, args):

        self.batch_size   = args.batch_size
        self.epoch        = args.epoch

        self.device       = 'cuda' if torch.cuda.is_available() else: 'cpu'
        self.data         = DataGenerator(args.path, args.max_length)
        self.model        = GPT2(args.blocks, self.data.vocab_size,args.em_dim) 
        self.optimizer    = torch.optim.adam(self.model.parameters, lr = args.lr)
        self.loss_fn      = nn.CrossEntropyLoss()
        self.train_loader = None
        self.test_loader  = None

        self.writer       = SummaryWriter('./runs/' + args.exp_id)

    def show_model(self):
        print("Trainable Parameters: %d"%sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def get_data(self)
        train_feature, train_label, test_feature, test_label = data.get_data()
       
        train_feature, train_label, test_feature, test_label = \
            torch.FloatTensor(train_feature).to(device), torch.LongTensor(train_label).to(device), \
            torch.FloatTensor(test_feature).to(device), torch.LongTensor(test_label).to(device)
        
        torch_dataset  = Data.TensorDataset(train_feature,train_label)
        self.train_loader   = Data.DataLoader(
            dataset    = torch_dataset,
            batch_size = self.batch_size,
            shuffle    = True,
        )

        torch_dataset  = Data.TensorDataset(test_feature,test_label)
        self.test_loader   = Data.DataLoader(
            dataset    = torch_dataset,
            batch_size = len(self.test_label),
            shuffle    = True,
        )

    def train(self):
        iteration = 0
        for epoch in tqdm(range(self.epoch)):
            for step, x, y in enumerate(self.train_loader):
                out = self.model(x)
                loss = self.loss_fn(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    for x, y in enumerate(self.test_loader):
                        out = self.model(x)
                        test_loss = self.loss_fn(out, y)

                        self.writer.add_scalar('Training Loss', loss, iteration)
                        self.writer.add_scalar('Testing Loss', test_loss, iteration)

                iteration+= 1
                

    def eval(self):
        pass