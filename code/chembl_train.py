import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from dgl import batch
from GNN_utils.utils import Model_molnet,set_seed,get_valid
from GNN_utils.data_process_ours import multi_process, Dataset_multiprocess_ecfp, Collator_ecfp, \
    _compute_df_weights
from GNN_utils.pytorchtools import EarlyStopping
from GNN_utils.mol_tree import Vocab
from GNN_utils.ScaffoldSplit import scaffold_randomized_spliting_xiong
from GNN_utils.nnutils_ours import MLP_revised, tree_gru_onehot_revised, GatEncoder_raw_gru_revised

"""Utilizza un dataset specifico, presumibilmente da ChEMBL, 
e sembra concentrarsi su un singolo tipo di task (class), come indicato dalla lista tasks = ['class'].
Utilizza uno split basato su scaffold per dividere i dati, 
che è comune nei dataset chimici per mantenere la diversità chimica tra i set di training e test."""


def show_figure_loss(train_loss, val_loss, test_loss, trn_roc, val_roc, test_roc):
    plt.figure('Training Process')  # Create art board
    plt.plot(train_loss, 'r-', label='train loss', )
    plt.plot(val_loss, 'b-', label='val loss', )
    plt.plot(test_loss, 'g-', label='test loss')

    plt.title("Training Process For " + args['dataset_name'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=0)
    plt.savefig(args['save'] + '/' + ' for ' + args['dataset_name'] + '_loss.png')

    plt.cla()

    plt.figure('Training Process')  # Create art board
    if args['trn']:
        plt.plot(trn_roc, 'r-', label='train roc-auc', )
    plt.plot(val_roc, 'b-', label='val roc-auc', )
    plt.plot(test_roc, 'g-', label='test roc-auc')

    plt.title("Training Process For " + args['dataset_name'])
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC')
    plt.legend(loc=0)
    plt.savefig(args['save'] + '/' + ' for ' + args['dataset_name'] + '_roc.png')



class All_old(nn.Module):
    def __init__(self, hidden_state_size, vocab_path,nums_task, head, conv, rhead, rconv):
        super(All_old, self).__init__()
        self.vocab = Vocab([x.strip("\r\n ") for x in open(vocab_path)])
        self.hidden_state_size = hidden_state_size

        #initialization of the tree fragmentator layer
        self.GATencoder = tree_gru_onehot_revised(vocab=self.vocab, hidden_size=self.hidden_state_size,
                                          head_nums=head,conv_nums=conv)
        
        #initialization of the raw molecule feature extractor
        self.GATencoder_raw = GatEncoder_raw_gru_revised(hidden_size=self.hidden_state_size,
                                                 head_nums=rhead,conv_nums=rconv)
        #initialization of the DNN for the ECFP
        self.dnn_ecfp = MLP_revised(n_feature=512, n_hidden=[256, 128],
                                    n_output=self.hidden_state_size, dropout=0.1)
        
        #initialization of the final classification layer
        self.classify = MLP_revised(self.hidden_state_size * 3, [32],  # [64,32],[128]
                                    2*nums_task, dropout=0.1)

        self.mol_output = None



    def forward(self, data, device):
        #fp means fingerprint
        _, raw, fp = self.to_device(data, device)
        raw_h, x_r = self.GATencoder_raw(raw)
        tree = self.test_(data['mol_trees'],raw_h,device)
        self.mol_output, x_t = self.GATencoder(tree)
        x_fp = self.dnn_ecfp(fp)
        x = torch.cat([x_t, x_r, x_fp], dim=-1)
        y = self.classify(x)
        return y


    def to_device(self, mol_batch, device):
        tree = batch(mol_batch['mol_trees']).to(device)
        raw = batch(mol_batch['mol_raws']).to(device)
        fp = mol_batch['ecfp'].to(device)

        return tree, raw, fp

    def test_(self,tree,raw_h,device):
        assert len(tree) == len(raw_h)
        all_data = []
        for i in range(len(raw_h)):
            tt = tree[i].nodes_dict # nodes_dict contiene i dettagli di tutti i nodi (o vertici) di quell'albero.
            r = raw_h[i]

            #una clique è un sottoinsieme di vertici di un grafo tale che ogni due vertici distinti sono adiacenti. 
            # In altre parole, una clique in un grafo è un sottoinsieme di vertici che formano un sottografo completo. 
            # Un sottografo completo è un sottografo in cui ogni coppia di vertici è connessa da un arco.
            cliques = []
            for key in tt:
                clique = tt[key]['clique']
                cliques.append(torch.sum(r[clique],dim=0))
            all_data.append(torch.stack(cliques,dim=0))

        assert len(all_data) == len(tree)
        for i in range(len(tree)):
            tree[i].ndata['h'] = all_data[i].cpu()

        return batch(tree).to(device)



def data_load():
    print("============Loading Data============")
    dataset_name = args['dataset_name']
    load_data = pd.read_csv(args['data_path'])
    tasks = ['class']

    try:
        load_data.rename(columns={'smiles': 'Smiles'}, inplace=True)
        print('rename the columns `smiles` to `Smiles`....')
    except:
        pass

    load_data = get_valid(load_data,shuffle=True,random_seed=args['seed'])
    y, task_weights = _compute_df_weights(load_data, tasks)
    print("The numbers of data= %s  \nThe dimension of label= %s" % (load_data.shape[0], y.shape))

    print("============Spliting Data============")


    print('Use scaffold split (8:1:1)')
    trn_index, val_index, test_index = scaffold_randomized_spliting_xiong(smiles_tasks_df=load_data, tasks=tasks,
                                                                      weights=task_weights, random_seed=args['seed'])

    print("============Processing Data============")
    print("argomenti: ")
    print(args)
    multi_process(X=load_data.iloc[trn_index].Smiles,y=y[trn_index],data_type='train',vocab_path=args['vocab'], data_name = dataset_name,workers=args['workers'],reprocess = args['reprocess'])
    multi_process(X=load_data.iloc[val_index].Smiles,y=y[val_index],data_type='val',vocab_path=args['vocab'], data_name = dataset_name,workers=args['workers'],reprocess = args['reprocess'])
    multi_process(X=load_data.iloc[test_index].Smiles,y=y[test_index],data_type='test',vocab_path=args['vocab'], data_name = dataset_name,workers=args['workers'],reprocess = args['reprocess'])


    # Dataloader : generate enable iter object
    train_data = Dataset_multiprocess_ecfp(data_name=dataset_name, data_type='train')
    dataloader_trn = DataLoader(train_data, batch_size=args['bs'], shuffle=True, num_workers=0,
                                collate_fn=Collator_ecfp(), drop_last=False, )

    val_data = Dataset_multiprocess_ecfp(data_name=dataset_name, data_type='val')
    dataloader_val = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=0,
                                collate_fn=Collator_ecfp(), drop_last=False, )

    test_data = Dataset_multiprocess_ecfp(data_name=dataset_name, data_type='test')
    dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0,
                                 collate_fn=Collator_ecfp(), drop_last=False, )

    return dataloader_trn, dataloader_val, dataloader_test, task_weights, tasks


def model_bulid(weights,tasks):
    # model defined
    print("============Building Model============")
    gcn = All_old(hidden_state_size=args["hidden_state_size"], vocab_path=args['vocab'],nums_task=len(tasks),
                  head=args['head'],conv=args['conv'],rhead=args['rhead'],rconv=args['rconv'])

    print("Model #Params: %dK" % (sum([x.nelement() for x in gcn.parameters()]) / 1000,))

    device = args['gpu'] if torch.cuda.is_available() and args['gpu'] != -1 else 'cpu'
    optimizer = optim.Adam(gcn.parameters(), lr=0.005)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = [nn.CrossEntropyLoss(torch.tensor(weight, device=device), reduction='mean') for weight in weights]

    model = Model_molnet(model=gcn, optimizer=optimizer,
                  criterion=loss_function, scheduler=scheduler, device=device, tasks=tasks)

    print(f'use the device:{device}')

    return model


def train(model, trn, val, test):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    earlystop = EarlyStopping(args['patience'], verbose=True, dataset_name=dataset_name, model_save_path=args['save'])

    file = open(args['save'] + '/train_info.txt', 'w')
    best_roc = - float("inf")

    trn_list, val_list, test_list = [], [], []
    trn_roc_list, val_roc_list, test_roc_list = [], [], []
    for e in range(args['epochs']):
        trn_loss = model.train(trn)
        
        if args['trn']:  # verbose the roc-score in training data
            _, trn_roc = model.eval(trn)
            trn_roc_list.append(trn_roc)

        val_loss, val_roc = model.eval(val)
        test_loss, test_roc = model.eval(test)

        date = f'epochs {e} --> loss: train {trn_loss}, val {val_loss}, test {test_loss}\n val_roc:{val_roc} test_roc:{test_roc}\n'
        file.write(date)
        print(date, end='')

        trn_list.append(trn_loss)
        val_list.append(val_loss)
        test_list.append(test_loss)

        val_roc_list.append(val_roc)
        test_roc_list.append(test_roc)

        if args['earlystop']:
            earlystop(val_roc, model.model)
            if earlystop.early_stop:
                print('Early stopping')
                break
        else:
            if val_roc > best_roc:
                best_roc = val_roc
                print('save the best model....')
                torch.save(model.model.state_dict(), os.path.join(args['save'], 'best_model.pt'))

    best_index = np.argmax(val_roc_list)
    val_ = val_roc_list[best_index]
    test_ = test_roc_list[best_index]
    date = 'Finished! roc-score in test-data (best in val-data {}(epochs{})) is: {}'.format(val_, best_index, test_)
    print(date)

    file.write(date)
    file.flush()

    show_figure_loss(trn_list, val_list, test_list, trn_roc_list, val_roc_list, test_roc_list)

    all_res_path = os.path.split(args['save'])[0]
    with open(os.path.join(all_res_path, 'all_res.txt'), 'a+') as f:
        m = args['model']
        seed = args['seed']
        f.write(f'{test_} {m} {seed}\n')
        f.flush()


def main():
    trn, val, test, weights, tasks = data_load()
    model = model_bulid(weights, tasks)
    #questa riga sotto il commento l'hai aggiunta tu perchè il training ti restituisce errore a 246, m = args['model'] !
    args['model'] = model.__class__.__name__
    train(model, trn, val, test)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-path", required=True,
                        help='Path to a csv file for loading a dataset')
    parser.add_argument("-hs", "--hidden-state-size", type=int, default=128,
                        help="feature dimension of graph representation.(default:128)")
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpus used (default device:0, -1 is cpu)')
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max number of epochs for training.(default:150)")
    parser.add_argument("--seed", default=42, type=int,
                        help='random-seed for reproduct')
    parser.add_argument('--reprocess', default=False, action='store_true',
                        help='whether reprocess the data or not.(default:False)')
    parser.add_argument('--earlystop', default=False, action='store_true',
                        help='whether use earlystop or not.(default:False)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience when training if `earlystop` is used')
    parser.add_argument('--workers', type=int, default=8,
                        help='workers for data pre-process')
    parser.add_argument('--trn',default=False, action='store_true',
                        help='whether verbose the training roc')
    parser.add_argument('--save', default='./result_save', help='path to save model and result')
    parser.add_argument("--vocab", default="/home/marco/MSSGAT_paper/MSSGAT/code/dataset/vocabulary_chembl.txt", help="path to vocab file, l'hai aggiunto tu questo argomento!")
    parser.add_argument('--head', default=4, type=int)
    parser.add_argument('--conv', default=2, type=int)
    parser.add_argument('--rhead', default=4, type=int)
    parser.add_argument('--rconv', default=3, type=int)
    parser.add_argument('--bs', default=256, type=int)


    args = parser.parse_args().__dict__

    if not os.path.exists(args["data_path"]):
        raise ValueError("Path to dataset file not found , plz check the path again ")

    dataset_name = os.path.split(args["data_path"])[-1].split('.')[0]
    args['dataset_name'] = dataset_name

    args['save'] = os.path.join(args['save'], dataset_name)
    args['save'] = os.path.join(args['save'], 'process-{}-{}'.format('all', time.strftime('%y%m%d%H%M')))
    if not os.path.exists(args['save']):
        os.makedirs(args['save'])

    # vocab_path = './dataset/vocab_tree_zinc.txt'
    vocab_path = '/home/marco/MSSGAT_paper/MSSGAT/code/dataset/vocabulary_chembl.txt'
    print(vocab_path)
    args['vocab'] = vocab_path

    set_seed(seed=args['seed'])

    main()
