import os,pickle,multiprocessing,torch
import shutil

import dgl
import numpy as np
from dgl import DGLGraph
from torch.utils.data import Dataset
from tqdm import tqdm

from .mol_tree import Vocab, DGLMolTree
from .chemutils import atom_features, get_mol, get_morgan_fp, bond_features,get_dgl_node_feature,get_dgl_bond_feature

"""
2. data_process_ours.py
Contenuto: Gestisce il pre-processamento dei dati specifici per il progetto.
essenziale per preprocessare i dati molecolari e trasformarli in una rappresentazione utilizzabile per l'addestramento del modello MSSGAT. 
Utilizza sia processi singoli che multipli per gestire grandi dataset in modo efficiente. 
Le classi Dataset e Collator facilitano il caricamento e la gestione dei dati durante l'addestramento.
Funzioni Principali:
multi_process: Funzione per il pre-processamento parallelo dei dati.
Dataset_multiprocess_ecfp: Classe che gestisce il dataset, particolarmente per le rappresentazioni ECFP delle molecole.
Collator_ecfp: Funzione che prepara i batch di dati per il training."""

def _unpack_field(examples, field):  # get batch examples (dictionary) values by key = Estrae i valori di un campo specifico da un elenco di dizionari di esempi
    return [e[field] for e in examples]

def _set_node_id(mol_tree, vocab):  #hash函数，找到mol_tree中每个顶点(官能团簇)在vocab中的索引
    """Assegna identificatori ai nodi di un albero molecolare utilizzando un vocabolario.
    Ritorna una Lista di identificatori di nodi.
    Viene usata per poter assergnare degli indici a cliques o nodi di un albero molecolare già costruito."""
    wid = []
    for i, node in enumerate(mol_tree.nodes_dict):
        mol_tree.nodes_dict[node]['idx'] = i
        wid.append(vocab.get_index(mol_tree.nodes_dict[node]['smiles']))
    return wid



# single_process for data preprocess
"""Descrizione: Preprocessa i dati molecolari e li salva in un file, utilizzando un singolo processo.
Input:
X: Lista di stringhe SMILES delle molecole.
y: Lista di etichette associate alle molecole.
vocab_path: Percorso del vocabolario.
data_type: Tipo di dati (train, val, test).
data_name: Nome del dataset.
reprocess: Flag per ri-processare i dati se già esistente il file di output.

Output: Salva i dati preprocessati in un file."""

#Viene importato in molnet_train.py ma mai usato (probabilmente perchè favorita versione multiprocess)
def sigleprocess_get_graph_and_save(X, y, vocab_path, data_type, data_name=None,reprocess = False ):
    assert data_name is not None
    assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"

    save_path = './code/dataset/graph_data_ours'
    data_path = save_path + '/' + data_name + '_' + data_type + '.p'

    if os.path.exists(data_path) and not reprocess:
        print(data_name + '_' + data_type + ' is already finshed')
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vocab = Vocab([x.strip("\r\n ") for x in open(vocab_path)])
    X_zip = list(zip(X, y)) # zip the X (SMILES strings) and y (labels) as tuple
    return_list = []
    for idx, (smiles, label) in enumerate(X_zip):
        # mol_tree = DGLMolTree(smiles)  # mol_tree
        # wid = _set_node_id(mol_tree, vocab)  # idx_cliques

        # get the raw mol graph
        mol = get_mol(smiles)

        feats = get_dgl_node_feature(mol)
        mol_raw = DGLGraph()  #una struttura dati fornita dalla libreria Deep Graph Library (DGL), progettata per rappresentare grafi in modo efficiente e per facilitare l'implementazione e l'addestramento di modelli di deep learning su grafi. 
        mol_raw.add_nodes((len(mol.GetAtoms())))
        mol_raw.ndata['h'] = feats

        for bonds in mol.GetBonds():
            src_id = bonds.GetBeginAtomIdx()
            dst_id = bonds.GetEndAtomIdx()
            mol_raw.add_edges([src_id, dst_id], [dst_id, src_id]) #By adding two edges with swapped source and destination nodes, you are essentially creating a bidirectional connection between the nodes so making it undirected.

        mol_raw = dgl.add_self_loop(mol_raw)
        e_f = get_dgl_bond_feature(mol)
        mol_raw.edata['e_f'] = e_f
        result = {'mol_raw': mol_raw,'label':label}
        return_list.append(result)
        print('\r{}/{} molecules to process..'.format(idx + 1, len(X)), end='')

    print("\n  Process of {} is Finshed".format(data_name + '_' + data_type))

    with open(data_path, 'wb') as file:
        pickle.dump(return_list, file)
    file.close()



# praticamente fa il lavoro di sigleprocess_get_graph_and_save ma aggiunge ecfp  -> mette insieme più molecole processate in una lista e serve in multi_process
def get_batchs(X,vocab,data_path):
    """Input
    X: Lista di tuple contenenti stringhe SMILES e le etichette corrispondenti.
    vocab: Vocabolario delle sottostrutture molecolari.
    data_path: Percorso del file di destinazione per salvare i dati preprocessati.

    Output
    Grafi molecolari preprocessati salvati in file .pkl."""
    return_res = []
    for idx,(smiles,label) in enumerate(X):
        mol_tree = DGLMolTree(smiles)  # mol_tree
        if mol_tree.warning == True:
            continue
        mol_tree = dgl.add_self_loop(mol_tree)  # mol_tree 是否加自环边
        wid = _set_node_id(mol_tree, vocab)  # idx_cliques

        # get the raw mol graph
        atom_list = []
        mol = get_mol(smiles)

        # # original 44 feat size
        # for atoms in mol.GetAtoms():
        #     atom_f = atom_features(atoms)  # one-hot features for atoms
        #     atom_list.append(atom_f)
        # atoms_f = np.vstack(atom_list)  # get the atoms features
        #
        # mol_raw = DGLGraph()
        # mol_raw.add_nodes(len(mol.GetAtoms()))
        # mol_raw.ndata['h'] = torch.Tensor(atoms_f)


        ## dgl 74 feat size  revise by 6/26
        feats = get_dgl_node_feature(mol)
        mol_raw = DGLGraph()
        mol_raw.add_nodes((len(mol.GetAtoms())))
        mol_raw.ndata['h'] = feats


        for bonds in mol.GetBonds():
            src_id = bonds.GetBeginAtomIdx()
            dst_id = bonds.GetEndAtomIdx()
            mol_raw.add_edges([src_id, dst_id], [dst_id, src_id])

        mol_raw = dgl.add_self_loop(mol_raw)
        e_f = get_dgl_bond_feature(mol)
        mol_raw.edata['e_f'] = e_f


        # ############################ add e_features
        # edges_list = []
        # for bonds in mol.GetBonds():
        #     src_id = bonds.GetBeginAtomIdx()
        #     dst_id = bonds.GetEndAtomIdx()
        #     mol_raw.add_edges([src_id, dst_id], [dst_id, src_id])
        #     edges_list.append(bond_features(bonds, self_loop=True))
        #     edges_list.append(bond_features(bonds, self_loop=True))
        #
        # for idx in range(len(mol.GetAtoms())):
        #     mol_raw.add_edges([idx],[idx])  # add self-loop
        #     edges_list.append(np.array([.0]*10 + [1.]))
        #
        # edges_f = np.vstack(edges_list)
        # mol_raw.edata['e_f'] = torch.Tensor(edges_f)
        # #####################  add edge features

        ecfp = get_morgan_fp(smiles)
        result = {'mol_tree': mol_tree, 'wid': wid, 'mol_raw': mol_raw,'label':label,'fp':ecfp}
        
        return_res.append(result)

        print('\r{}/{} molecules to process..'.format(idx + 1, len(X)), end='')

    id = os.getpid()
    print("\n\n\n\nReturn res len: ",len(return_res))
    with open(data_path + f'/{id}.pkl', 'wb') as file:
        print("\n\n\n\nsave the result to file.. ", data_path + f'/{id}.pkl\n\n\n')
        pickle.dump(return_res, file)
    #return id

"""Descrizione: Preprocessa i dati molecolari utilizzando più processi e li salva in un file.
    Input:
    X: Lista di stringhe SMILES delle molecole.
    y: Lista di etichette associate alle molecole.
    data_type: Tipo di dati (train, val, test).
    vocab_path: Percorso del vocabolario.
    data_name: Nome del dataset.
    workers: Numero di processi da utilizzare.
    reprocess: Flag per ri-processare i dati.

    Output: Salva i dati preprocessati in un unico file che concatena i contenuti di tutti i file temporanei."""
def multi_process(X,y,data_type,vocab_path, data_name = None,workers=8,reprocess = False):
    assert data_name is not None
    assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"

    if workers == -1:
        workers = multiprocessing.cpu_count()

    print(f"Use {workers} cpus to process: ")

    save_path = './code/dataset/graph_data_ours'
    data_path = save_path + '/' + data_name + '_' + data_type + '.p'

    if os.path.exists(data_path) and not reprocess:
        print(data_name + '_' + data_type + ' is already finshed')
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tmp_path = './code/tmp'
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path) # remove the tmp_path if exists
    os.makedirs(tmp_path)

    vocab = Vocab([x.strip("\r\n") for x in open(vocab_path)]) # get the vocab from the file and remove the '\r\n' in the end of each line
    X_zip = list(zip(X, y))
    radio = len(X_zip)//workers # split the data_set for multiprocess so radio represent the number of data for each process
    jobs = []
    #the loop below is to create the process get_batchs and start the process
    for i in range(workers):
        if i == workers-1:
            X = X_zip[i * radio:] # the last process get the rest of the data
        else:
            X = X_zip[i * radio:(i+1) * radio] # split the data_set for multiprocess 
        s = multiprocessing.Process(target=get_batchs,kwargs={'X':X,'vocab':vocab,'data_path':tmp_path})
        jobs.append(s)
        s.start()

    for proc in jobs:
        proc.join()    # join process until the main process finished

    print("\n  Process of {} is Finshed".format(data_name + '_' + data_type))

    concat_result = []
    #below loop is to concatenate the results from each process
    for name in tqdm(os.listdir(tmp_path),'===process'): #for each file in the tmp_path where each file is the result from each process
        with open(tmp_path + f'/{name}','rb') as file: 
            d = pickle.load(file) #load the result from each process
        concat_result.extend(d) #concatenate the result from each process

    with open(data_path,'wb') as file:
        pickle.dump(concat_result,file) #save the concatenated result to the data_path

    shutil.rmtree(tmp_path) #remove the tmp_path after the process finished

"""Descrizione: Classe Dataset per caricare dati preprocessati con impronte digitali ECFP.
Metodi:
__init__(self, data_name, data_type, load_path='./dataset/graph_data_ours'): Inizializza il dataset.
__len__(self): Restituisce il numero di esempi nel dataset.
__getitem__(self, idx): Restituisce un esempio dal dataset fornito di mol_tree, wid, mol_raw, label e fps."""
class Dataset_multiprocess_ecfp(Dataset):   # torch.dataset is abstract class, need to override the __len__ and __getitem__
    def __init__(self,data_name, data_type,load_path='./code/dataset/graph_data_ours'):
        self.data_name = data_name
        self.data_path = load_path + '/' + data_name + '_' + data_type + '.p'

        assert os.path.exists(self.data_path),"not exists the path to data.p"
        assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"
        print(f"\n\n\n\ndata path is {self.data_path}\n\n\n\n\n")
        self.data_and_label = pickle.load(open(self.data_path,'rb'))

    def __len__(self):
        return len(self.data_and_label)

    def __getitem__(self, idx):
        '''get datapoint with index'''
        mol_tree = self.data_and_label[idx]['mol_tree'] # oggetto di classe DGLMolTree
        wid = self.data_and_label[idx]['wid']
        mol_raw = self.data_and_label[idx]['mol_raw'] 
        label = self.data_and_label[idx]['label']
        fps = self.data_and_label[idx]['fp']

        result = {'mol_tree': mol_tree, 'wid': wid, 'class': label, 'mol_raw': mol_raw,'fps':fps}

        return result

"""Descrizione: Classe Collator per raggruppare batch di dati preprocessati con impronte digitali ECFP.
Metodi:
__call__(self, examples): Ritorna un batch di dati raggruppati.

Usato in DataLoader in chemprop_train.py come argomento del parametro "collate_fn" per raggruppare i dati preprocessati in batch per l'addestramento del modello MSSGAT.
Citando testualmente infatti - 
            collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset."""
class Collator_ecfp(object):   # get the batch_data as input, __call__ must be implement
    '''get list of trees and label'''
    def __call__(self, examples):
        mol_trees = _unpack_field(examples, 'mol_tree')
        wid = _unpack_field(examples, 'wid')
        label = _unpack_field(examples, 'class')
        mol_raws=_unpack_field(examples,'mol_raw')
        ecfp = _unpack_field(examples,'fps')

        # wid is the index of cliques on word embedding matrix
        for _wid, mol_tree in zip(wid, mol_trees):  # zip wid,mol_trees,label as tuple
             mol_tree.ndata['wid'] = torch.LongTensor(_wid) # set the node data of mol_tree with the index of cliques and convert it to tensor

        ecfp = torch.Tensor(np.vstack(ecfp)) # get the ecfp and convert it to tensor and stack it vertically 

        batch_data = {'mol_trees': mol_trees,'class':np.array(label),
                      'mol_raws': mol_raws,'ecfp':ecfp}

        return batch_data

"""Descrizione: Classe Dataset per caricare altri tipi di dati preprocessati."""
class Dataset_others(Dataset):   # torch.dataset is abstract class, need to override the __len__ and __getitem__
    def __init__(self,data_name, data_type,load_path='./code/dataset/graph_data_ours'):
        self.data_name = data_name
        self.data_path = load_path + '/' + data_name + '_' + data_type + '.p'

        assert os.path.exists(self.data_path),"not exists the path to data.p"
        assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"

        self.data_and_label = pickle.load(open(self.data_path,'rb'))

    def __len__(self):
        return len(self.data_and_label)

    def __getitem__(self, idx):
        '''get datapoint with index'''
        mol_raw = self.data_and_label[idx]['mol_raw']
        label = self.data_and_label[idx]['label']

        result = {'class': label, 'mol_raw': mol_raw}

        return result

"""Descrizione: Classe Collator per raggruppare batch di altri tipi di dati preprocessati."""
class Collator_others(object):   # get the batch_data as input, __call__ must be implement
    '''get list of trees and label'''
    def __call__(self, examples):
        label = _unpack_field(examples, 'class')
        mol_raws=_unpack_field(examples,'mol_raw')

        batch_data = {'class':np.array(label),
                      'mol_raws': mol_raws}

        return batch_data

"""Descrizione: Calcola i pesi per ogni etichetta per evitare dataset sbilanciati.
Input:
df: DataFrame contenente i dati.
tasks: Lista di compiti/etichette.
Output: Array di etichette e lista di pesi.

Usato in chemprop_train.py per calcolare i pesi delle etichette per evitare dataset sbilanciati durante l'addestramento del modello MSSGAT.
Tali pesi inseriti come input di scaffold_randomized_spliting_xiong.py per dividere il dataset in training e validation set."""
def _compute_df_weights(df, tasks):
    weights = []
    for i, task in enumerate(tasks):
        negative_df = df[df[task] == 0][["Smiles", task]]
        positive_df = df[df[task] == 1][["Smiles", task]]
        try:
            weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                            (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])  # 计算正负样本比例权重
        except:
            weights.append([1.0,1.0])

    n_samples = df.shape[0]
    y = np.hstack([np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks]) # get the labels from the df and reshape it to (n_samples,1) and stack it horizontally 

    return y.astype(float), weights

""""Descrizione: Estrae le etichette per compiti di regressione.
Input:
df: DataFrame contenente i dati.
tasks: Lista di compiti/etichette.
Output: Array di etichette."""
def regression_y(df, tasks):
    y = df[tasks].values
    return y.astype(float) # convert the y to float
