import torch
import torch.nn as nn
from dgl import batch
import dgl
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import kaiming_uniform_,zeros_, constant_

init_feats_size = 74  # 44 original 74 dgl
init_efeats_size = None

"""nnutils_ours.py
Contenuto: Contiene le implementazioni delle reti neurali utilizzate nel progetto.
Funzioni Principali:
MLP_revised: Implementazione di un perceptron multistrato per la classificazione o la regressione.
tree_gru_onehot_revised: Rete neurale che elabora gli alberi molecolari.
GatEncoder_raw_gru_revised: Encoder che utilizza l'attenzione grafica per elaborare le caratteristiche grezze delle molecole."""

# 2021/12/18 revised
""""
La classe MultiHead_gat implementa una variante multi-testa della rete di attenzione grafica (GAT).

Metodi:
- __init__(self, node_feats, heads=3, negative_slope=0.2): Inizializza la rete con il numero di caratteristiche dei nodi, il numero di teste di attenzione e il coefficiente angolare negativo della funzione LeakyReLU.
- message_func(self, edges): Calcola i messaggi tra i nodi, determinando i coefficienti di attenzione e preparando i messaggi da passare ai nodi di destinazione.
- reduce_func(self, nodes): Aggrega i messaggi ricevuti dai nodi di destinazione, normalizzando i coefficienti di attenzione, pesando le caratteristiche dei nodi sorgente, e aggiornando le caratteristiche dei nodi di destinazione.
- forward(self, graph, h): Definisce il passaggio in avanti della rete, trasformando le caratteristiche dei nodi, applicando l'attenzione multi-testa tramite `update_all` e aggiornando le caratteristiche dei nodi.

Legami tra le Funzioni:
- `message_func` e `reduce_func` sono collegate tramite `update_all`, che gestisce il passaggio dei messaggi tra i nodi del grafo. `message_func` calcola i messaggi da inviare ai nodi di destinazione, mentre `reduce_func` aggrega questi messaggi per aggiornare le caratteristiche dei nodi di destinazione.
- `forward` chiama `update_all` per orchestrare l'interazione tra `message_func` e `reduce_func`, assicurando che i messaggi siano calcolati, inviati e aggregati correttamente per aggiornare le caratteristiche dei nodi nel grafo.
"""

class MultiHead_gat(nn.Module):
    def __init__(self, node_feats, heads=3, negative_slope=0.2):  #num caratteristiche nodi, numero teste e coeff angolare della LeakyReLU sono gli argomenti
        super(MultiHead_gat, self).__init__()
        self.node_feats = node_feats
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_node = nn.Linear(node_feats, node_feats*heads,bias=True) # nn.Linear: applica una trasformazione lineare ai dati in ingresso con input shape (node_feats) e output shape (node_feats*heads)
        """1: che è la Dimensione batch, specifica che i pesi sono applicati a ciascun batch singolarmente. Qui, è impostato su 1 poiché il peso di attenzione è calcolato per ogni batch individualmente.
        heads: Numero di teste di attenzione e Specifica il numero di meccanismi di attenzione paralleli. Ogni testa di attenzione apprende una rappresentazione differente delle relazioni tra i nodi.
        2 * node_feats: Numero totale di caratteristiche concatenate dai nodi sorgente e destinazione. 2 * node_feats rappresenta la concatenazione delle caratteristiche del nodo sorgente e del nodo destinazione. Ogni nodo ha node_feats caratteristiche, quindi la concatenazione delle caratteristiche di due nodi richiede 2 * node_feats."""
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 2*node_feats)) #weight_triplet_att: pesi per la funzione di attenzione triplet (1, heads, 2*node_feats) 
       
        """La funzione di scala si riferisce alla trasformazione lineare applicata alle caratteristiche aggregate delle teste di attenzione. 
        Questo step ridimensiona le caratteristiche aggregate per riportarle alla dimensione originale dei nodi, 
        assicurando che le nuove caratteristiche combinate siano nella giusta scala per l'input successivo. 
        Questo è essenziale per mantenere coerenza nelle dimensioni delle caratteristiche durante le fasi successive di elaborazione."""
        self.weight_scale = Parameter(torch.Tensor(heads*node_feats, node_feats)) #weight_scale: pesi per la funzione di scala (heads*node_feats, node_feats)
        self.bias = Parameter(torch.Tensor(node_feats)) #bias: bias aggiunto alla funzione di scala (node_feats)
        self.ln = nn.LayerNorm(node_feats) #applica la normalization alle caratteristiche dei nodi
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters() #meotodo per inizializzare i parametri del modello


    def reset_parameters(self):
        kaiming_uniform_(self.weight_triplet_att) #initialize the input Tensor with values using a Kaiming uniform distribution.
        kaiming_uniform_(self.weight_scale) #initialize the input Tensor with values using a Kaiming uniform distribution.
        zeros_(self.bias) #initialize the input Tensor with zeros.

    """Questa funzione determina i coefficienti di attenzione e e prepara i messaggi da passare tra i nodi in una rete di attenzione grafica multi-testa.
    obbligatoria per una classe DGLGraph per calcolare i messaggi da inviare tra i nodi ed ha sempre come firma message_func(self, edges).
    Alla fine quindi per ogni testa di attenzione avremo un vettore di coefficienti di dimensione num_edge, quindi ogni arco avrà un peso
    Passata alla funzione update_all() di un DGLGraph avviene tutto in automatico"""
    def message_func(self, edges):
        u,v =  edges.src['wv'],edges.dst['wv'] #extract the source and destination node features from the edges
        """view: Returns a new tensor with the same data but different size. It is the same data as the original tensor but with a different shape.
        -1 means the size of the dimension is inferred from the number of elements in the tensor and the remaining dimensions.
        since u and v are taken from the edges which have been modified by the Linear function, they have shape (edge_size,node_feats*head) where node_size is the number of nodes in the graph
        so we reshape them to (edge_size,heads,node_feats) where node_size is the number of nodes in the graph."""
        u = u.view(-1,self.heads,self.node_feats)  # with view we can reshape the tensor to the desired shape (edge_size,heads,node_feats) where node_size is the number of nodes in the graph
        v = v.view(-1,self.heads,self.node_feats) # with view we can reshape the tensor to the desired shape (edge_size,heads,node_feats) where node_size is the number of nodes in the graph
        tmp = torch.cat([u,v],dim=-1) #concatenate the source and destination node features along the last dimension so now we have shape (edge_size,heads,2*node_feats)
        attn = tmp * self.weight_triplet_att #broadcast, only the first dimension change between the two variables
        attn = torch.sum(attn,dim=-1,keepdim=True) #sum the last dimension and keep the dimension so now we have shape (edge_size,heads,1)
        attn = F.leaky_relu(attn,self.negative_slope) #apply the LeakyReLU activation function to the attention scores so now we still have shape (edge_size,heads,1)
        return {'attn': attn,'u': u} #this function through update_all() will send the attention scores and the source node features to the destination nodes

    """Questa funzione calcola la somma pesata delle caratteristiche dei nodi vicini e aggiorna le caratteristiche dei nodi in una rete di attenzione grafica multi-testa.
    obbligatoria per una classe DGLGraph per calcolare i messaggi da inviare tra i nodi ed ha sempre come firma reduce_func(self, nodes).
    Passata ala funzione update_all() di un DGLGraph avviene tutto in automatico"""
    def reduce_func(self, nodes):
        """nodes.mailbox è una struttura di DGL che raccoglie i messaggi da tutti gli edge in ingresso per ogni nodo di destinazione.
        Quando message_func è chiamata, i risultati vengono inviati a nodes.mailbox dei nodi di destinazione quindi
        nodes.mailbox['attn'] ha dimensione (num_nodes, num_incoming_edges, heads, 1)
        nodes.mailbox['u'] ha dimensione (num_nodes, num_incoming_edges, heads, node_feats)
        """
        score = F.softmax(nodes.mailbox['attn'], dim=1) #apply the softmax function to the attention scores -> shape (num_nodes, num_incoming_edges, heads, 1)
        result = torch.sum(score*nodes.mailbox['u'],dim=1)  #sum the source node features weighted by the attention scores along the indoming_edges dimension -> from shape (num_nodes, num_incoming_edges, heads, node_feats) to shape (num_nodes, num_incoming_edges, heads, node_feats) and then sum along the heads dimension so now we have shape (num_nodes, heads, node_feats)
        result = result.view(-1,self.heads*self.node_feats) #reshape the result to the original shape of the node features -> shape (num_nodes,heads*node_feats)
        result = torch.matmul(result, self.weight_scale) #apply the linear transformation to the result -> shape (num_nodes,node_feats)
        return {'h_new': result + self.bias} #return the new node features after the GAT layer -> shape (num_nodes,node_feats)

    """Questa funzione definisce il passaggio in avanti della rete di attenzione grafica multi-testa.
    Viene chiamata quando si passa un grafo e le caratteristiche dei nodi alla rete.
    h rappresenta le caratteristiche dei nodi quindi shape (node_size,node_feats) e graph rappresenta il grafo"""
    def forward(self, graph, h ):
        with graph.local_scope(): # operation not affect the original graph data
            # import pdb;pdb.set_trace()
            wv = self.weight_node(h) #wv shape (node_size,heads*node_feats)
            graph.ndata['wv'] = wv

            tmp = h
            graph.update_all(self.message_func, self.reduce_func) #send and receive the messages between the nodes using message_func and update the node features using the reduce_func
            h = graph.ndata.pop('h_new')

            return self.dropout(F.relu(self.ln(h + tmp))) #apply the dropout, the ReLU activation function and the normalization to the new node features and return them

"""La classe tree_gru_onehot_revised implementa una rete neurale che elabora gli alberi molecolari.
Metodi:
- __init__(self, vocab, hidden_size, head_nums, conv_nums): Inizializza la rete con il vocabolario, la dimensione nascosta, il numero di teste e il numero di convoluzioni.
- forward(self, g): Definisce il passaggio in avanti della rete, applicando la proiezione lineare alle caratteristiche dei nodi, eseguendo le convoluzioni e calcolando il readout finale.
"""
class tree_gru_onehot_revised(nn.Module):
    def __init__(self, vocab,hidden_size, head_nums, conv_nums):
        super(tree_gru_onehot_revised, self).__init__() #call the __init__ method of the parent class nn.Module 

        self.hidden_size = hidden_size #shape of the hidden representationm 
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = nn.Embedding(self.vocab_size, hidden_size) #so we use an encoding layer to encode the node features using a vocabulary with outpu shape (vocab_size,hidden_size)
        self.project = nn.Linear(self.vocab_size + hidden_size, hidden_size) #apply a linear transformation to the node features to project them to the hidden size
        # self.project = nn.Linear(hidden_size,hidden_size)

        self.convs = nn.ModuleList([MultiHead_gat(hidden_size, heads=head_nums) for _ in range(self.conv_nums)])
        # self.dropout = nn.Dropout(0.1)
        self.gru_readout = GRU_ReadOut(self.hidden_size, self.hidden_size, )

    def forward(self, g):
        with g.local_scope(): # operation not affect the original graph data
            device = g.device
            """One-Hot Encoding: Crea una rappresentazione one-hot dei nodi basata sugli ID del vocabolario (wid).
            Utilizza scatter_ per impostare le colonne corrispondenti agli ID dei nodi (wid) a 1."""
            one_hot = torch.zeros(g.number_of_nodes(), self.vocab_size).to(device)
            one_hot.scatter_(dim=1, index=g.ndata['wid'].unsqueeze(dim=1), 
                             src=torch.ones(g.number_of_nodes(), self.vocab_size).to(device)) #scatter the one hot encoding of the node features to the nodes so now we have shape (node_size,vocab_size) to encode the nodes
            #one_hot shape (num_nodes,vocab_size)

            """Concatenazione e Proiezione: Concatena la rappresentazione one-hot con le caratteristiche dei nodi e proietta la combinazione in una nuova rappresentazione nascosta.
            La rappresentazione one-hot viene concatenata con le caratteristiche originali dei nodi (h).
            Viene utilizzato un layer lineare (self.project) per proiettare la combinazione in una nuova rappresentazione nascosta di dimensione hidden_size."""
            g.ndata['h'] = torch.cat([one_hot, g.ndata['h']], dim=-1) #shape (num_nodes,vocab_size+hidden_size)
            h = g.ndata.pop('h') #remove the node features from the graph and store them in h so still h has shape (num_nodes,vocab_size+hidden_size)
            h = self.project(h) #apply the linear transformation to the node features to project them to the hidden size -> shape (num_nodes,hidden_size)
            g.ndata.update({'h': h})

            # print(h.device)
            """Convoluzione Multi-Head GAT: Applica una serie di convoluzioni Multi-Head GAT e calcola la media delle caratteristiche dei nodi.
            Le rappresentazioni nascoste vengono passate attraverso una serie di strati Multi-Head GAT.
            Dopo ogni strato, la media delle caratteristiche dei nodi viene calcolata e aggiunta a una lista (gru_list)."""
            gru_list = []
            gru_list.append(dgl.mean_nodes(g,'h')) #shape (batch_size, hidden_size)
            for convs in self.convs:
                h = convs.forward(g,h) #apply the GAT convolution to the node features so shape (num_nodes,hidden_size)
                g.ndata['h'] = h 
                gru_list.append(dgl.mean_nodes(g,'h')) #calculate the mean of the node features

            """"Aggregazione tramite GRU: Utilizza la GRU per aggregare le rappresentazioni nascoste e calcola la media delle rappresentazioni finali."""
            out,h = self.gru_readout(gru_list) #apply the GRU readout to the node features so shape of out: (batch_size, seq_len, num_directions * hidden_size) h:(num_layers * num_directions, batch_size, hidden_size)
            h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru   #shape (1, batch_size, hidden_size)
            return None, h.squeeze(0)  # reduce dimension so now h's size is (batch_size, hidden_size)

"""La classe GatEncoder_raw_gru_revised implementa un encoder che utilizza l'attenzione grafica per elaborare le caratteristiche grezze delle molecole."""
class GatEncoder_raw_gru_revised(nn.Module):
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(GatEncoder_raw_gru_revised, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        #notice that for row molecule representation we don't need to use the vocab

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.project = nn.Linear(self.in_size, hidden_size)
        self.convs = nn.ModuleList([MultiHead_gat(hidden_size, heads=head_nums) for _ in range(self.conv_nums)])
        # self.dropout = nn.Dropout(0.1)
        self.gru_readout = GRU_ReadOut(self.hidden_size, self.hidden_size, )


    def forward(self, g):
        with g.local_scope():
            h = g.ndata.pop('h') #remove the node features from the graph and store them in h so still h has shape (num_nodes,in_size)
            h = self.project(h) #apply the linear transformation to the node features to project them to the hidden size -> shape (num_nodes,hidden_size)
            g.ndata.update({'h': h})

            gru_list = []
            gru_list.append(dgl.mean_nodes(g,'h'))
            for convs in self.convs: 
                h = convs.forward(g,h) #apply the GAT convolution to the node features so shape (num_nodes,hidden_size)
                g.ndata['h'] = h
                gru_list.append(dgl.mean_nodes(g,'h')) #calculate the mean of the node features so shape (batch_size, hidden_size)

            out,h = self.gru_readout(gru_list) #apply the GRU readout to the node features so shape of out: (batch_size, seq_len, num_directions(so 2) * hidden_size) h:(num_layers(likely 2) * num_directions (so 2), batch_size, hidden_size)
            h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru #shape (1, batch_size, hidden_size)

            from dgl import unbatch
            return [hh.ndata['h'] for hh in unbatch(g)], h.squeeze(0) #reduce dimension so now h's size is (batch_size, hidden_size)


# non usata non serve studiarla
class tree_gru_onehot_only(nn.Module):
    def __init__(self, vocab,hidden_size, head_nums, conv_nums):
        super(tree_gru_onehot_only, self).__init__()

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.project = nn.Linear(self.vocab_size + hidden_size, hidden_size)

        self.convs = nn.ModuleList([MultiHead_gat(hidden_size, heads=head_nums) for _ in range(self.conv_nums)])
        # self.dropout = nn.Dropout(0.1)
        self.gru_readout = GRU_ReadOut(self.hidden_size, self.hidden_size, )

    def forward(self, g):
        with g.local_scope():
            device = g.device
            one_hot = torch.zeros(g.number_of_nodes(), self.vocab_size).to(device)
            one_hot.scatter_(dim=1, index=g.ndata['wid'].unsqueeze(dim=1),
                             src=torch.ones(g.number_of_nodes(), self.vocab_size).to(device))
            g.ndata['h'] = torch.cat([one_hot,self.embedding(g.ndata['wid'])],dim=-1)

            h = g.ndata.pop('h')
            h = self.project(h)
            g.ndata.update({'h': h})

            # print(h.device)

            gru_list = []
            gru_list.append(dgl.mean_nodes(g,'h'))
            for convs in self.convs:
                h = convs.forward(g,h)
                g.ndata['h'] = h
                gru_list.append(dgl.mean_nodes(g,'h'))

            out,h = self.gru_readout(gru_list)
            h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru
            return None, h.squeeze(0)  # reduce dimension

# non usata non serve studiarla
class GatEncoder_raw_gru_only(nn.Module):
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(GatEncoder_raw_gru_only, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.project = nn.Linear(self.in_size, hidden_size)
        self.convs = nn.ModuleList([MultiHead_gat(hidden_size, heads=head_nums) for _ in range(self.conv_nums)])
        # self.dropout = nn.Dropout(0.1)
        self.gru_readout = GRU_ReadOut(self.hidden_size, self.hidden_size, )


    def forward(self, g):
        with g.local_scope():
            h = g.ndata.pop('h')
            h = self.project(h)
            g.ndata.update({'h': h})

            gru_list = []
            gru_list.append(dgl.mean_nodes(g,'h'))
            for convs in self.convs:
                h = convs.forward(g,h)
                g.ndata['h'] = h
                gru_list.append(dgl.mean_nodes(g,'h'))

            out,h = self.gru_readout(gru_list)
            h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru
            return h.squeeze(0)







# GAT conv layer
"""La classe GATLayer implementa un singolo strato di attenzione grafica (Graph Attention Network, GAT). 
Questo strato esegue operazioni di attenzione sui nodi di un grafo, 
calcolando coefficienti di attenzione per gli archi e aggiornando le caratteristiche dei nodi in base a queste attenzioni."""
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)  # nn类下定义的模型都是可训练参数
        self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        # bn
        self.bn = nn.BatchNorm1d(out_feats)

#calcola i coefficienti di attenzione per gli archi del grafo.
    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=1) #z means nodes features but after linear transformation
        src_e = self.attention_func(concat_z)
        src_e = F.leaky_relu(src_e)  # Ottenere i coefficienti di attenzione degli archi
        return {'e': src_e}  # Restituisce il risultato dell'aggiornamento dell'arco

#Definisce i messaggi che vengono inviati lungo gli archi.
    def message_func(self, edges):  # La funzione messaggio viene utilizzata solo dagli archi
        return {'z': edges.src['z'], 'e': edges.data['e']}  # Memorizza il contenuto della mailbox, registra le informazioni ricevute dal punto di destinazione dst e la mailbox classifica la matrice tensoriale tridimensionale in base all'incidenza dei vertici.

    def reduce_func(self, nodes):  # La funzione reduce è utilizzata solo dai nodi
        a = F.softmax(nodes.mailbox['e'], dim=1)  # a è il coefficiente di attenzione elaborato con softmax
        h = torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, h):
        z = self.linear_func(h)
        graph.ndata['z'] = z
        graph.apply_edges(self.edge_attention)  # Apply the function on the edges to update their features.
        graph.update_all(self.message_func, self.reduce_func)  # Combinazione di funzioni di invio e di rev, una comoda API.

        # bn
        temp_h = graph.ndata['h']
        h = self.bn(F.relu(temp_h)) # Update the node features and apply the batch normalization
        graph.ndata.update({'h': h}) # Update the node features

        return graph

class GATLayer_revised(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer_revised, self).__init__()
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        # bn
        self.bn = nn.BatchNorm1d(out_feats)

    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        src_e = self.attention_func(concat_z)
        src_e = F.leaky_relu(src_e)  # 得到边的注意力系数
        return {'e': src_e}  # 返回边的注意力分数,更新边的特征

    def message_func(self, edges):  # message函数只有边使用
        return {'z': edges.src['z'], 'e': edges.data['e']}  # 存入mailbox中的内容,记录目标点dst接受到的信息,mailbox按顶点的入度进行3维tensor矩阵分类

    def reduce_func(self, nodes):  # reduce函数只有节点使用
        a = F.softmax(nodes.mailbox['e'], dim=1)  # a是经softmax处理的注意力系数
        h = torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, h):
        with graph.local_scope(): # operation not affect the original graph data
            z = self.linear_func(h)
            graph.ndata['z'] = z
            graph.apply_edges(self.edge_attention)  # Apply the function on the edges to get the edges attention_score

            graph.update_all(self.message_func, self.reduce_func)  # send和rev函数的组合,方便的api
            h = graph.ndata.pop('h')
            h = self.bn(F.relu(h))

            return h


class GATLayer_adde(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer_adde, self).__init__()
        self.hidden_state = out_feats
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_func_e = nn.Linear(11, out_feats, bias=False)

        self.W_q = nn.Linear(self.hidden_state, self.hidden_state, bias=False)
        self.W_k = nn.Linear(self.hidden_state, self.hidden_state, bias=False)
        self.W_v = nn.Linear(self.hidden_state, self.hidden_state, bias=False)
        # bn
        self.ln = nn.LayerNorm(out_feats)

    def message_func(self, edges):
        x = torch.stack([edges.src['z'], edges.dst['z'], edges.data['e_f']], dim=1)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_score = torch.matmul(Q, torch.transpose(K, 2, 1)).softmax(-1)
        z = torch.matmul(attn_score, V).sum(1)

        return {'m': z}

    def reduce_func(self, nodes):
        h = torch.mean(nodes.mailbox['m'], dim=1)
        return {'h_new': h}

    def forward(self, graph, h):
        with graph.local_scope(): # operation not affect the original graph data
            z = self.linear_func(h)
            e_f = self.linear_func_e(graph.edata['e_f'])
            graph.ndata['z'] = z
            graph.edata['e_f'] = e_f

            graph.update_all(self.message_func, self.reduce_func)  # send和rev函数的组合,方便的api
            h = graph.ndata.pop('h_new')
            h = self.ln(F.relu(h))

            return h


######## 6/26 revise by yexianbin
class self_atten(nn.Module):
    def __init__(self):
        super(self_atten, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, Q, K, V):
        score = torch.matmul(Q, torch.transpose(K, -2, -1))
        attn_score = self.softmax(score)
        context = torch.matmul(attn_score, V)

        return context


class mutip_attention(nn.Module):
    def __init__(self, nhead, hidden_size):
        super(mutip_attention, self).__init__()

        assert hidden_size % nhead == 0, f'hidden_size must be the times of nhead, but got {hidden_size} _ {nhead}'
        self.n_head = nhead
        self.d_x = hidden_size // nhead

        self.W_q = nn.Linear(hidden_size,self.d_x * nhead,bias=False)
        self.W_k = nn.Linear(hidden_size,self.d_x * nhead,bias=False)
        self.W_v = nn.Linear(hidden_size,self.d_x * nhead,bias=False)
        self.fc = nn.Linear(self.d_x * nhead,hidden_size,bias=False)
        self.scaled_dot = self_atten()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self,input_Q, input_K,input_V):
        residual,node_size = input_Q,input_Q.shape[0]

        Q = self.W_q(input_Q).view(node_size,self.n_head,-1)
        K = self.W_k(input_K).view(node_size,self.n_head,-1)
        V = self.W_v(input_V).view(node_size,self.n_head,-1)

        context = self.scaled_dot(Q,K,V)
        context = context.reshape(node_size,self.n_head * self.d_x)

        output = self.fc(context)
        return self.layernorm(output + residual)


class raw_attention(nn.Module):  # V1.0.1
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(raw_attention, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        for i in range(self.head_nums): # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.in_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))

        for j in range(self.conv_nums-1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))


        self.muti_attn = mutip_attention(nhead=4,hidden_size=hidden_size)

    def forward(self, g):
        with g.local_scope():
            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.cat(output,dim=1)
                h = self.__getattr__("out{}".format(j))(h)
                g.ndata.update({'h':h})

            gs = dgl.unbatch(g)
            res = torch.zeros(len(gs),self.hidden_size).cuda()
            for i,gg in enumerate(gs):
                hh = gg.ndata.pop('h')
                hnew = self.muti_attn(hh,hh,hh)
                res[i] = hnew.mean(0)

            return res

################## 6/26 revise by yexianbin


################################# gru readout
class GRU_ReadOut(nn.Module):
    def __init__(self,in_feats,hidden_feats,dropout = 0.0):
        super(GRU_ReadOut,self).__init__()
        self.in_feats = in_feats
        self.gru = nn.GRU(in_feats, hidden_feats,batch_first=True,dropout=dropout,num_layers=2,bidirectional=True) #(batch_size,seq_len,in_features)
        self.linear_project = nn.Linear(init_feats_size,hidden_feats)

    def forward(self,x):
        if x[0].shape[-1] != self.in_feats:
            x.append(self.linear_project(x[0]))
            x = x[-1:] + x[1:-1]

        batch_size,feats = x[0].shape
        tmp_list = [it.view(batch_size,-1,feats) for it in x] # (batch_size,feats) -> (batch_size,1,feats)
        x = torch.cat(tmp_list,dim=1) # # (batch_size,1,feats) -> (batch_size,num_conv,feats)
        return self.gru(x)


# GAT model by gru readout
class GatEnconder_tree_gru(nn.Module):  # Graph conv for tree_mol V1.0
    def __init__(self, vocab, hidden_size, embedding=None):
        super(GatEnconder_tree_gru, self).__init__()  # 调用父类的构造器

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.GATcov1 = GATLayer(hidden_size, hidden_size)
        self.GATcov2 = GATLayer(hidden_size, hidden_size)
        self.GATcov3 = GATLayer(hidden_size, hidden_size)
        self.GATcov4 = GATLayer(hidden_size, hidden_size)

        self.GAT = nn.ModuleList([self.GATcov1, self.GATcov2, self.GATcov3, self.GATcov4])
        self.out = nn.Linear(4 * hidden_size, hidden_size)

        self.gru_readout = GRU_ReadOut(self.hidden_size, self.hidden_size,dropout=0.0)

    def forward(self, mol_tree_batch):
        mol_tree_batch.ndata.update({
            'h': self.embedding(mol_tree_batch.ndata['wid']),  # 'wid'是官能团在vocab的索引，这个得到的是节点对应的词向量矩阵
        })

        gru_list = []
        gru_list.append(dgl.mean_nodes(mol_tree_batch, 'h'))

        output = []
        for gat in self.GAT:
            temp_h = torch.clone(mol_tree_batch.ndata['h'])
            mol_tree_batch = gat(mol_tree_batch, mol_tree_batch.ndata['h'])
            output.append(mol_tree_batch.ndata['h'])
            mol_tree_batch.ndata.update({'h': temp_h})

        h = torch.cat(output, dim=1)
        mol_tree_batch.ndata.update({'h': h})
        new_h = self.out(mol_tree_batch.ndata['h'])
        mol_tree_batch.ndata.update({'h': new_h})
        gru_list.append(dgl.mean_nodes(mol_tree_batch, 'h'))

        out, h = self.gru_readout(gru_list)

        h = torch.sum(h, dim=0, keepdim=True)  # add 2 layer gru
        return mol_tree_batch, h.squeeze(0)  # 返回readout方法，代表批中每个小图的表示向量

        # return mol_tree_batch, out[:,-1,:]


class GatEncoder_raw_gru(nn.Module):  # V1.0.1
    def __init__(self, hidden_size, head_nums, conv_nums, input_size=None):
        super(GatEncoder_raw_gru, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        for i in range(self.head_nums):  # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.in_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))

        for j in range(self.conv_nums - 1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))

        self.gru_readout = GRU_ReadOut(self.hidden_size, self.hidden_size, )

    def forward(self, g):
        with g.local_scope():
            gru_list = []
            gru_list.append(dgl.mean_nodes(g, 'h'))
            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j, i))(g, h))
                h = torch.cat(output, dim=1)
                h = self.__getattr__("out{}".format(j))(h)
                g.ndata.update({'h': h})
                gru_list.append(dgl.mean_nodes(g, 'h'))

            out, h = self.gru_readout(gru_list)

            h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru
            return h.squeeze(0)  # reduce dimension


# 2021/12/16 add set2set readout
from dgl.nn import Set2Set
class GatEncoder_raw_gru_s2s(nn.Module):
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(GatEncoder_raw_gru_s2s, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.project = nn.Linear(self.in_size, hidden_size)
        for i in range(self.head_nums): # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(hidden_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))
        self.__setattr__('LN0', nn.LayerNorm(hidden_size))

        for j in range(self.conv_nums-1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))
            self.__setattr__('LN{}'.format(j + 1), nn.LayerNorm(hidden_size))

        self.gru_readout = GRU_ReadOut(self.hidden_size,self.hidden_size,)

        self.s2s = Set2Set(hidden_size, 2, 1)  # set2set readout
        self.linear = nn.Linear(hidden_size*2,hidden_size)



    def forward(self, g):
        with g.local_scope():
            h = g.ndata.pop('h')
            h = self.project(h)
            g.ndata.update({'h': h})

            gru_list = []
            # gru_list.append(self.linear(self.s2s(g, g.ndata['h'])))
            gru_list.append(dgl.mean_nodes(g, 'h'))
            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                tmp = h
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.cat(output,dim=1)
                h = self.__getattr__('LN{}'.format(j))(self.__getattr__("out{}".format(j))(h) + tmp)

                g.ndata.update({'h':h})
                gru_list.append(dgl.mean_nodes(g, 'h'))
                # gru_list.append(self.linear(self.s2s(g, g.ndata['h'])))

            out,h = self.gru_readout(gru_list)

            h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru
            return h.squeeze(0)  # reduce dimension


class tree_gru_onehot_s2s(nn.Module):
    def __init__(self, vocab,hidden_size, head_nums, conv_nums):
        super(tree_gru_onehot_s2s, self).__init__()

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)

        self.project = nn.Linear(self.vocab_size + hidden_size, hidden_size)
        # self.project = nn.Linear(hidden_size, hidden_size)

        for i in range(self.head_nums):  # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(hidden_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))
        self.__setattr__('LN0',nn.LayerNorm(hidden_size))

        for j in range(self.conv_nums - 1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))
            self.__setattr__('LN{}'.format(j+1), nn.LayerNorm(hidden_size))

        self.gru_readout = GRU_ReadOut(self.hidden_size, self.hidden_size, )
        self.s2s = Set2Set(hidden_size, 2, 1)  # set2set readout
        self.linear = nn.Linear(hidden_size*2,hidden_size)


    def forward(self, g):
        with g.local_scope():
            # one_hot = torch.zeros(g.number_of_nodes(), self.vocab_size).cuda()
            # one_hot.scatter_(dim=1, index=g.ndata['wid'].unsqueeze(dim=1),
            #                  src=torch.ones(g.number_of_nodes(), self.vocab_size).cuda())
            # g.ndata['h'] = torch.cat([one_hot,self.embedding(g.ndata['wid'])],dim=-1)

            one_hot = torch.zeros(g.number_of_nodes(), self.vocab_size)
            one_hot.scatter_(dim=1, index=g.ndata['wid'].unsqueeze(dim=1),
                             src=torch.ones(g.number_of_nodes(), self.vocab_size))
            g.ndata['h'] = torch.cat([one_hot,self.embedding(g.ndata['wid'])],dim=-1)

            # g.ndata['h'] = self.embedding(g.ndata['wid'])
            h = g.ndata.pop('h')
            h = self.project(h)
            g.ndata.update({'h': h})

            # gru_list = []
            # gru_list.append(self.linear(self.s2s(g, g.ndata['h'])))

            gru_list = []
            gru_list.append(dgl.mean_nodes(g,'h'))

            for j in range(self.conv_nums):
                output = []
                tmp = h
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.cat(output,dim=1)
                h = self.__getattr__('LN{}'.format(j))(self.__getattr__("out{}".format(j))(h) + tmp)
                g.ndata.update({'h':h})
                gru_list.append(dgl.mean_nodes(g, 'h'))
                # gru_list.append(self.linear(self.s2s(g, g.ndata['h'])))

            out,h = self.gru_readout(gru_list)
            h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru
            return None, h.squeeze(0)  # reduce dimension



class tree_gru(nn.Module):  # V1.0.1
    def __init__(self, vocab,hidden_size, head_nums, conv_nums):
        super(tree_gru, self).__init__()

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)


        for j in range(self.conv_nums):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j), nn.Linear(head_nums * hidden_size, hidden_size))

        self.gru_readout = GRU_ReadOut(self.hidden_size,self.hidden_size,)

    def forward(self, g):
        with g.local_scope():
            g.ndata.update({
                'h': self.embedding(g.ndata['wid']),  # 'wid'是官能团在vocab的索引，这个得到的是节点对应的词向量矩阵
            })

            gru_list = []
            gru_list.append(dgl.mean_nodes(g,'h'))
            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.cat(output,dim=1)
                h = self.__getattr__("out{}".format(j))(h)
                g.ndata.update({'h':h})
                gru_list.append(dgl.mean_nodes(g,'h'))

            out,h = self.gru_readout(gru_list)

            h = torch.sum(h, dim=0, keepdim=True)  # add 2 layer gru
            return None, h.squeeze(0)  # reduce dimension

# 2021/6/22 revised
class tree_gru_onehot(nn.Module):  # V1.0.1
    def __init__(self, vocab,hidden_size, head_nums, conv_nums):
        super(tree_gru_onehot, self).__init__()

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)

        for i in range(self.head_nums):  # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.vocab_size + hidden_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))

        for j in range(self.conv_nums - 1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))

    def forward(self, g):
        with g.local_scope():
            one_hot = torch.zeros(g.number_of_nodes(), self.vocab_size).cuda()
            one_hot.scatter_(dim=1, index=g.ndata['wid'].unsqueeze(dim=1),
                             src=torch.ones(g.number_of_nodes(), self.vocab_size).cuda())

            g.ndata['h'] = torch.cat([one_hot,self.embedding(g.ndata['wid'])],dim=-1)

            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.cat(output,dim=1)
                h = self.__getattr__("out{}".format(j))(h)
                g.ndata.update({'h':h})

            # out,h = self.gru_readout(gru_list)

            # h = torch.sum(h, dim=0, keepdim=True)  # add 2 layer gru
            # return None, h.squeeze(0)  # reduce dimension

            return None, dgl.mean_nodes(g,'h')


# 2021/6/26
class tree_gru_s2s(nn.Module):  # V1.0.1
    def __init__(self, vocab,hidden_size, head_nums, conv_nums):
        super(tree_gru_s2s, self).__init__()

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)

        for i in range(self.head_nums):  # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.vocab_size + hidden_size, hidden_size))
        self.__setattr__('out0', nn.Linear(hidden_size, hidden_size))

        for j in range(self.conv_nums - 1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(hidden_size, hidden_size))

    def forward(self, g):
        with g.local_scope():
            one_hot = torch.zeros(g.number_of_nodes(), self.vocab_size).cuda()
            one_hot.scatter_(dim=1, index=g.ndata['wid'].unsqueeze(dim=1),
                             src=torch.ones(g.number_of_nodes(), self.vocab_size).cuda())

            g.ndata['h'] = torch.cat([one_hot,self.embedding(g.ndata['wid'])],dim=-1)

            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.stack(output,dim=1)
                h = self.__getattr__("out{}".format(j))(h.mean(1))
                g.ndata.update({'h':h})

            # out,h = self.gru_readout(gru_list)

            # h = torch.sum(h, dim=0, keepdim=True)  # add 2 layer gru
            # return None, h.squeeze(0)  # reduce dimension

            return None, dgl.max_nodes(g,'h')


class raw_gru_s2s(nn.Module):  # V1.0.1
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(raw_gru_s2s, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums


        for i in range(self.head_nums): # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.in_size, hidden_size))
        self.__setattr__('out0', nn.Linear(hidden_size, 1))

        for j in range(self.conv_nums-1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(hidden_size, 1))

    def forward(self, g):
        with g.local_scope():
            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.stack(output,dim=1) # node,chanel,hidden
                score = self.__getattr__("out{}".format(j))(h).softmax(1) # node,chanel,1
                h = torch.matmul(torch.transpose(score,2,1),h).squeeze(1)
                g.ndata.update({'h':h})

            return dgl.mean_nodes(g,'h')




# 2021/6/26 revised
from dgl.nn import Set2Set
class raw_set2set(nn.Module):
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(raw_set2set, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        for i in range(self.head_nums): # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.in_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))

        for j in range(self.conv_nums-1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))

        self.s2s = Set2Set(hidden_size, 2, 1)  # set2set readout
        self.linear = nn.Linear(hidden_size*2,hidden_size)

        # self.s2s = Set2Set(hidden_size, n_iters=3,n_layers=1)
        # self.linear = nn.Sequential(
        #     nn.Linear(2 * hidden_size, 300),
        #     nn.LayerNorm(300),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.02),
        #     nn.Linear(300, hidden_size)
        # )

    def forward(self, g):
        with g.local_scope():
            h = g.ndata.pop('h')
            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.cat(output,dim=1)
                h = self.__getattr__("out{}".format(j))(h)
                g.ndata.update({'h':h})

            return self.linear(self.s2s(g,g.ndata['h']))



## self_attention add_edge features 2021/6/10
class raw_gru_adde(nn.Module):  # V1.0.1
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(raw_gru_adde, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        for i in range(self.head_nums): # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_adde(self.in_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))

        for j in range(self.conv_nums-1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_adde(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))

        self.gru_readout = GRU_ReadOut(self.hidden_size,self.hidden_size,)

    def forward(self, g):
        with g.local_scope():
            gru_list = []
            gru_list.append(dgl.mean_nodes(g,'h'))
            h = g.ndata.pop('h')

            for j in range(self.conv_nums):
                output = []
                for i in range(self.head_nums):
                    output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
                h = torch.cat(output,dim=1)
                h = self.__getattr__("out{}".format(j))(h)
                g.ndata.update({'h':h})
                gru_list.append(dgl.mean_nodes(g,'h'))

            out,h = self.gru_readout(gru_list)

            h = torch.sum(h, dim=0, keepdim=True)  # add 2 layer gru
            return h.squeeze(0)  # reduce dimension

################################# gru readout



# add by yexianbin 2021/3/28.  for test
# lstm readout
class lstm_ReadOut(nn.Module):
    def __init__(self,in_feats,hidden_feats,dropout = 0.0):
        super(lstm_ReadOut,self).__init__()
        self.in_feats = in_feats
        self.gru = nn.LSTM(in_feats, hidden_feats,batch_first=True,dropout=dropout,num_layers=2) #(batch_size,seq_len,in_features)
        self.linear_project = nn.Linear(init_feats_size,hidden_feats)

    def forward(self,x):
        if x[0].shape[-1] != self.in_feats:
            x.append(self.linear_project(x[0]))
            x = x[-1:] + x[1:-1]

        batch_size,feats = x[0].shape
        tmp_list = [it.view(batch_size,-1,feats) for it in x] # (batch_size,feats) -> (batch_size,1,feats)
        x = torch.cat(tmp_list,dim=1) # # (batch_size,1,feats) -> (batch_size,num_conv,feats)
        return self.gru(x)

class GatEnconder_tree_lstm(nn.Module):  # Graph conv for tree_mol V1.0
    def __init__(self, vocab, hidden_size, embedding=None):
        super(GatEnconder_tree_lstm, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.GATcov1 = GATLayer(hidden_size, hidden_size)
        self.GATcov2 = GATLayer(hidden_size, hidden_size)
        self.GATcov3 = GATLayer(hidden_size, hidden_size)
        self.GATcov4 = GATLayer(hidden_size, hidden_size)

        self.GAT = [self.GATcov1, self.GATcov2, self.GATcov3, self.GATcov4]
        self.out = GATLayer(4 * hidden_size, hidden_size)

        self.lstm_readout = lstm_ReadOut(self.hidden_size, self.hidden_size, dropout=0.0)

    def forward(self, mol_trees):
        mol_tree_batch = batch(mol_trees)
        return self.run(mol_tree_batch)

    def run(self, mol_tree_batch):
        mol_tree_batch.ndata.update({
            'h': self.embedding(mol_tree_batch.ndata['wid']),  # 'wid'是官能团在vocab的索引，这个得到的是该节点对应的词向量
        })

        gru_list = []
        gru_list.append(dgl.mean_nodes(mol_tree_batch,'h'))

        output = []
        for gat in self.GAT:
            temp_h = torch.clone(mol_tree_batch.ndata['h'])
            mol_tree_batch = gat(mol_tree_batch, mol_tree_batch.ndata['h'])
            output.append(mol_tree_batch.ndata['h'])
            mol_tree_batch.ndata.update({'h': temp_h})

        h = torch.cat(output, dim=1)
        mol_tree_batch.ndata.update({'h': h})
        mol_tree_batch = self.out(mol_tree_batch, mol_tree_batch.ndata['h'])
        gru_list.append(dgl.mean_nodes(mol_tree_batch,'h'))

        out,h = self.lstm_readout(gru_list)
        h = torch.sum(h[0],dim=0,keepdim=True)

        return mol_tree_batch, h.squeeze(0)  # 返回readout方法，代表批中每个小图的表示向量

class GatEncoder_raw_revised_lstm(nn.Module):  # V1.0.1
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(GatEncoder_raw_revised_lstm, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        for i in range(self.head_nums): # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.in_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))

        for j in range(self.conv_nums-1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))

        self.bn_out = nn.BatchNorm1d(hidden_size)

        self.lstm_readout = lstm_ReadOut(self.hidden_size,self.hidden_size,)

    def forward(self, mol_raws):
        mol_raws_batch = batch(mol_raws)
        h = mol_raws_batch.ndata['h']
        return self.run(mol_raws_batch, h)

    def run(self, g, h):
        gru_list = []
        gru_list.append(dgl.mean_nodes(g,'h'))
        h = g.ndata.pop('h')

        for j in range(self.conv_nums):
            output = []
            for i in range(self.head_nums):
                output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
            h = torch.cat(output,dim=1)
            h = self.__getattr__("out{}".format(j))(h)
            g.ndata.update({'h':h})
            gru_list.append(dgl.mean_nodes(g,'h'))

        out,h = self.lstm_readout(gru_list)
        h = torch.sum(h[0],dim=0,keepdim=True)

        return h.squeeze(0)  # reduce dimension


# add by yexianbin 2021/3/28 for test
# concat + FC
class GatEnconder_tree_concat(nn.Module):  # Graph conv for tree_mol V1.0
    def __init__(self, vocab, hidden_size, embedding=None):
        super(GatEnconder_tree_concat, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.GATcov1 = GATLayer(hidden_size, hidden_size)
        self.GATcov2 = GATLayer(hidden_size, hidden_size)
        self.GATcov3 = GATLayer(hidden_size, hidden_size)
        self.GATcov4 = GATLayer(hidden_size, hidden_size)

        self.GAT = [self.GATcov1, self.GATcov2, self.GATcov3, self.GATcov4]
        self.out = GATLayer(4 * hidden_size, hidden_size)

        self.project = nn.Linear(2*hidden_size,hidden_size)

    def forward(self, mol_trees):
        mol_tree_batch = batch(mol_trees)
        return self.run(mol_tree_batch)

    def run(self, mol_tree_batch):
        mol_tree_batch.ndata.update({
            'h': self.embedding(mol_tree_batch.ndata['wid']),  # 'wid'是官能团在vocab的索引，这个得到的是该节点对应的词向量
        })

        gru_list = []
        gru_list.append(dgl.mean_nodes(mol_tree_batch,'h'))

        output = []
        for gat in self.GAT:
            temp_h = torch.clone(mol_tree_batch.ndata['h'])
            mol_tree_batch = gat(mol_tree_batch, mol_tree_batch.ndata['h'])
            output.append(mol_tree_batch.ndata['h'])
            mol_tree_batch.ndata.update({'h': temp_h})

        h = torch.cat(output, dim=1)
        mol_tree_batch.ndata.update({'h': h})
        mol_tree_batch = self.out(mol_tree_batch, mol_tree_batch.ndata['h'])
        gru_list.append(dgl.mean_nodes(mol_tree_batch,'h'))


        h = torch.cat(gru_list,dim=-1)
        h = self.project(h)

        return mol_tree_batch, h # 返回readout方法，代表批中每个小图的表示向量

class GatEncoder_raw_revised_concat(nn.Module):  # V1.0.1
    def __init__(self, hidden_size, head_nums, conv_nums, input_size = None):
        super(GatEncoder_raw_revised_concat, self).__init__()

        if input_size is None:
            self.in_size = init_feats_size
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        for i in range(self.head_nums): # first conv for input feature size
            self.__setattr__("GATconv0_{}".format(i), GATLayer_revised(self.in_size, hidden_size))
        self.__setattr__('out0', nn.Linear(head_nums * hidden_size, hidden_size))

        for j in range(self.conv_nums-1):
            for i in range(self.head_nums):
                self.__setattr__("GATconv{}_{}".format(j + 1, i), GATLayer_revised(hidden_size, hidden_size))
            self.__setattr__("out{}".format(j + 1), nn.Linear(head_nums * hidden_size, hidden_size))

        self.bn_out = nn.BatchNorm1d(hidden_size)

        self.project = nn.Linear(init_feats_size+conv_nums*hidden_size,hidden_size)

    def forward(self, mol_raws):
        mol_raws_batch = batch(mol_raws)
        h = mol_raws_batch.ndata['h']
        return self.run(mol_raws_batch, h)

    def run(self, g, h):
        gru_list = []
        gru_list.append(dgl.mean_nodes(g,'h'))
        h = g.ndata.pop('h')

        for j in range(self.conv_nums):
            output = []
            for i in range(self.head_nums):
                output.append(self.__getattr__("GATconv{}_{}".format(j,i))(g,h))
            h = torch.cat(output,dim=1)
            h = self.__getattr__("out{}".format(j))(h)
            g.ndata.update({'h':h})
            gru_list.append(dgl.mean_nodes(g,'h'))

        h = torch.cat(gru_list,dim=-1)
        h = self.project(h)

        return h  # reduce dimension


# add 2021/6/10
from dgl.nn import GINConv
class GIN_tree(nn.Module):
    def __init__(self, vocab, hidden_size, embedding=None):
        super(GIN_tree, self).__init__()  # 调用父类的构造器

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.linear1 = torch.nn.Linear(hidden_size,hidden_size)
        self.conv1 = GINConv(self.linear1,'max')
        # self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        # self.conv2 = GINConv(self.linear2,'max')

    def forward(self, mol_trees):
        mol_trees_batch = batch(mol_trees)
        return self.run(mol_trees_batch)

    def run(self,g):
        with g.local_scope():  # operation not affect the original graph data
            g.ndata.update({'h': self.embedding(g.ndata['wid'])})
            res = self.conv1(g,g.ndata.pop('h'))
            # res = self.conv2(g,res)
            g.ndata.update({'res':res})
            return None,dgl.max_nodes(g,'res')


class GIN_raw(nn.Module):
    def __init__(self, hidden_size, head_nums, conv_nums, input_size=None):
        super(GIN_raw, self).__init__()

        if input_size is None:
            self.in_size = 44  # our feature
        else:
            self.in_size = input_size

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.project = nn.Linear(self.in_size,self.hidden_size)
        self.linear = nn.Linear(self.hidden_size,self.hidden_size)

        for j in range(self.conv_nums):
            self.__setattr__("GATconv{}".format(j + 1), GINConv(self.linear, 'mean'))

        self.out = nn.Linear(self.hidden_size*conv_nums,self.hidden_size)

    def forward(self, mol_raws):
        h = mol_raws.ndata.pop('h')
        return self.run(mol_raws, h)

    def run(self, g, h):
        h = self.project(h)
        with g.local_scope():
            output = []
            for j in range(self.conv_nums):
                h = self.__getattr__("GATconv{}".format(j+1))(g, h)
                output.append(h)

            res = torch.cat(output,dim=1)
            res = self.out(res)
            g.ndata.update({'res':res})
            return dgl.mean_nodes(g,'res')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.BCEWithLogitsLoss(reduce=False)(inputs, targets)
        else:
            BCE_loss = nn.BCELoss(reduce=False)(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class MLP_revised(nn.Module):
    def __init__(self,n_feature,n_hidden:list,n_output = 2,dropout=0.2):
        super(MLP_revised,self).__init__()

        assert isinstance(n_hidden,list),"n_hidden param must be the list"

        self.num_hidden_layers = len(n_hidden)
        self.layers = [n_feature] + n_hidden  # the layer list of NN except the output layer

        self.predict = nn.Sequential()
        self.predict.add_module('dropout_input',nn.Dropout(dropout)) # input_layers: dropout first
        for idx,(in_,out_) in enumerate(zip(self.layers[:-1],self.layers[1:])):
            self.predict.add_module('linear{}'.format(idx),nn.Linear(in_,out_))  # add_module(dict): key -> nn.Module
            # self.predict.add_module('relu{}'.format(idx), nn.ReLU())
            self.predict.add_module('bn{}'.format(idx),nn.BatchNorm1d(out_))
            # self.predict.add_module('ln{}'.format(idx), nn.LayerNorm(out_))
            self.predict.add_module('relu{}'.format(idx), nn.ReLU())
            self.predict.add_module('dropout'.format(idx),nn.Dropout(dropout))

        self.predict.add_module('output',nn.Linear(self.layers[-1],n_output))

    def forward(self,x):
        return self.predict(x)



class MLP_residual(nn.Module):
    def __init__(self,n_feature,n_hidden:list,n_output = 2,dropout=0.2):
        super(MLP_residual,self).__init__()

        assert isinstance(n_hidden,list),"n_hidden param must be the list"

        self.num_hidden_layers = len(n_hidden)
        self.layers = [n_feature] + n_hidden  # the layer list of NN except the output layer

        self.block1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.block3 = nn.Sequential()

        all = list(zip(self.layers[:-1], self.layers[1:]))

        idx = 0
        in_,out_ = all[idx]
        self.block1.add_module('linear{}'.format(idx),nn.Linear(in_,out_))  # add_module(dict): key -> nn.Module
        self.block1.add_module('relu{}'.format(idx),nn.ReLU())
        self.block1.add_module('bn{}'.format(idx),nn.BatchNorm1d(out_))
        self.block1.add_module('dropout'.format(idx),nn.Dropout(dropout))

        idx = 1
        in_,out_ = all[idx]
        self.block2.add_module('linear{}'.format(idx),nn.Linear(in_,out_))  # add_module(dict): key -> nn.Module
        self.block2.add_module('relu{}'.format(idx),nn.ReLU())
        self.block2.add_module('bn{}'.format(idx),nn.BatchNorm1d(out_))
        self.block2.add_module('dropout'.format(idx),nn.Dropout(dropout))

        idx = 2
        in_,out_ = all[idx]
        self.block3.add_module('linear{}'.format(idx),nn.Linear(in_,out_))  # add_module(dict): key -> nn.Module
        self.block3.add_module('relu{}'.format(idx),nn.ReLU())
        self.block3.add_module('bn{}'.format(idx),nn.BatchNorm1d(out_))
        self.block3.add_module('dropout'.format(idx),nn.Dropout(dropout))


        self.final = nn.Linear(self.layers[-1], n_output)

    def forward(self,x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)

        return  self.final(x)



class Residual(nn.Module):  # test
    def __init__(self,in_feats,fc1_feats,fcw_feats,fc2_feats,out_feats,n_class=2,dropout = 0.1):
        super(Residual,self).__init__()

        self.BN_in = nn.BatchNorm1d(in_feats)
        self.FC_1 = nn.Linear(in_feats,fc1_feats)
        self.FC_2 = nn.Linear(fc1_feats,fc2_feats)
        self.FC_wide = nn.Linear(in_feats,fcw_feats)
        self.BN_fc1 = nn.BatchNorm1d(fc1_feats)
        # self.BN_wide = nn.BatchNorm1d(fcw_feats)
        # self.out = nn.Linear(fc2_feats + fcw_feats,out_feats)
        self.out = nn.Linear(fc2_feats, out_feats)
        self.BN_out = nn.BatchNorm1d(out_feats)
        self.predict = nn.Linear(out_feats,n_class)

    def forward(self,x):
        x = self.BN_in(x)
        x1 = F.relu(self.BN_fc1(self.FC_1(x)))
        x_w = self.FC_wide(x)
        x2 = self.FC_2(x1)
        # x_concated = torch.cat([x2,x_w],dim=1)
        x_concated = x2 + x_w
        x_out = F.relu(self.BN_out(self.out(x_concated)))
        return self.predict(x_out)


# TrimNet Test 2021/12/17
from torch.nn.init import kaiming_uniform_, zeros_
from torch.nn import Parameter
class MultiHead(nn.Module):
    def __init__(self, node_feats, e_feats, heads=3, negative_slope=0.2):
        super(MultiHead, self).__init__()
        self.node_feats = node_feats
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_node = nn.Linear(node_feats, node_feats*heads,bias=False)
        self.weight_edge = nn.Linear(e_feats, node_feats*heads,bias=False)
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3*node_feats))
        self.weight_scale = Parameter(torch.Tensor(heads*node_feats, node_feats))
        self.bias = Parameter(torch.Tensor(node_feats))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def message_func(self, edges):
        u,v,e =  edges.src['wv'],edges.dst['wv'],edges.data['we']
        u = u.view(-1,self.heads,self.node_feats)  # node_size, head_nums, dims
        v = v.view(-1,self.heads,self.node_feats)
        e = e.view(-1,self.heads,self.node_feats)
        tmp = torch.cat([u,e,v],dim=-1)
        attn = tmp * self.weight_triplet_att
        attn = torch.sum(attn,dim=-1,keepdim=True)
        attn = F.leaky_relu(attn,self.negative_slope)
        return {'attn': attn,'e':e,'u':u}

    def reduce_func(self, nodes):
        score = F.softmax(nodes.mailbox['attn'], dim=1)
        result = torch.sum(score*nodes.mailbox['u']*nodes.mailbox['e'],dim=1)
        result = result.view(-1,self.heads*self.node_feats)
        result = torch.matmul(result, self.weight_scale)
        return {'h_new': result + self.bias}

    def forward(self, graph, h, e):
        with graph.local_scope(): # operation not affect the original graph data
            wv = self.weight_node(h)
            we = self.weight_edge(e)
            graph.ndata['wv'] = wv
            graph.edata['we'] = we

            graph.update_all(self.message_func, self.reduce_func)  # send和rev函数的组合,方便的api
            h = graph.ndata.pop('h_new')

            return h


class Block(torch.nn.Module):
    def __init__(self,dim, edge_dim, heads=4, time_step=3):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = MultiHead(dim,edge_dim,heads)
        self.gru = nn.GRU(dim,dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self,g,h,eh):
        with g.local_scope():
            x = h.unsqueeze(0)
            for i in range(self.time_step):
                m = F.celu(self.conv.forward(g,h,eh))
                h, x = self.gru(m.unsqueeze(0),x)
                h = self.ln(h.squeeze(0))
            return h


class TrimNet(torch.nn.Module):
    def __init__(self, hidden_dim=32, depth=3, heads=4, dropout=0.1, outdim=2):
        super(TrimNet, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.project = nn.Linear(44, hidden_dim)
        self.project_e = nn.Linear(13, hidden_dim)
        self.convs = nn.ModuleList([
            Block(hidden_dim, hidden_dim, heads)
            for i in range(depth)
        ])
        self.set2set = Set2Set(hidden_dim, 2, 1)

        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, hidden_dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, g):
        with g.local_scope():
            x = h = F.celu(self.project(g.ndata['h']))
            e = F.celu(self.project_e(g.edata['e_f']))
            for conv in self.convs:
                x = x + self.dropout(conv(g, x, e))
            g.ndata['x'] = x
            x = self.dropout(self.set2set(g, g.ndata['x']))
            x = self.out(x)
            return x


class tri_gat(nn.Module):
    def __init__(self, hidden_size, head_nums=4, conv_nums=3, input_size = None):
        super(tri_gat, self).__init__()

        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.conv_nums = conv_nums

        self.project = nn.Linear(44, hidden_size,bias=False)
        self.project_e = nn.Linear(13,hidden_size,bias=False)

        self.convs = nn.ModuleList([MultiHead(hidden_size, hidden_size, heads=head_nums) for _ in range(conv_nums)])

        self.gru_readout = GRU_ReadOut(self.hidden_size,self.hidden_size,)

        self.s2s = Set2Set(hidden_size, 2, 1)  # set2set readout
        self.linear = nn.Linear(hidden_size*2,hidden_size)
        self.dropout = nn.Dropout(0.1)


    def forward(self, g):
        with g.local_scope():
            h = g.ndata.pop('h')
            e = g.edata.pop('e_f')
            h = self.project(h)
            e = self.project_e(e)
            g.ndata.update({'h': h})
            g.edata.update({'ef': e})

            # gru_list = []
            # gru_list.append(self.linear(self.s2s(g, g.ndata['h'])))
            h = g.ndata.pop('h')
            e = g.edata.pop('ef')
            for conv in self.convs:
                h = h + self.dropout(conv(g,h,e))
                # gru_list.append(self.linear(self.s2s(g, g.ndata['h'])))

            # out,h = self.gru_readout(gru_list)
            # h = torch.mean(h, dim=0, keepdim=True)  # add 2 layer gru
            # return h.squeeze(0)  # reduce dimension

            g.ndata['h'] = h
            return self.linear(self.s2s(g,g.ndata['h']))




if __name__ == '__main__':
    test = MultiHead(2,2,3)
    from dgl import DGLGraph
    g = DGLGraph()
    g.add_nodes(4)
    g.add_edges(u=[0,1,2],v=[3,3,3])
    print(g)
    h = torch.randn(size=(4,2))
    e = torch.randn(size=(3,2))
    print(test(g,h,e))