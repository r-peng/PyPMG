#from anytree import Node,RenderTree
import networkx as nx
import matplotlib.pyplot as plt
def get_all_pmg(hop_ls,ctr_ls,iprint=0):
    G = nx.Graph()
    G.add_node('root')
    G.nodes['root']['hop_ls_left'] = hop_ls
    G.nodes['root']['ctr_ls_left'] = ctr_ls
    G.nodes['root']['ctr_ls'] = [] 

    def get_all_children(parent):
        hop_ls_left = G.nodes[parent]['hop_ls_left']
        ctr_ls_left = G.nodes[parent]['ctr_ls_left']
        ctr_ls = G.nodes[parent]['ctr_ls']
        children = []
        for ctr in ctr_ls_left:
            ctr_ls_ = ctr_ls + [ctr]
            ctr_ls_.sort()
            name = tuple(ctr_ls_)
            ctr_ls_left_ = ctr_ls_left.copy()
            ctr_ls_left_.remove(ctr)
            hop_ls_left_ = []
            for (p,q) in hop_ls_left:
                if p==ctr:
                    continue
                if q==ctr:
                    continue
                hop_ls_left_.append((p,q))
            if len(hop_ls_left_)==0:
                continue

            if G.has_node(name):
                n = G.nodes[name]
                assert ctr_ls_left_==n['ctr_ls_left']
                assert set(hop_ls_left_)==set(n['hop_ls_left'])
            else:
                G.add_node(name)
                G.nodes[name]['hop_ls_left'] = hop_ls_left_
                G.nodes[name]['ctr_ls_left'] = ctr_ls_left_
                G.nodes[name]['ctr_ls'] = ctr_ls_
                children.append(name)
            if not G.has_edge(parent,name):
                G.add_edge(parent,name)
        return children
    parents = ['root']
    nix = 0
    layers = [parents]
    while True:
        children = [] 
        for p in parents: 
            children += get_all_children(p)
        if len(children)==0:
            break 
        layers.append(children)
        parents = children
        nix += 1
        print(f'\nlayer={nix},number of nodes={len(parents)}')
        if iprint>0:
            for node in children:
                print('\tnode=',node)
                print('\thop_ls_left=',G.nodes[node]['hop_ls_left'])
    return G,layers

if __name__=='__main__':
    hop_ls = (0,2),(0,3),(1,2),(1,3)
    hop_ls = [(2*p+i,2*q+i) for p,q in hop_ls for i in (0,1)]
    ctr_ls = list(range(8))
    G,layers = get_all_pmg(hop_ls,ctr_ls,iprint=1)

