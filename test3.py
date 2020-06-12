import pandas as pd
import numpy as np
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df1=pd.read_csv('dataset.csv')
df1.head()
print (df1.state.value_counts())

df1 = df1[df1.state == 'CA']
df1['product_name'] = df1['product_name'].str.strip()

df1 = df1[df1.products_quantity >0]



basket_ca=(df1[df1.state == 'CA'].head(30)
           .groupby(['orders_id', 'product_name'])['products_quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))


#print(basket_ca.head())
#print(basket_ca['Cheese Pizza'].head())

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets=basket_ca.applymap(encode_units)

""""
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
#basket_sets= basket_sets.drop(columns=['POSTAGE'])


frequent_items = apriori(basket_sets, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.head()
rules.sort_values(by = ["lift"],ascending=False)
print(rules.head())


#print( rules.head())
#print( rules[ (rules['lift'] >= 6) &
#       (rules['confidence'] >= 0.8) ])

#basket_nj=(df1[df1.state == 'NJ'].head(20)
#           .groupby(['orders_id', 'product_name'])['products_quantity']
#           .sum().unstack().reset_index().fillna(0)
#           .set_index('orders_id'))


#basket_sets2 = basket_nj.applymap(encode_units)

#frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
#rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

#print(rules2[ (rules2['lift'] >= 4) &
#        (rules2['confidence'] >= 0.5)])


#df1 = df1[df1.state == 'NJ']
#df1['product_name'] = df1['product_name'].str.strip()
#basket_nj=(df1[df1.state == 'NJ'].head(20)
#           .groupby(['orders_id', 'product_name'])['products_quantity']
#           .sum().unstack().reset_index().fillna(0)
#           .set_index('orders_id'))
#print(basket_nj)

#df1 = df1[df1.state == 'NY']
#df1['product_name'] = df1['product_name'].str.strip()
#basket_ny=(df1[df1.state == 'NY'].head(10)
#           .groupby(['orders_id', 'product_name'])['products_quantity']
#           .sum().unstack().reset_index().fillna(0)
#           .set_index('orders_id'))
#print(basket_ny)

"""
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)



#print(frequent_itemsets)

rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#print(rules_mlxtend.head())

res=(pd.DataFrame(rules_mlxtend[ (rules_mlxtend['lift'] >= 6) & (rules_mlxtend['confidence'] >= 0.8) ]))
print(res)

import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(rules, rules_to_show):
    G1 = nx.DiGraph()
    color_map=[]
    N = 50
    colors = np.random.rand(N)
    strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']

    for i in range(rules_to_show):
        G1.add_nodes_from(["R"+str(i)])
        for a in rules.iloc[i]['antecedents']:
            G1.add_nodes_from([a])
            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 4)
        for c in rules.iloc[i]['consequents']:
            G1.add_nodes_from([c])
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=4)

    for node in G1:
        found_a_string = False
        for item in strs:
            if node==item:
                found_a_string = True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]

    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16,
            with_labels=False)

    for p in pos:  # raise text positions
        pos[p][1] += 0.07
        nx.draw_networkx_labels(G1, pos)
        plt.show()

draw_graph (rules_mlxtend, 10)

