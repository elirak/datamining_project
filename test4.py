import matplotlib as matplotlib
import numpy as np
import pandas as pd
from jedi.api.refactoring import inline
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.graph_objects as go
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
#%matplotlib inline
import random
import os
# Changing the working location to the location of the file

# Loading the Data
data = pd.read_csv('dataset.csv')
data.head()

# Exploring the columns of the data
# Exploring the columns of the data

print (data.state.unique())

# Stripping extra spaces in the description
data['product_name'] = data['product_name'].str.strip()

# Dropping the rows without any invoice number
data.dropna(axis = 0, subset =['orders_id'], inplace = True)
data['orders_id'] = data['orders_id'].astype('str')

# Dropping all transactions which were done on credit
data = data[~data['orders_id'].str.contains('C')]


basket_ca=(data[data['state'] == 'CA'].head(300)
           .groupby(['orders_id', 'product_name'])['products_quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))

basket_pa=(data[data['state'] == 'PA'].head(300)
           .groupby(['orders_id', 'product_name'])['products_quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))


basket_nj=(data[data['state'] == 'NJ'].head(200)
           .groupby(['orders_id', 'product_name'])['products_quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))


basket_ny=(data[data['state'] == 'NY'].head(300)
           .groupby(['orders_id', 'product_name'])['products_quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))

basket_ok=(data[data['state'] == 'OK'].head(200)
           .groupby(['orders_id', 'product_name'])['products_quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))

basket_la=(data[data['state'] == 'LA'].head(100)
           .groupby(['orders_id', 'product_name'])['products_quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))

# Defining the hot encoding function to make the data suitable
# for the concerned libraries
def hot_encode(x):
	if(x<= 0):
		return 0
	if(x>= 1):
		return 1


# Encoding the datasets
basket_encoded = basket_pa.applymap(hot_encode)
basket_pa = basket_encoded

basket_encoded = basket_ca.applymap(hot_encode)
basket_ca = basket_encoded

basket_encoded = basket_nj.applymap(hot_encode)
basket_nj = basket_encoded

basket_encoded = basket_ny.applymap(hot_encode)
basket_ny = basket_encoded

basket_encoded = basket_ok.applymap(hot_encode)
basket_ok = basket_encoded

basket_encoded = basket_la.applymap(hot_encode)
basket_la = basket_encoded

frq_items = apriori(basket_encoded, min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = pd.DataFrame(rules.sort_values(['confidence', 'lift'], ascending =[False, False]))

pd.set_option('display.max_columns', None)
print ('Assocation Rules')
print(rules.head())


# Building the model
frq_items = apriori(basket_ca, min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = pd.DataFrame(rules.sort_values(['confidence', 'lift'], ascending =[False, False]))
pd.set_option('display.max_columns', None)
print ('CA Rules')
print(rules.head())



frq_items = apriori(basket_nj, min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = pd.DataFrame(rules.sort_values(['confidence', 'lift'], ascending =[False, False]))
pd.set_option('display.max_columns', None)
print ('NJ Rules')
print(rules.head())



frq_items = apriori(basket_ny, min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = pd.DataFrame(rules.sort_values(['confidence', 'lift'], ascending =[False, False]))
pd.set_option('display.max_columns', None)
print ('NY Rules')
print(rules.head())








frq_items = apriori(basket_pa, min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = pd.DataFrame(rules.sort_values(['confidence', 'lift'], ascending =[False, False]))
pd.set_option('display.max_columns', None)
print ('PA Rules')
#print(rules.head())
print(rules.head())




frq_items = apriori(basket_ok, min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = pd.DataFrame(rules.sort_values(['confidence', 'lift'], ascending =[False, False]))
pd.set_option('display.max_columns', None)
print ('Ok Rules')
print(rules.head())


"""frq_items = apriori(basket_la, min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = pd.DataFrame(rules.sort_values(['confidence', 'lift'], ascending =[False, False]))
pd.set_option('display.max_columns', None)
print ('LA Rules')
print(rules.head())
"""

rules.hist('confidence', grid=False, bins=30)
plt.title('confidence')
plt.show()

rules.hist('lift', grid=False, bins=30)
plt.title('Lift')
plt.show()

support=rules['support']
confidence=rules['confidence']
lift=rules['lift']

for i in range(len(support)):
    support[i]= support[i]+ 0.0025 * (random.randint(1,10)- 5)
    confidence[i] = confidence[i] + 0.0025 * (random.randint(1, 10) - 5)
    #lift[i] = lift[i] + 0.0025 * (random.randint(1, 10) - 5)
plt.scatter(rules['support'],  rules['confidence'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')

plt.show()

plt.scatter(support, confidence, label=None,
            c=np.log10(lift), cmap='viridis',
            linewidth=0, alpha=0.5)


plt.xlabel('support')
plt.ylabel('confidence')
plt.colorbar(label='log$_{10}$(lift)')
plt.clim(3, 7)

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Frquent Items')
plt.show()



"""plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel(frq_items)

plt.show()"""

"""
plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'],
 fit_fn(rules['lift']))"""


#results = list(rules)
"""import matplotlib.pyplot as plt


support = rules['support']
confidence = rules['confidence']

plt.title('Association Rules')
plt.xlabel(plt.scatter(rules['support'], rules['lift'], alpha=0.5))
plt.ylabel(plt.scatter(rules['antecedents'], rules['consequents'],alpha=0.5))

plt.show()
"""

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
            #G1.add_node(rules['support'])

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

draw_graph (rules, 10)