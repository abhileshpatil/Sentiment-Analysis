"""
cluster.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import json
from TwitterAPI import TwitterAPI
g=nx.Graph()

def readfile():
    user_friends={}
    all_friends=[]
    f = open("users.txt", "r+")
    for line in f:
        temp=[]
        all_friends.append((line.split(":-"))[0])
        data=(line.split(":-"))[1]
        data=(data.split("["))[1]
        data=(data.split("]"))[0]
        data=(data.split(","))
        for d in data:
            temp.append(d)
            all_friends.append(d)
        user_friends[((line.split(":-"))[0])]=temp
    f.close()
    return user_friends,all_friends

def countfriends(all_friends):
    cnt=Counter()
    for cn in all_friends:
        cnt[cn] +=1
    return cnt

def create_graph(user_friends,all_friends,friends_count):
    for k in all_friends:
        g.add_node(k)
    
    for i in user_friends:
        for j in user_friends[i]:
            g.add_edge(i,j)
    
    return g


def girvan_newman(G, depth=0):
    if G.order() == 1:
        return [G.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    result = [c.nodes() for c in components]
    return result

def write_file(f,result):
    total_count=0
    f.write("Number of communities discovered: %d"%len(result))
    f.write("\n")
    i=0
    for r in result:
        total_count=total_count+len(r)
    f.write("Average number of users per community: %d"%(total_count/len(result)))



def main():
    f= open("cluster.txt","w+")
    user_friends,all_friends=readfile()
    friends_count=countfriends(all_friends)
    # print(friends_count)
    g=create_graph(user_friends,all_friends,friends_count)
    # print(g.degree())
    result = girvan_newman(g)
    write_file(f,result)
    f.close()

if __name__ == '__main__':
    main()
