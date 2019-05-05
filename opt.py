#!/usr/bin/env python
# coding: utf-8


# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from random import choice
import random
import networkx as nx
import os
import json
from pandas.io.json import json_normalize
import collections
import itertools
import warnings
import requests

ln_cli = "docker exec -ti btcpayserver_clightning_bitcoin lightning-cli"


# In[3]:


# FUNCTIONS

# c-lightning FUNCTIONS
# GET DATA
def get_data():
    global lightning_dir
    global nodes_table
    global channels_table
    global my_node_id

    listnodes = "{} listnodes > list_of_nodes.json".format(ln_cli)
    listchannels = "{} listchannels > list_of_channels.json".format(ln_cli)
    getinfo = "{} getinfo > info.json".format(ln_cli)

    os.system(listnodes)
    os.system(listchannels)
    os.system(getinfo)

    # LOAD AND FORMAT DATA
    nodes_temp = pd.read_json('./list_of_nodes.json')
    nodes_table = json_normalize(nodes_temp['nodes'])
    channels_temp = pd.read_json('./list_of_channels.json')
    channels_table = json_normalize(channels_temp['channels'])
    with open("./info.json") as json_data:
        info = json.load(json_data)
    my_node_id = info['id']

    # GET MAIN GRAPH
    global G
    G = nx.Graph()
    G.add_nodes_from(nodes_table['nodeid'])
    edges_list = [(channels_table['source'][i], channels_table['destination'][i]) for i in range(len(channels_table))]
    G.add_edges_from(edges_list)
    G = get_main_subgraph(G)

    print('Number of nodes = ' + str(len(G.nodes())))
    print('Number of edges (payment channels) = ' + str(len(G.edges())))
    print('node ID: ' + my_node_id)


# CONNECT TO SELECTED NODES
def connect_to_new_neighbors(neighbors, channel_capacity_sats):
    node_alias = []
    num_channels = []
    ip_address = []
    for i in range(len(neighbors)):
        nd = nodes_table[nodes_table['nodeid'] == neighbors[i]]
        node_alias.append(str(list(nd.alias)[0]))
        num_channels.append(len(list(G.neighbors(neighbors[i]))))
        ip_address.append(list(nd['addresses'])[0][0]['address'])

        print("Setting up payment channel with " + node_alias[i] + "\n")
        connect = "%s connect %s@%s".format(ln_cli, neighbors[i], ip_address[i])
        print(connect)
        # os.system(connect);

        fund_channel = "%s fundchannel %s %s".format(ln_cli, neighbors[i], str(channel_capacity_sats))


#         print(fund_channel)
#         print("\n")
#     os.system(fund_channel);


# FUNCTIONS FOR PICKING NEIGHBORS


def pick_highest_metric_nodes(G, centrality_measure, num_channels_to_make):
    centrality_dict = get_centrality_dict(G, centrality_measure)
    centrality_list = [(id, centrality_dict.get(id)) for id in centrality_dict]
    sorted_by_second = sorted(centrality_list, key=lambda tup: tup[1], reverse=True)  # Sort by betweenness centrality
    return [id for id, val in sorted_by_second[0: (num_channels_to_make)]]


def pick_poor_connected_nodes(G, min_degree, num_channels_to_make):
    degree = get_centrality_dict(G, 'degree')
    between_centrality = get_centrality_dict(G, 'betweenness')

    min_degree_nodes = set()
    for id, deg in degree.items():
        if deg > min_degree:
            min_degree_nodes.add(id)

    bet_centrality = [(id, between_centrality.get(id)) for id in min_degree_nodes]
    sorted_by_second = sorted(bet_centrality, key=lambda tup: tup[1])  # Sort by betweenness centrality
    return [id for id, val in sorted_by_second[0: (num_channels_to_make)]]


# PLOTTING FUNCTIONS

def plot_ego_graph(fig, ax, G, new_node_id, centrality_measure, edge_radius):
    # Create ego graphs
    ego_graph = nx.ego_graph(G, new_node_id, radius=edge_radius)

    pos = nx.spring_layout(ego_graph, seed=3)
    centrality_dict = get_centrality_dict(G, centrality_measure)

    # Draw larger extended network
    graph1_color_vals = [centrality_dict.get(node) for node in ego_graph.nodes()]
    nx.draw_networkx_nodes(ego_graph, ax=ax, pos=pos, cmap=plt.get_cmap('viridis'),
                           node_color=graph1_color_vals, node_size=100, alpha=0.6)
    nx.draw_networkx_edges(ego_graph, ax=ax, pos=pos, alpha=0.3, edge_color='grey')

    # Draw immediate network with stronger alpha
    immediate_graph = nx.ego_graph(G, new_node_id, radius=1)

    graph2_color_vals = [centrality_dict.get(node) for node in immediate_graph.nodes()]
    nx.draw(immediate_graph, ax=ax, pos=pos, cmap=plt.get_cmap('viridis'), with_labels=False,
            node_color=graph2_color_vals, node_size=400, alpha=1, edge_color='k', width=5)

    # Create 'X' label for new node
    labels = {}
    for node in immediate_graph.nodes():
        if node == new_node_id:
            # set the node name as the key and the label as its value
            labels[node] = 'X'
    nx.draw_networkx_labels(immediate_graph, pos, labels, font_size=18, font_color='r', font_weight='bold', ax=ax)


def get_centrality_dict(G, centrality_measure):
    switcher = {
        'degree': dict(nx.degree(G)),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eccentricity': nx.eccentricity(G)
    }
    return switcher.get(centrality_measure)


def plot_centrality_hist(fig, ax, G, centrality_measure, new_node_id):
    centrality_dict = get_centrality_dict(G, centrality_measure)

    ax.hist(centrality_dict.values(), bins=25)
    node_centrality_value = centrality_dict.get(new_node_id)
    ax.axvline(x=node_centrality_value, color='r', linewidth=5)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.title.set_text(centrality_measure + ' = ' + '%.2g' % node_centrality_value)


def plot_new_node_summary_fig(G, new_node_id, edge_radius):
    sns.set(font_scale=2)

    fig = plt.figure(0, figsize=(16, 16))
    ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
    ax1 = plt.subplot2grid((3, 3), (2, 0))
    ax2 = plt.subplot2grid((3, 3), (2, 1))
    ax3 = plt.subplot2grid((3, 3), (2, 2))

    plot_ego_graph(fig, ax0, G, new_node_id, centrality_measure='betweenness', edge_radius=edge_radius)
    plot_centrality_hist(fig, ax1, G, centrality_measure='degree', new_node_id=new_node_id)
    plot_centrality_hist(fig, ax2, G, centrality_measure='betweenness', new_node_id=new_node_id)
    plot_centrality_hist(fig, ax3, G, centrality_measure='closeness', new_node_id=new_node_id)

    plt.show()


# NETWORKX FUNCTIONS

# GET THE MAIN GRAPH
def get_main_subgraph(G):
    all_sub_G = list(nx.connected_component_subgraphs(G))
    largest_sg = 0
    for i, sg in enumerate(all_sub_G):
        if sg.number_of_nodes() > largest_sg:
            largest_sg = sg.number_of_nodes()
            main_G = sg
    return main_G


# CREATE NEW GRAPH WITH NEW NODE AND EDGES
def make_graph_with_new_neighbors(G, neighbors, new_node_id):
    G_new = G.copy()
    G_new.add_node(new_node_id)
    new_edges = [(new_node_id, i) for i in neighbors]
    G_new.add_edges_from(new_edges)
    return (G_new)


def print_neighbors(neighbors):
    node_alias = []
    num_channels = []
    ip_address = []
    for i in range(len(neighbors)):
        nd = nodes_table[nodes_table['nodeid'] == neighbors[i]]
        node_alias.append(str(list(nd.alias)[0]))
        num_channels.append(len(list(G.neighbors(neighbors[i]))))
        ip_address.append(list(nd['addresses'])[0][0]['address'])
        print("node ID: " + neighbors[i])
        print("node alias: " + node_alias[i])
        print("number of channels: " + str(num_channels[i]) + "\n")


# In[4]:


# GUI BUTTONS

def suggest_nodes(centrality_measure="closeness", num_channels_to_make=2):
    global new_neighbors
    new_neighbors = pick_highest_metric_nodes(
        G, centrality_measure, num_channels_to_make)
    print_neighbors(new_neighbors)


def suggest_poor_nodes(degree=5, num_channels_to_make=2):
    global poor_neighbors
    print(new_neighbors)
    poor_neighbors = pick_poor_connected_nodes(G, degree, num_channels_to_make)
    print("poor neighbors")
    print_neighbors(poor_neighbors)


def plot_suggested_nodes():
    G_new = make_graph_with_new_neighbors(G, new_neighbors, my_node_id)
    plot_new_node_summary_fig(G_new, new_node_id=my_node_id, edge_radius=2)


def connect_nodes():
    connect_to_new_neighbors(new_neighbors, channel_capacity_sats=20000)


# def preset_casual(b):
#     centrality_buttons.value = 'closeness'
#     payment_channel_slider.value = 2
#     channel_capacity_text.value = 20000
# # preset_casual_button.on_click(preset_casual)
#
#
# def preset_business(b):
#     centrality_buttons.value = 'betweenness'
#     payment_channel_slider.value = 50
#     channel_capacity_text.value = 10000000
# # preset_business_button.on_click(preset_business)


# # Lightning Payment Channel Optimizer
#
# ### Get Lightning Network Data

# In[5]:


get_data()

# ### Suggest nodes to form payment channels with


centrality_measures = {
    "business": "betweenness",
    "casual": "closeness"
}

suggest_nodes(centrality_measures['casual'], 2)
suggest_poor_nodes(5, 2)

# plot_suggested_nodes()

# display(suggest_node_button)


# ### Visualize suggested payment channels


# display(plot_suggested_nodes_button)


# ### Set up suggested payment channels

# display(connect_to_node_button)
