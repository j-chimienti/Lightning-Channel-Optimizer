#!/usr/bin/env python
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import networkx as nx
import os
import json
from pandas.io.json import json_normalize



def get_data():

    # LOAD AND FORMAT DATA
    nodes_temp = pd.read_json('./data/listnodes.json')
    nodes_table = json_normalize(nodes_temp['nodes'])
    channels_temp = pd.read_json('./data/listchannels.json')
    channels_table = json_normalize(channels_temp['channels'])
    with open("./data/getinfo.json", "r") as json_data:
        info = json.load(json_data)
    my_node_id = info['id']

    # GET MAIN GRAPH
    global G
    G = nx.Graph()
    G.add_nodes_from(nodes_table['nodeid'])
    edges_list = [(channels_table['source'][i], channels_table['destination'][i]) for i in range(len(channels_table))]
    G.add_edges_from(edges_list)
    G = get_main_subgraph(G)

    print("\nNode Info\n")
    print('Number of nodes = ' + str(len(G.nodes())))
    print('Number of edges (payment channels) = ' + str(len(G.edges())))
    print('node ID = ' + my_node_id)
    node_info = dict(nodes=len(G.nodes()), edges=len(G.edges()), id=my_node_id)
    with open("./data/nodeinfo.json", "w") as node_info_file:
        json.dump(node_info, node_info_file)

    return nodes_table, channels_table, my_node_id


def connect_to_new_neighbors(neighbors, channel_capacity_sats, nodes_table):
    node_alias = []
    num_channels = []
    ip_address = []
    statements = []
    for i in range(len(neighbors)):
        nd = nodes_table[nodes_table['nodeid'] == neighbors[i]]
        node_alias.append(str(list(nd.alias)[0]))
        num_channels.append(len(list(G.neighbors(neighbors[i]))))
        ip_address.append(list(nd['addresses'])[0][0]['address'])

        print("Setting up payment channel with " + node_alias[i] + "\n")

        fund_channel = {
            "id": neighbors[i],
            "capacity": channel_capacity_sats,
            "netaddr": ip_address[i],
            "alias": node_alias[i],
        }
        print(fund_channel)
        statements.append(fund_channel)
    return statements


# FUNCTIONS FOR PICKING NEIGHBORS


def pick_highest_metric_nodes(G, centrality_measure, num_channels_to_make):
    centrality_dict = get_centrality_dict(G, centrality_measure)
    centrality_list = [(_id, centrality_dict.get(_id)) for _id in centrality_dict]
    sorted_by_second = sorted(centrality_list, key=lambda tup: tup[1], reverse=True)  # Sort by betweenness centrality
    return [_id for _id, val in sorted_by_second[0: num_channels_to_make]]


def pick_poor_connected_nodes(G, min_degree, centrality='betweenness'):
    degree = get_centrality_dict(G, 'degree')
    between_centrality = get_centrality_dict(G, centrality)
    min_degree_nodes = set()
    for id, deg in degree.items():
        if deg > min_degree:
            min_degree_nodes.add(id)
    bet_centrality = [(id, between_centrality.get(id)) for id in min_degree_nodes]
    sorted_by_second = sorted(bet_centrality, key=lambda tup: tup[1])
    return sorted_by_second


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

    plt.savefig("data/node_summary.png")
    # plt.show()

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
    return G_new


def print_neighbors(neighbors, nodes_table):
    node_alias = []
    num_channels = []
    ip_address = []
    neighbors_ = []
    for i in range(len(neighbors)):
        nd = nodes_table[nodes_table['nodeid'] == neighbors[i]]
        node_alias.append(str(list(nd.alias)[0]))
        num_channels.append(len(list(G.neighbors(neighbors[i]))))
        ip_address.append(list(nd['addresses'])[0][0]['address'])

        # print("Node: {id: {}, alias: {}, num_of_channels: {} }".format(neighbors[i], node_alias[i], num_channels[i]))
        print("node ID: " + neighbors[i])
        print("node alias: " + node_alias[i])
        print("number of channels: " + str(num_channels[i]) + "\n")

        _node = dict(id=neighbors[i], alias=node_alias[i], channels=num_channels[i])
        neighbors_.append(_node)
    return neighbors_


def suggest_nodes(centrality_measure="closeness", num_channels_to_make=2):
    print("\nSuggest nodes\n")
    new_neighbors = pick_highest_metric_nodes(
        G, centrality_measure, num_channels_to_make)
    _new_neighbors = print_neighbors(new_neighbors, nodes_table)
    return new_neighbors, _new_neighbors


def suggest_poor_nodes(degree, centrality):
    print("Crunching poor neighbors")
    poor_neighbors = pick_poor_connected_nodes(G, degree, centrality)
    return poor_neighbors


def plot_suggested_nodes(new_neighbors, my_node_id):
    print("\nPlot nodes\n")
    G_new = make_graph_with_new_neighbors(G, new_neighbors, my_node_id)
    plot_new_node_summary_fig(G_new, new_node_id=my_node_id, edge_radius=2)


presets = {
    "casual": {
        "centrality": "closeness",
        "paymment_channels": 2,
        "capacity": 20e3  # 20k sats
    },
    "business": {
        "centrality": "betweenness",
        "paymment_channels": 50,
        "capacity": 10e6  # 10m sats
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="LN Channel Optimzer", description="Lightning Channel Optimizer")
    parser.add_argument("--centrality", type=str, default="betweenness", help="Available centralities - betweenness or closeness")
    parser.add_argument("--channels", type=int, default=5, help="the number of peers to suggest")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--plot", type=bool, default=False, help="plot network graph (slow)")
    parser.add_argument("--poor-nodes", type=bool, default=False, help="Calculate poor nodes")
    args = parser.parse_args()
    if args.centrality not in ['closeness', 'betweenness']:
        print("Invalid centrality {}. Must be closeness or betweenness".format(args.centrality))
        exit(1)
    # Fetch data from lightning
    os.system("./ln.sh")
    (nodes_table, channels_table, my_node_id) = get_data()
    # Get top 25 nodes (even if they are peers)
    (new_neighbors, _new_neighbors) = suggest_nodes(args.centrality, 25)
    if args.poor_nodes:
        poor_nodes = suggest_poor_nodes(args.degree,args.centrality)
        with open("./data/poornodes.json", "w") as poor_nodes_file:
            json.dump(poor_nodes, poor_nodes_file)
    with open("./data/listpeers.json", "r") as list_peers:
        peers = json.load(list_peers)['peers']
    peers_ids = [node['id'] for node in peers if node['connected'] is True
                 and len(node['channels']) > 0]
    connect_peers = [node for node in _new_neighbors if node['id'] not in peers_ids
                     and node['id'] not in ["032b2b3f4abda9677bb9563e226c068d3a2456fb8b036635a81c9bcaa1671d1ada", "02cdf83ef8e45908b1092125d25c68dcec7751ca8d39f557775cd842e5bc127469"]
                     ]
    with open("./data/suggest_nodes.json", "w") as suggest_nodes_file:
        json.dump(_new_neighbors, suggest_nodes_file)
    with open("./data/suggest_connect_nodes.json", "w") as suggested_nodes_file:
        json.dump(connect_peers, suggested_nodes_file)
    if args.plot:
        plot_suggested_nodes([peer['id'] for peer in connect_peers[:args.channels
                                                     ]], my_node_id)


    # Uncomment to connect and fund channels

    # cpeers = connect_peers[:args.channels]
    # statments = connect_to_new_neighbors(cpeers, args.capacity)
    # _json = {
    #     "nodes": statments
    # }
    # with open("./data/listconnect.json", "w") as list_connect:
    #     json.dump(_json, list_connect)
    # #fund_statment = "ansible-playbook fund.yml --extra-vars @./data/nodes.json"
    # fund_statment = "ansible-playbook -v fund.yml --extra-vars '%s'" % json.dumps(_json)
    # print(fund_statment)
    # fund_channels_input = input("Fund {} channels? [Y/n]".format(len(cpeers)))
    # for statement in statments:
    #     print("{}@{}".format(statement['id'], statement['netaddr']))
    #
    # if fund_channels_input in ["Y", "y"]:
    #     os.system(fund_statment)
    # else:
    #     print("Not funding")
    #     exit(1)
    #
