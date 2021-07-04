"""
Various functions for analyzing graphs, like disparity, degree distributions,
community detection and all that stuff.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse


import warnings

from collections import Counter

import networkx as nx


def plot_graph(G, filename="figure.pdf"):
    """Plot the graph and save it"""
    plt.figure(figsize=[12, 8])
    pos = nx.circular_layout(G)

    cmap = plt.cm.coolwarm

    labels = np.array(list(nx.get_edge_attributes(G, "weight").values()))
    labels = labels / np.max(labels)
    energies = list(nx.get_node_attributes(G, "energy").values())

    vmin = min(energies)
    vmax = max(energies)

    nx.draw_networkx_nodes(G, pos, node_color=energies, cmap=cmap, vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, width=labels, alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = plt.colorbar(sm)
    cbar.set_label("Energy")

    plt.savefig(filename)


def plot_graph_noenergy(G, filename=None):
    """Plot the graph and save it"""
    plt.figure(figsize=[12, 8])
    pos = nx.circular_layout(G)

    # cmap=plt.cm.coolwarm

    labels = np.array(list(nx.get_edge_attributes(G, "weight").values()))
    labels = labels / np.max(labels)
    # energies = list(nx.get_node_attributes(G, 'energy').values())

    # vmin = np.min(labels)
    # vmax = np.max(labels)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, width=labels, alpha=0.7)

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    # sm._A = []
    # cbar = plt.colorbar(sm)
    # cbar.set_label("weight")

    # plt.savefig(filename)


"""
def plot_ccdf(graph, xscale = 'log', yscale = 'log'):
    # xscale and yscale come from matplotlib:
    # acceptable values are "linear", "log", "symlog", "logit", or custom scale

    adj_matrix = nx.to_numpy_matrix(graph) # load graph as weighed adj. matrix
    adj_matrix_weights = np.tril(adj_matrix).flatten() # get lower triangle of the matrix, flattened
    adj_matrix_weights = adj_matrix_weights[adj_matrix_weights>0] # ignore all zero-weight edges
    adj_matrix_weights.sort() # sort by size

    # ccdf routine
    n = len(adj_matrix_weights)
    ccdf = 1-np.array(range(n))/float(n)

    # plotting
    plt.plot(adj_matrix_weights,ccdf)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel('weights')
    plt.ylabel('CCDF of weights')
    plt.show()
"""


def ccdf(nums):
    """Complementary cumulative distribution of the set of values"""
    nums.sort()  # sort by size

    # ccdf routine
    n = len(nums)
    ccdf = 1 - np.array(range(n)) / float(n)

    return nums, ccdf


####################
#  NETWORK METRICS #
####################

# Binary structure


def Pc_k(G):
    """ccdf of the degrees"""
    return ccdf(list(set(dict(G.degree()).values())))


def c_k(G):
    """clustering spectrum"""
    ck = {}
    clustering = nx.clustering(G)
    for node in G.nodes():
        if G.degree(node) not in ck:
            ck[G.degree(node)] = []
        c = clustering[node]
        ck[G.degree(node)].append(c)

    k_lst = sorted(ck)
    c_lst = []
    for k in k_lst:
        c_lst.append(np.array(ck[k]).mean())

    return (k_lst, c_lst, ck)


"""
Weighted structure
"""


def s_k(G):
    """Strength as a function of degree"""
    sk = {}
    for node in G.nodes():
        if G.degree(node) not in sk:
            sk[G.degree(node)] = []
        s = nx.degree(G, weight="weight")[node]
        sk[G.degree(node)].append(s)

    k_lst = sorted(sk)
    s_lst = []
    for k in k_lst:
        s_lst.append(np.array(sk[k]).mean())

    return (k_lst, s_lst, sk)


def Pc_s(G):
    """ccdf of the strengths"""
    return ccdf(list(set(dict(nx.degree(G, weight="weight")).values())))


def Pc_w(G):
    """ccdf of the weights"""
    return ccdf(list(set(dict(nx.get_edge_attributes(G, "weight")).values())))


def Y_k(G):
    """Disparity as a function of degree"""
    y = {}
    weigths = nx.get_edge_attributes(G, "weight")
    for edge in weigths:
        ni = edge[0]
        nj = edge[1]
        if ni not in y:
            y[ni] = 0.0
        if nj not in y:
            y[nj] = 0.0
        y[ni] += weigths[edge] ** 2
        y[nj] += weigths[edge] ** 2
    for node in y:
        try:
            y[node] /= (nx.degree(G, weight="weight")[node]) ** 2
        except:
            pass
    yk = {}
    for node in y:
        if G.degree(node) not in yk:
            yk[G.degree(node)] = []
        yk[G.degree(node)].append(y[node])

    k_lst = sorted(yk)
    y_lst = []
    for k in k_lst:
        y_lst.append(np.array(yk[k]).mean())

    return (k_lst, y_lst, yk)


def Y_g(G):
    """Avg. disparity of full graph assuming full connectivity"""
    M = nx.attr_matrix(G, edge_attr="weight")[0]  # get weight matrix

    M2 = np.square(M)  # weights squared
    M2strengths = np.sum(M2, axis=0)  # strengths of squared weights

    strengths = np.sum(M, axis=0)  # strengths
    strengths2 = np.square(strengths)

    enum = M2strengths
    denom = strengths2

    try:
        return np.mean(enum / denom)
    except:
        return 0.0


def Y_i(G):
    """Disparity per node"""
    y = {}
    weights = nx.get_edge_attributes(G, "weight")
    for edge in weights:
        ni = edge[0]
        nj = edge[1]
        if ni not in y:
            y[ni] = 0.0
        if nj not in y:
            y[nj] = 0.0
        y[ni] += weights[edge] ** 2
        y[nj] += weights[edge] ** 2

    for node in y:
        try:
            y[node] /= (nx.degree(G, weight="weight")[node]) ** 2
        except:
            pass

    return y


def s_i(G):
    """Strength per node"""
    y = {}
    weights = nx.get_edge_attributes(G, "weight")
    for edge in weights:
        ni = edge[0]
        nj = edge[1]
        if ni not in y:
            y[ni] = 0.0
        if nj not in y:
            y[nj] = 0.0
        y[ni] += weights[edge]
        y[nj] += weights[edge]

    return y


def s_g(G):
    """Avg. strength of full graph assuming full connectivity"""
    M = nx.attr_matrix(G, edge_attr="weight")[0]  # get weight matrix

    strengths = np.sum(M, axis=0)  # strengths
    avgstr = np.mean(strengths)

    return avgstr


def CL(G):
    """Weighted clustering coefficient (Lincoln eq 4)"""
    mx = nx.attr_matrix(G, edge_attr="weight")[0]  # get attribute matrix
    mx2 = np.matmul(mx, mx)  # get matrix power 2
    mx3 = np.matmul(mx2, mx)  # get matrix power 3

    tr3 = np.trace(mx3)

    sums = np.sum(mx2) - np.trace(
        mx2
    )  # calculate sum over all elements except diagonal ones

    try:
        output = tr3 / sums
    except:
        output = 0.0

    return output


def CV_i(G, filter=1.0):
    """Weighted clustering coefficient by node (Vespignani eq 5)
    Simplified because graph is fully connected.
    """
    mx = np.array(nx.attr_matrix(G, edge_attr="weight")[0])  # get attribute matrix
    a_ij = 1.0 - np.identity(mx.shape[0])  # create an "inverse" identity matrix
    a_ij *= mx > 0.0  # set link to zero if weight is zero

    mx *= a_ij  # kill all diagonal (so we don't sum _ii elements)

    strengths = np.sum(mx, axis=0)
    degrees = np.sum(a_ij, axis=0)

    shp = mx.shape[0]
    mmx = np.broadcast_to(
        mx,
        (
            shp,
            shp,
            shp,
        ),
    )  # broadcast matrix to accommodate for third index

    mmx_ij = np.einsum("ijk->jki", mmx)  # reindex the mmx array for easier addition
    mmx_ih = np.einsum("ijk->jik", mmx)  #

    # getting the (w_ij + w_ih) array with indices as i,j,h
    mmx_ijh = mmx_ij + mmx_ih

    # get the a_ij*a_ih*a_jh term via index contraction, in the same shape as
    a_ijh = np.einsum("ij,ih,jh->ijh", a_ij, a_ij, a_ij)

    # multiply the elements
    mmx_ijh *= a_ijh

    # get output by summing over all the j and h
    out = np.sum(mmx_ijh, axis=(1, 2))
    out *= 0.5  # multiply by 1/2

    try:
        out /= strengths * (degrees - 1)

    except:
        out = np.array([0.0] * (degrees + 1))

    return out


def D_k(G):
    """Density as a function of a degree"""
    strengths = s_k(G)
    L = G.number_of_nodes()
    if L <= 1:
        return 0.0

    k_lst = strengths[0]
    avg_strengths = strengths[1]

    yk = {}
    y_lst = []

    for deg_idx in range(len(k_lst)):
        deg = k_lst[deg_idx]
        avg_strength = avg_strengths[deg_idx]
        yk[deg] = avg_strength
        #        yk[deg] = avg_strength/(L*(L-1))
        y_lst.append(yk[deg])

    return (k_lst, y_lst, yk)


def D_g(G, binarized=False):
    """Density of graph, assuming fully connected"""
    M = nx.attr_matrix(G, edge_attr="weight")[0]  # get weight matrix
    if binarized:
        M = np.logical_not(np.isclose(M, 0.0))
    strengths = np.sum(M, axis=0)  # strengths

    try:
        return np.mean(strengths) / (np.size(strengths) - 1)
    except:
        return 0.0


def P_ij(G, i, j):
    L = G.number_of_nodes()
    weights = np.array(
        nx.attr_matrix(G, edge_attr="weight")[0]
    )  # get matrix of weights
    means = np.mean(weights, axis=1)  # get means of columns
    el_i = weights[:, i] - means[i]
    el_j = weights[:, j] - means[j]
    enumerator = np.sum(el_i * el_j)
    denominator = np.sqrt(np.sum(el_i * el_i) * np.sum(el_j * el_j))
    try:
        output = enumerator / denominator
    except:
        output = 0.0
    return output


def plot_stuff(list_G):
    plt.figure(figsize=(9, 12))

    for G in list_G:
        fig_ax = 321
        plt.subplot(fig_ax)
        plt.title("Degree dist.")
        print("Degree dist.")
        x, y = Pc_k(G)
        plt.loglog(x, y, ".-")
        fig_ax += 1

        plt.subplot(fig_ax)
        plt.title("Clust. spectrum")
        print("Clust. spectrum")
        x, y, vals = c_k(G)
        plt.loglog(x, y, ".-")
        fig_ax += 1

        plt.subplot(fig_ax)
        plt.title("Strength vs deg.")
        print("Strength vs deg.")
        x, y, vals = s_k(G)
        plt.loglog(x, y)
        x = [min(x), max(x)]
        y = np.array(x) / min(x) * y[0]
        plt.loglog(x, y, ls="--", c="k")
        fig_ax += 1

        plt.subplot(fig_ax)
        plt.title("Strength dist.")
        print("Strength dist.")
        x, y = Pc_s(G)
        plt.semilogy(x, y)
        fig_ax += 1

        plt.subplot(fig_ax)
        plt.title("Disparity")
        print("Disparity")
        x, y, vals = Y_k(G)
        plt.loglog(x, y)
        y = 1.0 / np.array(x)
        plt.loglog(x, y, ls="--", c="k")
        fig_ax += 1

        plt.subplot(fig_ax)
        plt.title("Weigth dist.")
        print("Weigth dist.")
        x, y = Pc_w(G)
        plt.semilogy(x, y)


def draw_entanglement_graph(
    G,
    layout="circular",
    edge_scale_factor=1.0,
    node_scale_factor=1.0,
    labels={},
    cmap=plt.cm.jet,
    node_color="#0A7290",
    edge_offset=0.0,
    filename="filename.pdf",
    **kwargs,
):

    """Draw the graph G"""
    nq = nx.number_of_nodes(G) + 1
    valid_layout = {"circular", "spring"}
    if layout not in valid_layout:
        raise ValueError("Not a valid layout name (circular,spring).")

    th = np.pi / 2.0 - 2.0 * np.pi / float(nq)
    if layout == "circular":
        pos = {
            node: np.array(
                [
                    np.cos(2.0 * np.pi * float(i) / float(nq) + th),
                    np.sin(2.0 * np.pi * float(i) / float(nq) + th),
                ]
            )
            for i, node in enumerate(G.nodes)
        }
        label_pos = [pos[i] * 1.15 for i in pos]
    if layout == "spring":
        pos = nx.spring_layout(G)

    # minv, maxv = (0.007174451964390649, 0.1706297630123893)
    minv, maxv = (0.0, 0.75)  # 0.75)

    edgewidth = [
        edge_offset + d["weight"] * 10 * edge_scale_factor
        for (u, v, d) in G.edges(data=True)
    ]
    edgecolors = [
        cmap((d["weight"] - minv) / (maxv - minv) + edge_offset)
        for (u, v, d) in G.edges(data=True)
    ]
    # print(edgecolors)
    nodesize = [
        (e[1] * 200 + 10) * node_scale_factor for e in G.degree(weight="weight")
    ]

    if isinstance(node_color, list):
        node_color = [f"C{v - 1}" for v in node_color]

    if isinstance(node_color, dict):
        node_color = [f"C{node_color[n] - 1}" for n in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=nodesize,
        node_color=node_color,
        edgecolors="k",
        linewidths=0.3,
        cmap=cmap,
        **kwargs,
    )
    nx.draw_networkx_edges(G, pos, width=edgewidth, edge_color=edgecolors, **kwargs)
    nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=13, **kwargs)

    #plt.savefig(filename)
    #plt.clf()