import matplotlib.pyplot as plt
import networkx as nx
import quimb.tensor as qtn
import os
import numpy as np
import itertools
import copy


def draw(
    tn,
    mode="tn",
    font_size=10,
    figsize=10,
    arrow_center=0.85,
    arrow_length=0.15,
    layout="kamada_kawai",
    filename=None,
    **kwargs,
):
    """
    Draw a TensorNetwork `tn` using matplotlib.

    :param tn: a `TensorNetwork`
    :param mode: can be:
        - "tn": (default) a general tensor network. It is the dual graph of an MRF: nodes denote tensors, and edges denote shared indices.
        - "bn": a Bayesian network, i.e. directed acyclic graph. The child of each potential is assumed to be its first dimension. Nodes denote variables, and edges denote CPT's.
        - "mrf": a Markov random field. Works like "bn", but the graph is undirected and parents are moralized (i.e. connected to each other).
    :param font_size: font size for node or edge labels
    """

    assert mode in ("bn", "mrf", "tn")

    G, _ = qtn.drawing.draw_tn(
        tn,
        get="graph,pos",
        layout=layout,
        font_size=font_size,
        font_size_inner=font_size,
        label_color="black",
        node_color="lightgray",
        edge_color="black",
        **kwargs,
    )
    # from networkx.drawing.nx_pydot import graphviz_layout

    # print(G.edges[next(iter(G.edges))])
    # pos = graphviz_layout(G, prog="twopi")
    # print(pos)
    # for p in pos:
    #     G.nodes[p]["coo"] = pos[p]
    # print(G.nodes[p])
    # print(pos)
    # print(G.nodes[next(iter(G.nodes))])

    if mode == "tn":
        newG = G
    else:
        node_data = copy.deepcopy(G.nodes[next(iter(G.nodes))])
        edge_data = copy.deepcopy(G.edges[next(iter(G.edges))])
        edge_data["label"] = [None]
        edge_data["arrow_left"] = [False]
        if mode == "bn":
            edge_data["arrow_right"] = [True]
        edge_data["edge_size"] = [1]
        # edge_data['color'] = [[0, 0, 0]]

        pos = {}
        for edge in list(G.edges):
            ed = G.get_edge_data(edge[0], edge[1])
            center = (np.array(ed["coos"][0]) + np.array(ed["coos"][1])) / 2
            if ed["label"][0] != "":
                pos[ed["label"][0]] = center
            # if mode == "bn":
            #     ed["label"][0] = ""
            # ed['color'] = [[0, 0, 0, 1]]
        for node in list(G.nodes):
            if "ind" in G.nodes[node]:
                pos[G.nodes[node]["ind"]] = G.nodes[node]["coo"]
        newG = nx.DiGraph()
        for child in pos:
            node_data["coo"] = pos[child]
            if show_labels:
                node_data["label"] = child
            else:
                node_data.discard("label")
            newG.add_node(child, **node_data)
        for t in tn.tensors:
            parents = t.inds[1:]
            child = t.inds[0]
            for parent in parents:
                edge_data["coos"] = [pos[parent], pos[child]]
                # print(edge_data, G.nodes[parent])
                newG.add_edge(parent, child, **edge_data)
            if mode == "mrf":
                for combination in itertools.combinations(parents, 2):
                    edge_data["coos"] = [pos[combination[0]], pos[combination[1]]]
                    newG.add_edge(combination[0], combination[1], **edge_data)

    qtn.drawing._draw_matplotlib(
        newG.edges,
        newG.nodes,
        arrow_opts=dict(center=arrow_center, width=0.05, length=arrow_length),
        return_fig=True,
        figsize=(figsize, figsize),
    )
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        os.system(f"pdfcrop {filename} {filename}")
        plt.clf()
