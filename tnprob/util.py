import copy
import itertools
import operator
import os
import time
import warnings
from collections import defaultdict
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
import quimb as qu
import quimb.experimental.tn_marginals
import quimb.tensor as qtn
import scipy.stats as stats
import tntorch
import torch
from rich import print

torch.set_default_dtype(torch.float64)


def open_bif(path: str, library="pyagrum") -> qtn.TensorNetwork:
    """
    Read a BN in .bif format into a TensorNetwork
    """

    if library == "pyagrum":
        import pyAgrum as gum

        bn = gum.loadBN(path)
        return qtn.TensorNetwork(
            [
                qtn.Tensor(
                    bn.cpt(name)
                    .toarray()
                    .transpose(*range(bn.cpt(name).nbrDim() - 1, -1, -1)),
                    inds=bn.cpt(name).names[::],
                )
                # qtn.Tensor(
                #     bn.cpt(name).toarray(),
                #     inds=bn.cpt(name).names,
                # )
                for name in bn.names()
            ]
        )
    elif library == "pgmpy":
        import pgmpy.readwrite

        g = pgmpy.readwrite.BIFReader(path).get_model()
        g = g.to_markov_model()
        return qtn.TensorNetwork(
            [qtn.Tensor(f.values, inds=f.variables) for f in g.get_factors()]
        )
    else:
        raise ValueError(f"Unsupported library {library}")


def save_bif(tn: qtn.TensorNetwork, path: str, library="pyagrum"):
    """
    Save a TensorNetwork to a BN in .bif format
    """

    if library == "pyagrum":
        raise NotImplementedError
        # import pyAgrum as gum

        # bn = gum.BayesNet()
        # for t in tn.tensors:
        #     bn.add(gum.CPT(t.data.transpose(*range(t.data.ndim - 1, -1, -1)), t.inds))
        # bn.saveBIF(path)
    elif library == "pgmpy":
        import pgmpy.readwrite

        g = pgmpy.models.BayesianNetwork()
        for t in tn.tensors:
            for ind in t.inds:
                g.add_node(ind)
            f = pgmpy.factors.discrete.TabularCPD(
                variable=t.inds[0],
                variable_card=t.data.shape[0],
                values=np.reshape(np.asarray(t.data), (t.data.shape[0], -1)),
                evidence=t.inds[1:],
                evidence_card=t.data.shape[1:],
            )
            g.add_cpds(f)
        writer = pgmpy.readwrite.BIFWriter(g)
        writer.write_bif(filename=path)
    else:
        raise ValueError(f"Unsupported library {library}")


def from_tntorch(t: tntorch.Tensor, names=None, rank_label="") -> qtn.TensorNetwork:
    """
    Create a `TensorNetwork` out of a `tntorch` tensor.

    See https://github.com/rballester/tntorch

    :param t: a `tntorch.tensor`
    :param names: the names for the physical dimensions of the tensor. If None (default), 'I1', 'I2', etc. will be used
    :param rank_label: optional tag to be attached at the end of each rank node, to avoid collisions when merging multiple tensors. Default is the empty string
    :return: a `qtn.TensorNetwork`
    """

    if names is None:
        names = ["I{}".format(n + 1) for n in range(t.dim())]
    assert len(names) == t.dim()

    result = qtn.TensorNetwork()
    rank_counter = 0
    for n in range(t.dim()):
        # Add Tucker factor if there is one
        if t.Us[n] is not None:
            result |= qtn.Tensor(
                t.Us[n], inds=[names[n], "S{}{}".format(n + 1, rank_label)]
            )
        if t.cores[n].dim() == 2:  # CP factor
            if t.Us[n] is None:  # Pure CP
                result |= qtn.Tensor(
                    t.cores[n],
                    inds=[names[n], "R{}{}".format(rank_counter, rank_label)],
                )
            else:  # CP-Tucker
                result |= qtn.Tensor(
                    t.cores[n],
                    inds=[
                        "S{}{}".format(n + 1, rank_label),
                        "R{}{}".format(rank_counter, rank_label),
                    ],
                )
        else:  # TT factor
            if t.Us[n] is None:  # Pure TT
                physical = names[n]
            else:  # TT-Tucker
                physical = "S{}{}".format(n + 1, rank_label)
            if n == 0:  # First TT core is added as matrix
                result |= qtn.Tensor(
                    t.cores[n][0, ...],
                    inds=[
                        physical,
                        "R{}{}".format(rank_counter + 1, rank_label),
                    ],
                )
            elif n == t.dim() - 1:  # Last TT core is a matrix too
                result |= qtn.Tensor(
                    t.cores[n][..., 0],
                    inds=["R{}{}".format(rank_counter, rank_label), physical],
                )
            else:  # Middle TT core
                result |= qtn.Tensor(
                    t.cores[n],
                    inds=[
                        "R{}{}".format(rank_counter, rank_label),
                        physical,
                        "R{}{}".format(rank_counter + 1, rank_label),
                    ],
                )
            rank_counter += 1
    return result


def numel(tn: qtn.TensorNetwork) -> int:
    return sum([t.size for t in tn.tensors])


def get_treewidth(tn: qtn.TensorNetwork) -> int:
    """
    Calculate the treewidth of this TN. Uses the `networkx` library
    """

    g = nx.Graph()
    for k in tn.tensor_map.keys():
        g.add_node(k)
    for ind in tn.all_inds():
        ind_map = list(tn.ind_map[ind])
        for i in range(len(ind_map)):
            for j in range(i + 1, len(ind_map)):
                g.add_edge(ind_map[i], ind_map[j])
    return nx.algorithms.approximation.treewidth_min_fill_in(g)[0]


def get_cpt(tn: qtn.TensorNetwork, name: str, pop=False) -> qtn.Tensor:
    """
    For a `TensorNetwork` representing a Bayesian network, return the tensor representing conditional probability table of a node.

    :param pop: if True, pop (remove) the tensor from the TensorNetwork in place.
    """

    matches = [i for i in tn.ind_map[name] if tn.tensor_map[i].inds[0] == name]
    if len(matches) > 1:
        raise ValueError(
            f"This TensorNetwork seems to have multiple CPTs for node {name}"
        )
    elif len(matches) == 0:
        raise ValueError(f"This TensorNetwork does not have a CPT for node {name}")
    tid = matches[0]
    result = tn.tensor_map[tid]
    if pop:
        tn.pop_tensor(tid)
    return result


############
# Operations
############


def product(tns: List[qtn.TensorNetwork], outer: List[str] = None) -> qtn.TensorNetwork:
    """
    Create a graph representing the product of two or more graphs, such that
        result[outer] = g1[outer] * g2[outer]

    :param tns: a list of `TensorNetworks`s, separated by commas
    :param outer: a list of nodes. Must be present in all TNs
    :return: a new `TensorNetwork`
    """

    if outer is None:
        outer = []
    result = qtn.TensorNetwork()
    for i, tn in enumerate(tns):
        for t in tn.tensors:
            tnew = t.reindex(
                {ind: ind if ind in outer else f"{ind}_product_{i}" for ind in t.inds}
            )
            result |= tnew
    return result


def division(tn1: qtn.TensorNetwork, tn2: qtn.TensorNetwork) -> qtn.TensorNetwork:
    tn2inv = qtn.TensorNetwork(
        [qtn.Tensor(1 / (t.data + 1e-9), t.inds) for t in tn2.tensors]
    )
    result = tn1.combine(tn2inv, check_collisions=False)
    return result
