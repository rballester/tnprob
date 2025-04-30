import itertools
import operator
import os
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
from tnprob.util import *
from rich import print

torch.set_default_dtype(torch.float64)


def modify(
    phi: qtn.Tensor, coords: np.ndarray, uncertainty_fun: callable, prepend: str = None, eps: float = 1e-6,
) -> tntorch.Tensor:
    """
    Given a tensor, encode uncertainties as extra dimensions
    """

    if prepend is None:
        prepend = ""
    else:
        prepend = prepend + "_"
    inds = phi.inds
    phi = phi.data
    tphi = tntorch.Tensor(phi)
    tphi = tntorch.round(tphi)
    if len(coords) > 0:
        coords = np.array(coords)
        tphi = tntorch.unsqueeze(tphi, range(len(coords)))
    ts = []
    disc = None
    for i, row in enumerate(coords):
        idx = list(row.copy())
        idx[0] = slice(None)
        vec = phi[tuple(idx)]
        # vec += 1e-7 * np.random.randn(*vec.shape)
        vec.data += 1e-7 * torch.ones(*vec.shape)
        current = vec.clone()
        uf = uncertainty_fun(current[row[0]])
        disc = len(uf)
        vec = torch.tile(vec[None, ...], [disc, 1])
        vec[:, row[0]] = torch.Tensor(uf)
        vec[:, np.delete(np.arange(phi.shape[0]), row[0])] /= torch.sum(
            vec[:, np.delete(np.arange(phi.shape[0]), row[0])], dim=1, keepdim=True
        ) / (1 - vec[:, row[0] : row[0] + 1])
        vec -= current[None, :]
        t = tntorch.Tensor(vec)
        dims = list(range(len(coords) + phi.ndim))
        dims.remove(i)
        dims.remove(len(coords))
        t = tntorch.unsqueeze(t, dim=dims)
        for j in range(len(coords) + 1, phi.ndim + len(coords)):
            assert t.shape[j] == 1
            core = torch.zeros(
                t.cores[j].shape[0], phi.shape[j - len(coords)], t.cores[j].shape[2]
            )
            core[:, row[j - len(coords)], :] = t.cores[j][:, 0, :]
            t.cores[j] = core
        for j in range(len(coords)):
            if j != i:
                t.cores[j] = t.cores[j].repeat(1, disc, 1)
        ts.append(t)
    for j in range(len(coords)):
        tphi.cores[j] = tphi.cores[j].repeat(1, disc, 1)
    tssum = tntorch.reduce([tphi] + ts, operator.add, eps=eps)
    names = []
    for i in range(len(coords)):
        conditions = ", ".join(
            [f"{inds[j]}={coords[i, j]}" for j in range(1, coords.shape[1])]
        )
        if len(conditions) > 0:
            conditions = " | " + conditions
        names.append(f"P({inds[0]}={coords[i, 0]}" + conditions + ")")
    # tssum = tssum.decompress_tucker_factors()
    tn = from_tntorch(tssum, names=names + list(inds))
    tn.reindex(
        {
            old_tag: (
                old_tag if old_tag in inds or old_tag[:2] == "P(" else prepend + old_tag
            )
            for old_tag in tn.all_inds()
        },
        inplace=True,
    )
    tn.check()
    return tn


def encode_cpt_uncertainty(
    tn: qtn.TensorNetwork, df: pd.DataFrame, uncertainty_fun: callable, eps: float = 1e-6
):
    """
    Encode uncertainty in a TN by adding a new dimensions to its CPT's.

    To prevent exponential blow-up of parameters, the new CPT's are compactly represented in the TT format.
    """

    tnmodified = tn.copy()
    additions = qtn.TensorNetwork()

    for name, group in df.groupby("child"):
        phi = get_cpt(tnmodified, name=name, pop=True)
        tnphi = modify(
            phi,
            np.row_stack(group["coordinates"].values),
            uncertainty_fun=uncertainty_fun,
            prepend=name,
            eps=eps,
        )
        additions = additions | tnphi
    return tnmodified | additions


def variance(
    tn: qtn.TensorNetwork,
    inputs: List[str],
    output: str,
    values: List,
    z: qtn.TensorNetwork = None,
) -> float:

    if z is None:
        marginals = (
            quimb.experimental.tn_marginals.compute_all_marginals_via_torch_autodiff(
                tn, output_inds=inputs
            )
        )
        z = qtn.TensorNetwork(
            [qtn.Tensor(torch.Tensor(v), inds=[k]) for k, v in marginals.items()]
        )

    if output is None:
        tno = tn
    else:
        tno = tn | qtn.Tensor(torch.Tensor(values), inds=[output])
    f = division(tno, z)
    f2z = product([f, tno], outer=inputs)
    ex2 = f2z.contract(output_inds=[])
    e2x = tno.contract(output_inds=[]) ** 2
    return ex2 - e2x


def _marginals(tn, output_inds, normalize=False):
    return {i: tn.contract(output_inds=[i]).data.numpy() for i in output_inds}
    # tn = tn.copy(deep=True)
    # if len(tn.tensors) == 1:
    #     tn = tn | qtn.Tensor(torch.ones([1]), inds="-")
    # tn.apply_to_arrays(torch.tensor)
    # result = quimb.experimental.tn_marginals.compute_all_marginals_via_torch_autodiff(
    #     tn, output_inds=output_inds
    # )
    # if normalize:
    #     return result
    # else:
    #     z = tn.contract(output_inds=[]).item()
    #     return {k: v * z for k, v in result.items()}


def sobol_all(
    tn: qtn.TensorNetwork,
    inputs: List[str],
    output: str,
    values: List,
    simplify: bool = False,
    z: qtn.TensorNetwork = None,
) -> float:
    """
    Compute the first-order variance components and total indices of all input nodes on the expected value of an output node.

    Note: this method assumes uncorrelated inputs.
    """

    assert all([j in tn.ind_map for j in inputs])

    if z is None:  # Used in the original BN Sobol examples
        tn2 = tn.full_simplify(output_inds=inputs)
        marginals = _marginals(tn2, output_inds=inputs)
        z = qtn.TensorNetwork(
            [qtn.Tensor(torch.Tensor(v), inds=[k]) for k, v in marginals.items()]
        )

    if output is None:
        tno = tn
    else:
        tno = tn | qtn.Tensor(torch.Tensor(values), inds=[output])
    tno.check()

    if simplify:
        tno.apply_to_arrays(np.array)
        tno = tno.full_simplify(output_inds=inputs)
        tno.apply_to_arrays(torch.tensor)

    # Required for both kinds of indices
    f = division(tno, z)
    ftno = product([f, tno], outer=inputs)
    Ef2 = ftno.contract(output_inds=[]).data.numpy()
    Ef = tno.contract(output_inds=[]).data.numpy()

    if simplify:
        tno.apply_to_arrays(np.array)
        tno = tno.full_simplify(output_inds=inputs, atol=1e-6)
        tno.apply_to_arrays(torch.tensor)

    mminusi = _marginals(tno, output_inds=inputs)
    zminusi = _marginals(z, output_inds=inputs)

    V = None

    variance_components = {}
    for k in mminusi:
        fminusi = mminusi[k] / zminusi[k]
        E2fminusi = np.sum(fminusi * zminusi[k]) ** 2
        Efminusi2 = np.sum(fminusi**2 * zminusi[k])

        # Denominator: Var[f]
        if V is None:
            E2f = E2fminusi
            V = Ef2 - E2f
            if V < 1e-4:
                print("Warning: variance is", V)
        variance_components[k] = (Efminusi2 - E2fminusi) / V

    total_indices = {}
    for i in inputs:
        tnoi = tno.copy()
        tnoi.sum_reduce(i, inplace=True)

        zi = z.copy()
        zi.sum_reduce(i, inplace=True)
        E2fi = Ef**2
        # E2fi = tnoi.contract(output_inds=[]).data.numpy() ** 2

        fi = division(tnoi, zi)
        fitnoi = product([fi, tnoi], outer=set(inputs).difference(i))
        Efi2 = fitnoi.contract(output_inds=[]).data.numpy()
        total_indices[i] = 1 - (Efi2 - E2fi) / V

    return {"variance_components": variance_components, "total_indices": total_indices}


def sobol_one(
    tn: qtn.TensorNetwork,
    inputs: List[str],
    i: Union[str, Iterable[str]],
    output: str,
    values: List,
    kind: str,
    z: qtn.TensorNetwork = None,
) -> float:
    """
    Compute the variance component of a single tuple of the input nodes on the expected value of an output node.

    Note: this method assumes uncorrelated inputs.
    """

    assert kind in ("variance_component", "total_index")

    assert all([j in tn.ind_map for j in inputs])
    if not isinstance(i, (list, tuple)):
        i = [i]
    assert set(i).issubset(inputs)

    if output is None:
        tno = tn
    else:
        tno = tn | qtn.Tensor(torch.Tensor(values), inds=[output])

    if z is None:
        marginals = (
            quimb.experimental.tn_marginals.compute_all_marginals_via_torch_autodiff(
                tn, output_inds=inputs
            )
        )
        z = qtn.TensorNetwork(
            [qtn.Tensor(torch.Tensor(v), inds=[k]) for k, v in marginals.items()]
        )

    # Required for both kinds of indices
    f = division(tno, z)
    ftno = product([f, tno], outer=inputs)
    Ef2 = ftno.contract(output_inds=[]).data.numpy()

    if kind == "variance_component":
        # Numerator: Var_i[E_{~i}[f]]
        mminusi = tno.contract(output_inds=i)

        zminusi = z.contract(output_inds=i).data.numpy()
        fminusi = (mminusi / zminusi).data.numpy()
        E2fminusi = np.sum(fminusi * zminusi) ** 2
        Efminusi2 = np.sum(fminusi**2 * zminusi)

        # Denominator: Var[f]
        E2f = E2fminusi
        V = Ef2 - E2f
        if V < 1e-4:
            print("Warning: variance is", V)
        # print("Variance:", V)
        return (Efminusi2 - E2fminusi) / V

    elif kind == "total_index":
        # Numerator: E_i[Var_{~i}[f]]
        tnoi = tno.copy()
        for var in i:
            tnoi.sum_reduce(var, inplace=True)

        zi = z.copy()
        for var in i:
            zi.sum_reduce(var, inplace=True)
        E2fi = tnoi.contract(output_inds=[]).data.numpy() ** 2

        fi = division(tnoi, zi)
        fitnoi = product([fi, tnoi], outer=set(inputs).difference(i))
        Efi2 = fitnoi.contract(output_inds=[]).data.numpy()

        # Denominator: Var[f]
        E2f = E2fi
        V = Ef2 - E2f
        # print(Ef2, E2f)
        if V < 1e-4:
            print("Warning: variance is", V)
        # print("Variance:", V)
        return 1 - (Efi2 - E2fi) / V


def yodo(tn, probability, given=None):
    """
    For every parameter of an MRF, finds:
        - Its sensitivity value
        - Its vertex proximity
        - Its second derivative
        - The highest derivative (in absolute value) in the interval [0, 1]

    :param tn: a `pgmpy.models.BayesianNetwork`
    :param probability: a dictionary with one key-value pair: {variable: integer value}
    :param given: [optional] a dictionary for all conditional evidence {variable: integer value}
    :return: a dictionary {tuple of names: dict}, where dict contains:
        - 'cpt': the original parameters
        - 'derivative'
        - 'sensitivity_value'
        - 'proximity'
        - 'second_derivative'
        - 'largest_first_derivative'
    """

    def evidence_tensors(tn, evidence):
        tensors = []
        for k, v in evidence.items():
            weight = torch.zeros(tn.ind_size(k))
            weight[v] = 1
            tensors.append(qtn.Tensor(weight, inds=[k]))
        return qtn.TensorNetwork(tensors)

    if given is None:
        # Marginal probability case: the function of interest if P(Y_O = y_O)
        given = {}
    # Else, conditional probability case: the function of interest if P(Y_O = y_O | y_E =
    # y_E)

    numerator = tn.copy(deep=True) | evidence_tensors(tn, {**probability, **given})
    denominator = tn.copy(deep=True) | evidence_tensors(tn, given)

    numerator.apply_to_arrays(lambda x: x.requires_grad_())
    denominator.apply_to_arrays(lambda x: x.requires_grad_())

    # return numerator
    out_numerator = numerator.contract(output_inds=[])
    out_numerator.backward()
    out_denominator = denominator.contract(output_inds=[])
    out_denominator.backward()

    result = {}
    for i in range(len(tn.tensors)):

        f = tn.tensors[i].data.to(torch.float64)
        eps = 1e-9  # CPT entries equal to 0 break the proportional covariation formula, so we perturb them
        if f.max() > 1 - eps:
            f = f * (1 - 2 * eps) + eps

        def get_coefficients(graph, i, out, evidence):
            """
            Find the coefficients a, b for the straight line y = a*theta + b
            """

            grad = graph.tensors[i].data.grad

            intersection = set(evidence).intersection(set(graph.tensors[i].inds))
            if len(intersection) > 0:  # Factor not affected by the evidence
                idx = [slice(None) for n in range(graph.tensors[i].data.dim())]
                names = list(graph.tensors[i].inds)
                for ev in intersection:
                    idx[names.index(ev)] = evidence[ev]
                    # names.remove(ev)

                mask = torch.zeros_like(graph.tensors[i].data)
                mask[tuple(idx)] = 1
                grad *= mask

            num = -torch.sum(grad * f, dim=0, keepdim=True) + grad * f
            denom = 1 - f
            grad = grad + num / denom
            a = grad
            b = out.item() - f * a
            return a, b

        if given is None:
            given = {}

        # Numerator coefficients (c1 and c2)
        c1, c2 = get_coefficients(
            numerator, i, out_numerator, evidence={**probability, **given}
        )

        # Denominator coefficients (c3 and c4)
        c3, c4 = get_coefficients(denominator, i, out_denominator, evidence=given)

        # Compute f'(\theta_i)
        derivative = (c1 * c4 - c2 * c3) / (f * c3 + c4) ** 2

        # Compute vertex proximity (Der Gaag et al., "Sensitivity analysis of probabilistic networks", 2007)
        s = -c4 / c3
        t = c1 / c3
        r = c2 / c3 + s * t
        vertex = (s < 0) * (s + torch.sqrt(torch.abs(r))) + (s > 0) * (
            s - torch.sqrt(torch.abs(r))
        )
        proximity = torch.abs(f - vertex)

        # Compute second derivative
        second = (
            2 * c3 * (-c1 + c3 * (c1 * f + c2) / (c3 * f + c4)) / (c3 * f + c4) ** 2
        )

        # Compute the largest |f'| in the interval [0, 1] (can be infinite if the hyperbola is centered inside the interval)
        max_first = torch.maximum(
            (torch.abs(c1 * c4 - c2 * c3) / c4**2).rename(None),
            (torch.abs(c1 * c4 - c2 * c3) / (c3 + c4) ** 2).rename(None),
        )
        infinity = torch.logical_and(-c4 / c3 > 0, -c4 / c3 < 1).rename(None)
        max_first[infinity] = float("inf")

        # Add obtained sensitivity values for this factor to the result dictionary
        result[tuple(tn.tensors[i].inds)] = {
            "cpt": f.detach(),
            "derivative": derivative.detach(),
            "sensitivity_value": torch.abs(derivative).detach(),
            "proximity": proximity.detach(),
            "second_derivative": second.detach(),
            "largest_first_derivative": max_first.detach(),
        }
    return result


def top_sensitivity_values(tn: qtn.TensorNetwork, probability, evidence):
    """
    Return the network's values and sensitivity values in a DatFrame
    """

    tmp = yodo(tn, probability=probability, given=evidence)

    df = pd.DataFrame(
        columns=[
            "name",
            "child",
            "coordinates",
            "value",
            "sensitivity_value",
            "variance_component",
            "total_index",
        ]
    )
    df.set_index("name", inplace=True)

    coord_columns = []
    for i in range(len(tn.tensors)):
        t = tn.tensors[i]
        v = tmp[t.inds]
        sv = v["sensitivity_value"]
        best = np.argmax(sv, axis=0).flatten().numpy()
        if v["cpt"].ndim > 1:
            coords = np.unravel_index(
                np.arange(np.prod(v["cpt"].shape[1:])), v["cpt"].shape[1:]
            )
            coords = np.array(coords).T
        else:
            coords = np.zeros([1, 0])
        assert len(best) == len(coords)
        for j in range(len(coords)):
            conditions = ", ".join(
                [f"{t.inds[k]}={coords[j, k-1]}" for k in range(1, len(t.inds))]
            )
            if len(conditions) > 0:
                conditions = " | " + conditions
            name = f"P({t.inds[0]}={best[j]}" + conditions + ")"
            df.loc[name, "child"] = t.inds[0]
            coordinates = tuple([best[j]] + list(coords[j]))
            coord_columns.append(np.array(coordinates))
            df.loc[name, "value"] = t.data[coordinates].item()
            df.loc[name, "sensitivity_value"] = v["sensitivity_value"][
                coordinates
            ].item()
    df["coordinates"] = coord_columns
    return df


def cond_index(tn: qtn.TensorNetwork, input, output, values) -> float:
    assert tn.inds_size([input]) == 2
    # f = qtn.TensorNetwork([qtn.Tensor(values, inds=[output])])
    joint = tn.contract(output_inds=[input, output]).data
    denominator = np.sum(joint, axis=1)
    conds = np.einsum("ij,j->i", joint, values) / denominator
    return conds[1] - conds[0]
