"""maxmatchstat.py -- functions for computing maximum matching of communities"""

from networkx.algorithms import bipartite
from scipy.sparse import coo_matrix
from typing import Any
import numpy as np
from typing import Any, Set, List, Dict, Tuple

from scipy.optimize import linear_sum_assignment


def jaccard_matrix(true_comms, pred_comms):
    M = np.zeros((len(true_comms), len(pred_comms)))

    for i, C in enumerate(true_comms):
        for j, F in enumerate(pred_comms):
            if len(C | F) > 0:
                M[i, j] = len(C & F) / len(C | F)

    return M


def max_jaccard_matching(true_comms, pred_comms):
    M = jaccard_matrix(true_comms, pred_comms)

    # maximize → convert to cost
    cost = 1 - M

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = [(i, j, M[i,j]) for i, j in zip(row_ind, col_ind)]

    return np.mean([score for _,_,score in matches])


# def max_match_communities_old(A: list[set[Any]], B: list[set[Any]]) -> dict[int, int]:
#     """Do maximum matching of communities
#
#     Parameters:
#         A, B    Communities given as lists of sets of vertices
#
#     Output:
#         match   dictionary giving the mapping from A to B s.t. A[i] is
#                 mapped to B[match[i]]. If len(B) < len(A), match[i] is None
#                 for the communities of A that are not mapped to communities of B.
#     """
#     nA = len(A)
#     nB = len(B)
#     edgelist = []
#     for i in range(nA):
#         commA = A[i]
#         for j in range(nB):
#             commB = B[j]
#             # Add edge if commA and commB intersect and set negative
#             # Jaccard for weight, so that min_matching finds the
#             # max Jaccard. Vertices are numbered 0 to nA-1 for A and
#             # nA to nA+nB-1 for C.
#             c = len(commA.intersection(commB))
#             if c > 0:
#                 u = len(commA.union(commB))
#                 edgelist.append(f"{i} {nA+j} {-c/u}")
#     G = bipartite.parse_edgelist(edgelist, nodetype=int, data=(("weight", float),))
#     orig_match = bipartite.minimum_weight_full_matching(G)
#     # Correct indices of values by subtracting nA
#     match = {i: orig_match[i] - nA for i in range(nA) if i in orig_match}
#
#     return match

def max_match_communities(A, B):
    nA = len(A)
    nB = len(B)

    size = max(nA, nB)
    cost = np.ones((size, size))  # high cost for non-match

    for i in range(nA):
        for j in range(nB):
            inter = len(A[i] & B[j])
            if inter > 0:
                union = len(A[i]) + len(B[j]) - inter
                cost[i, j] = 1 - inter / union  # convert Jaccard to cost

    row_ind, col_ind = linear_sum_assignment(cost)

    match = {}
    for i, j in zip(row_ind, col_ind):
        if i < nA:
            if j < nB and cost[i, j] <= 1:
                match[i] = j
            else:
                match[i] = None

    return match


def mm_stats(A: List[Set[Any]],
             B: List[Set[Any]],
             matching: Dict[int, int],
             beta: float = 1) -> Tuple[float, float, float]:
    """
    Compute average precision, recall, and F-score given maximum matching of communities.

    Parameters
    ----------
    A : List[Set[Any]]
        List of sets of vertices for true communities
    B : List[Set[Any]]
        List of sets of vertices for found communities
    matching : Dict[int, int]
        Dictionary mapping indices of A to indices of B (may contain None)
    beta : float, optional
        Parameter for F-score. Default is 1 (F1-score)

    Returns
    -------
    Tuple[float, float, float]
        (average precision, average recall, average F-beta score)
    """
    betaSq = beta**2
    prec_sum = 0.0
    recall_sum = 0.0
    f_sum = 0.0
    match_count = 0

    for i, j in matching.items():
        if j is None:
            continue  # skip unmatched communities

        commA = A[i]
        commB = B[j]

        cap_size = len(commA & commB)
        if cap_size == 0:
            continue  # skip zero overlap

        prec_comm = cap_size / len(commB)
        recall_comm = cap_size / len(commA)
        f_comm = ((1 + betaSq) * prec_comm * recall_comm) / ((betaSq * prec_comm) + recall_comm)

        prec_sum += prec_comm
        recall_sum += recall_comm
        f_sum += f_comm
        match_count += 1

    if match_count == 0:
        return 0.0, 0.0, 0.0

    return prec_sum / match_count, recall_sum / match_count, f_sum / match_count


def mm_stats_old(A: list[set[Any]],
             B: list[set[Any]],
             matching: dict[int, int],
             beta: float = 1) -> tuple[float, float, float]:
    """Compute average precision, recall, and F-score given maximum matching of communities

    Computes the precision, recall, and F-score between communities in A (assumed the truth)
    and communities in B (assumed estimated) given the matching. Results are returned in
    3-tuple and normalized by the number of matches (i.e. min(len(A), len(B))).

    The macro F-score is the average of F-scores, not F-score based on average precision
    and recall. Default is to compute F1-score, but setting beta computes other F-scores.

    Parameters:
    A       List of sets of vertices for true communities
    B       List of sets of vertices for found communities
    matching    Dictionary from indices of A to indices of B
    beta    Parameter for F-score. Default = 1.

    Output:
    stats   3-tuple that has average precision, average recall, and average F-beta score,
            in that order.
    """
    nA = len(A)
    nB = len(B)
    prec: float = 0.0
    recall: float = 0.0
    F_score: float = 0.0
    betaSq = 1 if beta == 1 else beta**2
    for i in range(nA):
        if i in matching:
            commA = A[i]
            # if matching[i] is None:
            #     continue

            commB = B[matching[i]]
            cap_size = len(commA.intersection(commB))
            prec_comm = cap_size/len(commB)
            recall_comm = cap_size/len(commA)
            F_score_comm = ((1 + betaSq)*prec_comm*recall_comm)/((betaSq * prec_comm) + recall_comm)
            prec += prec_comm
            recall += recall_comm
            F_score += F_score_comm
    match_size = len(matching)
    assert(match_size == min(nA, nB))
    return (prec/match_size, recall/match_size, F_score/match_size)


def maximum_matching_stats(true_comm: List[Set[Any]],
                           found_comm: List[Set[Any]],
                           beta: float = 1) -> tuple[float, float, float]:
    """Computes maximum matching and returns average precision, recall, and F1

    Given true communities and found communities, computes a maximum matching between them
    and then average precision, average recall, and average F-score. F-score is average
    of communities' F-scores. By default, F1 is computed, but parameter beta controls
    the type of F-score. 

    Returns a tuple (avg_precision, avg_recall, avg_f_score)
    """

    matching = max_match_communities(true_comm, found_comm)
    return mm_stats(true_comm, found_comm, matching, beta)

