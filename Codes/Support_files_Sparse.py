from collections import defaultdict
import numpy as np
from collections import Counter

import random


def sample_topk_percent_per_label( vertices, labels, A, percentage=0.2):

    """
    Select top-k% nodes per community
    based on internal degree.

    A: scipy.sparse CSR adjacency matrix
    percentage: e.g. 0.2 = top 20%
    """

    # ------------------------------------------------------------
    # Group vertices according to community
    # ------------------------------------------------------------
    label_to_vertices = defaultdict(set)

    for v, l in zip(vertices, labels):
        label_to_vertices[l].add(v)


    sampled_lists = []

    # ------------------------------------------------------------
    # Process each community
    # ------------------------------------------------------------
    for verts in label_to_vertices.values():

        verts = np.fromiter( verts, dtype=np.int64)

        # --------------------------------------------------------
        # Extract sparse adjacency submatrix for this community
        # --------------------------------------------------------
        A_comm = A[verts][:, verts]

        # --------------------------------------------------------
        # Internal degree
        #
        # --------------------------------------------------------
        internal_degree = A_comm.getnnz(axis=1)

        # --------------------------------------------------------
        # Number of nodes to select
        # --------------------------------------------------------
        k = max(int(len(verts) * percentage),1)


        # --------------------------------------------------------
        # Select nodes with highest internal degree
        # --------------------------------------------------------
        if len(verts) >= k:

            idx = np.argpartition(-internal_degree, k - 1)[:k]
            sampled = verts[idx]

        else:
            sampled = verts

        sampled_lists.append( sampled.tolist())

    return sampled_lists



def random_walk_seed_percent(G, start, walk_length, num_walks, k_percent=0.1):

    counts = Counter()

    for _ in range(num_walks):

        v = start
        counts[v] += 1

        for _ in range(walk_length):

            nbrs = list(G.neighbors(v))

            if len(nbrs) == 0:
                break

            v = random.choice(nbrs)
            counts[v] += 1

    top_k = max( 1, int(np.ceil(k_percent * walk_length )  ) ) #max( 1, int(np.ceil(k_percent * len(counts))) )
    seed = {v for v, _ in counts.most_common(top_k)}

    return seed


def collect_random_walk_seeds_precent_v1(G, seed_groups, walk_length=100, num_walks=50, top_k=20):

    random_walk_seeds = []

    for group in seed_groups:

        # Remove duplicates
        group = list(dict.fromkeys(group))

        if len(group) == 0:
            continue

        # Select one starting vertex from this seed group
        random_vertices = random.sample(group, min(2, len(group))) #1,3

        for random_vertex in random_vertices:
            random_walk = random_walk_seed_percent(G, random_vertex, walk_length, num_walks, top_k)

            random_walk_seeds.append(list(random_walk))

    return random_walk_seeds



def collect_random_walk_seeds_precent_v0(G, num_seeds, walk_length=100, num_walks=50,top_k=20):

    nodes = list(G.nodes())

    # Select different starting vertices without replacement
    random_vertices = random.sample(nodes,min(num_seeds, len(nodes)))
    random_walk_seeds = []

    for random_vertex in random_vertices:

        random_walk = random_walk_seed_percent(G, random_vertex, walk_length, num_walks,top_k)
        random_walk_seeds.append(list(random_walk))

    return random_walk_seeds

def select_random_walk_seeds_Jaccard_v_avg(seeds, n_select):

    seeds = [set(s) for s in seeds]

    # Remove duplicates
    unique_seeds = []
    seen = set()

    for s in seeds:
        key = frozenset(s)

        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    seeds = unique_seeds

    if len(seeds) == 0 or n_select <= 0:
        return []

    # First seed selected randomly
    selected = [random.choice(seeds)]

    remaining = [ s for s in seeds if s != selected[0] ]

    while len(selected) < min(n_select, len(seeds)):

        best_seed = None
        best_score = float("inf")

        for S in remaining:

            similarities = [ len(S & T) / len(S | T) for T in selected ]

            # Average Jaccard similarity
            score = sum(similarities) / len(similarities)

            # Smaller average similarity = better
            if score < best_score:
                best_score = score
                best_seed = S

        selected.append(best_seed)
        remaining.remove(best_seed)

    return [list(s) for s in selected]


def select_random_walk_seeds_Jaccard(seeds, n_select):

    seeds = [set(s) for s in seeds]

    # Remove duplicates
    unique_seeds = []
    seen = set()

    for s in seeds:
        key = frozenset(s)

        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    seeds = unique_seeds

    if len(seeds) == 0 or n_select <= 0:
        return []

    # Select first seed randomly
    selected = [random.choice(seeds)]

    remaining = [s for s in seeds if s != selected[0]]

    while len(selected) < min(n_select, len(seeds)):

        best_seed = None
        best_score = float("inf")

        for S in remaining:

            similarities = [ len(S & T) / len(S | T) for T in selected]

            # Maximum similarity with any selected seed
            score = max(similarities)

            # Smaller maximum similarity = more diverse
            if score < best_score:
                best_score = score
                best_seed = S

        selected.append(best_seed)
        remaining.remove(best_seed)

    return [list(s) for s in selected]


def select_max_random_walk_seeds(seeds, n_select):

    seeds = [set(s) for s in seeds]

    # Remove duplicate seed sets
    unique_seeds = []
    seen = set()

    for s in seeds:
        key = frozenset(s)

        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    seeds = unique_seeds

    if len(seeds) == 0 or n_select <= 0:
        return []

    # First seed is selected randomly
    selected = [random.choice(seeds)]

    remaining = [ s for s in seeds if s != selected[0] ]

    while len(selected) < min(n_select, len(seeds)):

        best_seed = None
        best_score = float("inf")

        for S in remaining:

            # Minimum Jaccard similarity with
            # any already selected seed
            score = min(len(S & T) / len(S | T) for T in selected )

            # Select the candidate with the
            # smallest minimum similarity
            if score < best_score:
                best_score = score
                best_seed = S

        selected.append(best_seed)
        remaining.remove(best_seed)

    return [list(s) for s in selected]

def select_max_random_walk_seeds_ori(seeds, n_select):

    seeds = [set(s) for s in seeds]

    # Step 1:pick a random seed
    selected = [random.choice(seeds)]

    remaining = [s for s in seeds if s != selected[0]]

    while len(selected) < min(n_select, len(seeds)):

        best_seed = None
        best_score = float("inf")

        for S in remaining:

            # similarity to closest selected seed
            score = min(
                len(S & T) / len(S | T)
                for T in selected
            )

            if score < best_score:
                best_score = score
                best_seed = S

        selected.append(best_seed)
        remaining.remove(best_seed)

    return [list(s) for s in selected]


def grow_density_quasi_clique(G,start_node,gamma=0.7):

    Q = {start_node}

    E_Q = 0

    changed = True

    while changed:

        changed = False

        candidates = set()

        for u in Q:
            candidates |= set(G.neighbors(u))

        candidates -= Q

        best_v = None
        best_density = -1

        n_Q = len(Q)

        for v in candidates:

            # edges from v into Q
            dQv = len(set(G.neighbors(v)) & Q)

            n_new = n_Q + 1

            E_new = E_Q + dQv

            density = (
                2 * E_new
                /
                (n_new * (n_new - 1))
            )

            if density >= gamma:

                if density > best_density:

                    best_density = density
                    best_v = v
                    best_edges = E_new

        if best_v is not None:

            Q.add(best_v)

            E_Q = best_edges

            changed = True

    return Q


def remove_duplicates(communities, overlap_thresh=0.9):

    filtered = []

    for Q in communities:

        keep = True

        for D in filtered:

            overlap = len(Q & D) / min(len(Q), len(D))

            if overlap > overlap_thresh:
                keep = False
                break

        if keep:
            filtered.append(Q)

    return filtered


def find_density_quasi_clique_seeds(G, gamma=0.7, min_size=10, top_fraction=0.1):

    seeds = []

    nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)

    top_k = int(len(nodes_sorted) * top_fraction)

    for v, _ in nodes_sorted[:top_k]:

        Q = grow_density_quasi_clique(G, v, gamma)

        if len(Q) >= min_size:
            seeds.append(Q)

    return seeds
