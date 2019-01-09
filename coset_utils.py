import pdb
import perm2
from wreath import young_subgroup_canonical, young_subgroup, young_subgroup_perm


def tup_set(perms):
    return set([p.tup_rep for p in perms])

def intersect(p1, p2):
    '''
    p1: list of perm2 objects
    p2: list of perm2 objects
    Return permutation objects of the intersection of the two
    '''

    p1_tups = tup_set(p1)
    p2_tups = tup_set(p2)
    return tup_set(p1_tups.intersect(p2_tups))

def check_equal(g1, g2):
    '''
    g1: list of Perm2
    g2: list of Perm2
    Returns true if g1 contains the same group elements as g2
    '''
    return set(g1) == set(g2)

def left_coset(g, H):
    return [g * h for h in H]

def right_coset(g, H):
    return [h * g for h in H]

# TODO: This is almost certainly not the most efficient way of getting the coset reps
def coset_reps(G, H):
    '''
    G: list of Perm2 objects
    H: list of Perm2 objects
    Returns a list of Perm2 objects
    '''
    reps = []
    to_visit = set([g.tup_rep for g in G])
    g_map = {g.tup_rep: g for g in G}

    for g_tup in g_map.keys():
        # grab a group element, hit each H element
        # g * h for h \in H
        for h in H:
            g = g_map[g_tup]
            gh = g*h
            if gh.tup_rep in to_visit:    
                # then all of these are good and this is a coset rep
                reps.append(g)
                for _gh in left_coset(g, H):
                    to_visit.remove(_gh.tup_rep)
            else:
                # this is a repeat and we can stop
                break
        # continue
        if len(reps) == len(G) / len(H):
            break

    return reps

def reduced_young_subgroup(alpha, idx):
    '''
    Returns the young subgroup that results from decreasing the index idx
    in the partition alpha.

    Ex:
        alpha = (2, 2)
        idx = 0
        Returns the young subgroup for the partition (1, 2)

        alpha = (2, 2)
        idx = 1
        Returns the young subgroup for the partition (2, 1)

    '''
    #aprime = tuple(a for i, a in enumerate(alpha) if i != idx else a-1)
    #return young_subgroup(aprime)
    pass

def young_subgroup_coset(n, alpha):
    # base case?
    _coset_reps = []
    ds = []
    m = len(alpha)

    for i in range(1, m):
        endpt = 0
        cyc = tuple(n - j for j in range(endpt))
        #perm = perm2.Perm2.from_cyc(cyc)
        perm = perm2.Perm2.from_tup(cyc)
        ds.append((perm, i))

    ds.append(Perm2.eye(n), m)

    for d, idx in ds:
        # idx is 1 indexed
        for p in young_subgroup_coset(n, reduced_young_subgroup(alpha, idx-1)):
            _coset_reps.append(p * d)

    return _coset_reps

if __name__ == '__main__':
    pass
