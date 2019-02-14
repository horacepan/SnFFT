import itertools
import perm2

def perm_from_young_tuple(cyc_tup):
    n = sum(map(len, cyc_tup))
    lst = [0] * n
    idx = 0
    for cyc in cyc_tup:
        for x in cyc:
            lst[idx] = x
            idx += 1

    return perm2.Perm2.from_tup(tuple(lst))

def young_subgroup(weak_partition):
    '''
    weak_partition: tuple of ints

    Let alpha be the weak partition
    Returns the generator for the product group S_alpha_0 x S_alpha_1 x ... x S_alpha_k
    '''
    #sym_subgroups = [perm2.sn(p) for p in weak_partition if p > 0]
    sym_subgroups = [itertools.permutations(range(1, p+1)) for p in weak_partition if p > 0]
    return itertools.product(*sym_subgroups)

def young_subgroup_perm(weak_partition):
    return [perm_from_young_tuple(t) for t in young_subgroup_canonical(weak_partition)]

def young_subgroup_canonical(weak_partition):
    '''
    This is not quite right...
    '''
    subgroups = []
    idx = 1
    for p in weak_partition:
        if p == 0:
            continue
        subgroups.append(itertools.permutations(range(idx, idx+p)))
        idx += p

    #sym_subgroups = [itertools.permutations(range(1, p+1)) for p in weak_partition if p > 0]
    return itertools.product(*subgroups)
#================================================================================================

def tup_set(perms):
    return {p.tup_rep for p in perms}

def check_coset(g1, g2, H):
    '''
    Check if g1 and g2 are in the same coset
    '''
    s = set(left_coset(g1, H))
    return (g2 in s)

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
    to_visit = {g.tup_rep for g in G}
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


if __name__ == '__main__':
    G = perm2.sn(4)
    subgroup = young_subgroup_perm((2, 2))
    print(subgroup)
    for rep in coset_reps(G, subgroup):
        print('======')
        for g in left_coset(rep, subgroup):
            print(g)
