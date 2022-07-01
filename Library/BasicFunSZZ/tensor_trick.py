from collections.abc import Iterable

import torch as tc


def inverse_permutation(perm):
    # perm is a torch tensor
    if not isinstance(perm, tc.Tensor):
        perm = tc.tensor(perm)
    inv = tc.empty_like(perm)
    inv[perm] = tc.arange(perm.size(0), device=perm.device)
    return inv.tolist()


def have_same_iterable(a_list, b_list):
    if isinstance(a_list, Iterable) and isinstance(b_list, Iterable):
        xx = [x for x in a_list if x in b_list]
        if len(xx) > 0:
            return True
        else:
            return False
    else:
        return False


def is_str_same(a_str, b_str):
    flag = True
    if len(a_str) != len(b_str):
        return False
    else:
        for nn in range(len(a_str)):
            if a_str[nn] != b_str[nn]:
                flag = False
        return flag
