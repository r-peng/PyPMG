# helper fxns
def gosper_next(x: int) -> int:
    """
        Return next higher integer with same number of 1-bits.
        If overflow (no next within the same bit-length), result will exceed limit.
    """
    c = x & -x
    r = x + c
    return (((r ^ x) >> 2) // c) | r
def all_dets_bitwise(n_orb: int, n_e: int):
    """
        Yield all determinants (bitstrings) with exactly n_e electrons
        in n_orb orbitals, as Python ints.
    """
    if n_e < 0 or n_e > n_orb:
        return
    if n_e == 0:
        yield 0
        return

    x = (1 << n_e) - 1              # e.g. 000111
    limit = 1 << n_orb              # stop when we exceed n_orb bits

    while x < limit:
        yield x
        x = gosper_next(x)
def interleave_up_bits(x: int) -> int:
    """
        Map compact up bitstring x (bit i = site i) to global determinant bits (2*i).
    """
    det = 0
    while x:
        lsb = x & -x
        i = lsb.bit_length() - 1
        det |= 1 << (2*i)
        x ^= lsb
    return det
def interleave_dn_bits(x: int) -> int:
    """
        Map compact up bitstring x (bit i = site i) to global determinant bits (2*i+1).
    """
    det = 0
    while x:
        lsb = x & -x
        i = lsb.bit_length() - 1
        det |= 1 << (2*i+1)
        x ^= lsb
    return det
def get_all_configs_u1(nsites,nelecs):
    return all_dets_bitwise(nsites,nelecs)
if __name__=='__main__':
    # example
    for det in all_dets_bitwise(6, 3):
        print(f"{det:06b}", det)
