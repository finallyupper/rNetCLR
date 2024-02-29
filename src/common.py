# NetAugment
def find_sizes(x):
    """
    (times, sizes) -> size only

    parameters
    ----
    x : nd array

    returns
    ----
    sizes : int list
    """
    sizes = x[:, 1]
    times = x[:, 0]

    return list(sizes)
