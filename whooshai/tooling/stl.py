class NIterator(object):
    __slots__ = ("_is_next", "_the_next", "it")

    def __init__(self, it):
        self.it = iter(it)
        self._is_next = None
        self._the_next = None

    def has_next(self) -> bool:
        if self._is_next is None:
            try:
                self._the_next = next(self.it)
            except:
                self._is_next = False
            else:
                self._is_next = True
        return self._is_next

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._is_next:
            response = self._the_next
        else:
            response = next(self.it)
        self._is_next = None
        return response


def higher_bound(a, key, fn=None):
    """
    The sorted array `a` has the comparability property, allowing for quick location of the desired element `key` in logarithmic time.
    Provide `fn` callable function serving as the comparator on top of `a`.
    param: a - an array of comparable keys
    param: key - an object of the same type as all the elements of `a`
    returns: the index `idx` of the "left most" element of the `a` such that `a[idx] vs key` is still valid
    """
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        m = lo + ((hi - lo) >> 1)
        pivot = a[m]
        check = fn(pivot, key) if fn is not None else pivot >= key
        if check > 0:  # To keep the constraint: j > hi => a[j] >= key
            hi = m - 1
        else:  # To keep constraint: i < lo => a[i] < key
            lo = m + 1
    # In the end of the day - lo = hi + 1 => a[hi] < key (since hi < lo) and a[lo] >= key (since lo > hi)
    return lo


def lower_bound(a, key, fn=None):
    """
    The sorted array `a` has the comparability property, allowing for quick location of the desired element `key` in logarithmic time.
    Provide `fn` callable function serving as the comparator on top of `a`.
    param: a - an array of comparable keys
    param: key - an object of the same type as all the elements of `a`
    returns: the index `idx` of the "right most" element of the `a` such that `a[idx] vs key` is still valid
    """
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        m = lo + ((hi - lo) >> 1)
        pivot = a[m]
        check = fn(pivot, key) if fn is not None else pivot <= key
        if check > 0:  # To keep the constraint: i < lo => a[i] <= key
            lo = m + 1
        else:  # To keep the constraint: j > hi => a[j] > key
            hi = m - 1
    # In the end of the day - lo = hi + 1 => a[hi] <= key (since hi < lo) and a[lo] > key (since lo > hi)
    return hi


__all__ = ["NIterator", "higher_bound", "lower_bound"]
