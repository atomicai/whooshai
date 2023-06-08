def chunkify(f, chunksize=10_000_000, sep="\n"):
    """
    Read a file separating its content lazily.

    Usage:

    >>> with open('INPUT.TXT') as f:
    >>>     for item in chunkify(f):
    >>>         process(item)
    """
    chunk = None
    remainder = None  # data from the previous chunk.
    while chunk != "":
        chunk = f.read(chunksize)
        if remainder:
            piece = remainder + chunk
        else:
            piece = chunk
        pos = None
        while pos is None or pos >= 0:
            pos = piece.find(sep)
            if pos >= 0:
                if pos > 0:
                    yield piece[:pos]
                piece = piece[pos + 1 :]
                remainder = None
            else:
                remainder = piece
    if remainder:  # This statement will be executed iff @remainder != ''
        yield remainder


__all__ = ["chunkify"]
