def get_batch_sizes(n:int, offset:int=5):
    return [
        i**2 + int(1.175**i) for i in range(offset, n+offset)
    ]