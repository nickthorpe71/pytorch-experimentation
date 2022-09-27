def iter_fact(n):
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    return fact


print(iter_fact(5))


def rec_fact(n):
    if n == 1:
        return n
    else:
        temp = rec_fact(n-1)
        temp = temp * n
    return temp


print(rec_fact(5))
