from functools import reduce


def remove_smallest(numbers):
    smallest = reduce(lambda x, y: x if x < y else y, numbers)
    res = []
    for i in range(len(numbers)):
        if numbers[i] == smallest:
            res.extend(numbers[i+1:])
            break
        else:
            res.append(numbers[i])

    return res


print(remove_smallest([1, 2, 3, 1, 4, 5]))
