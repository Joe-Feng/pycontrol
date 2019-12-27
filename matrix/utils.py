



def inverse_number(array):
    '''
    求一个排列的逆序数
    ---------------------------
    Find the permuted ordinal number
    '''
    n = 0
    for i in range(len(array)):
        for j in range(i):
            if array[j] > array[i]:
                n += 1

    return n



