




def factorial(n):
    '''
    阶乘
    '''
    if n==0 or n==1:
        return 1
    else:
        return n*factorial(n-1)

