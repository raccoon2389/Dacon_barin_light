def solution(A, K):
    B = list(A)
    l = len(A)
    r = K%l
    print*i
    for i in range(l):
        t =r
        while(t+r>l):
            t =r-l
        B[i] = A[r+i]
    return B

print(solution([1,2,3],1))