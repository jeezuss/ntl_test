a = [-2, 1, -3, 4, -1, 2, 1, -5, 4]


def findMaxSubArray(a):
    max_sum = 0
    i = ln = len(a)
    for num in a:
        max_sum += num
    sum = 0
    per = []
    i = i - 1
    res_a = a.copy()
    while i != 0:
        for y in range(ln - i + 1):
            per.clear()
            sum = 0
            for x in range(0 + y, i + y):
                per.append(a[x])
            for n in per:
                sum += n
            if sum > max_sum:
                max_sum = sum
                res_a = per.copy()
                print("!!!!!!!!!!!!")
            print(per)
            print(sum)
        i -= 1
    return res_a, max_sum


print(findMaxSubArray(a))
