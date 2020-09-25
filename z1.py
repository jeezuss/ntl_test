def findMaxSubArray(a):
    max_sum = 0
    i = ln = len(a)
    for num in a:  # вычисляем сумму всего массива, считаем максимальное значение
        max_sum += num
    perm = []
    i = i - 1
    res_a = a.copy()  # сохраняем исходный массив как самый длинный
    # перебираем все возможные непрерывные массивы, вычисляем их длину и сравниваем с максимальной
    # если находим сумму большую чем максимальную, запоминаем сумму и массив
    while i != 0:
        for y in range(ln - i + 1):
            perm.clear()
            sum = 0
            for x in range(0 + y, i + y):
                perm.append(a[x])
            for n in perm:
                sum += n
            if sum > max_sum:
                max_sum = sum
                res_a = perm.copy()
            #print(perm)
            #print(sum)
        i -= 1
    return res_a, max_sum  # выводим массив и сумму

print('Введите числа массива в одну строчку через запятую без пробелов. Например, -2,1,-3,4,-1,2,1,-5,4:')
data_in = input()
a = []
while data_in:
    z = data_in.find(',')
    if z == -1:
        a.append(int(data_in))
        data_in = ''
    else:
        a.append(int(data_in[:z]))
        data_in = data_in[z + 1:]
res_a, max_sum = findMaxSubArray(a)
print('Массив: {}. Максимальная сумма: {}'.format(res_a, max_sum))
