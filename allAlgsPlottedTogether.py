import random
import time
import numpy as np
import matplotlib.pyplot as plt

randomSeed = []
sortedSeed = []
backwardsSeed = []
partlySortedSeed = []

def select_sort(arr):
    for i in range(len(arr) - 1):
        min_index = i
        for k in range(i + 1, len(arr)):
            if arr[k] < arr[min_index]:
                min_index = k
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

slowSortStep = 250
slowSortLow = 1000
slowSortHigh = 3500

timeResultForRandom = []
n = []
for i in range(slowSortLow, slowSortHigh, slowSortStep):
    n.append(i)
    seed = time.time()
    random.seed(seed)
    randomSeed.append(seed)
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        select_sort(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))

n = np.array(n)

curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='red')
plt.title('Квадратичные сортировки')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.grid()
plt.plot(n, np.polyval(curveRandom, n), label = 'Сортировка выбором', color='red')

#---------------------------------------------------------------------

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
n = []
timeResultForRandom = []
for i in range(slowSortLow, slowSortHigh, slowSortStep):
    n.append(i)
    random.seed(randomSeed[(i - slowSortLow) // slowSortStep])
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        insertion_sort(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))

n = np.array(n)
curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='green')
plt.title('Квадратичные сортировки')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.grid()
plt.plot(n, np.polyval(curveRandom, n), label = 'Сортировка вставками', color='green')

 #---------------------------------------------------------------------

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

timeResultForRandom = []
n = []
for i in range(slowSortLow, slowSortHigh, slowSortStep):
    n.append(i)
    random.seed(randomSeed[(i - slowSortLow) // slowSortStep])
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        bubble_sort(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))


n = np.array(n)
curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='blue')
plt.title('Квадратичные сортировки')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.grid()
plt.plot(n, np.polyval(curveRandom, n), label = 'Сортировка пузырьком', color='blue')

plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------

randomSeed = []
sortedSeed = []
backwardsSeed = []
partlySortedSeed = []

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        merge_sort(left_half)
        merge_sort(right_half)
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr

fastSortsStep = 10000
fastSortsLow = 50000
fastSortsHigh = 110000

timeResultForRandom = []
n = []
for i in range(fastSortsLow, fastSortsHigh, fastSortsStep):
    n.append(i)
    seed = time.time()
    random.seed(seed)
    randomSeed.append(seed)
    time.sleep(0.1)
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        merge_sort(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))

n = np.array(n)

curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='red')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.plot(n, np.polyval(curveRandom, n), label = 'Сортировка слиянием', color='red')


# #---------------------------------------------------------------------

def shell_sort_shell(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

timeResultForRandom = []
n = []
for i in range(fastSortsLow, fastSortsHigh, fastSortsStep):
    n.append(i)
    random.seed(randomSeed[(i - fastSortsLow) // fastSortsStep])
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        shell_sort_shell(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))

n = np.array(n)
curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='blue')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.plot(n, np.polyval(curveRandom, n), label = 'Сортировка Шелла', color='blue')

#------------------------------------------------------------------

def hibbard_sequence(n):
    return [2**k - 1 for k in range(1, n) if 2**k - 1 < n][::-1]

def shell_sort_hibbard(arr):
    n = len(arr)
    gap_sequence = hibbard_sequence(n)
    for gap in gap_sequence:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
    return arr

fastSortsStep = 1000
fastSortsLow = 5000
fastSortsHigh = 11000

timeResultForRandom = []
n = []
for i in range(fastSortsLow, fastSortsHigh, fastSortsStep):
    n.append(i)
    random.seed(randomSeed[(i - fastSortsLow) // fastSortsStep])
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        shell_sort_hibbard(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))

n = np.array(n)
curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='gold')
plt.title('Сортировка слиянием')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.plot(n, np.polyval(curveRandom, n), label = 'Случайный массив', color='gold')

#--------------------------------------------------------------------

def pratt_sequence(n):
    sequence = []
    i = 0
    while True:
        for j in range(i + 1):
            gap = 2**i * 3**j
            if gap < n:
                sequence.append(gap)
            else:
                return sorted(sequence)[::-1]
        i += 1

def shell_sort_pratt(arr):
    n = len(arr)
    gap_sequence = pratt_sequence(n)
    for gap in gap_sequence:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
    return arr

timeResultForRandom = []
n = []
for i in range(fastSortsLow, fastSortsHigh, fastSortsStep):
    n.append(i)
    random.seed(randomSeed[(i - fastSortsLow) // fastSortsStep])
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        shell_sort_pratt(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne) / len(timeResultForOne)), 4))

n = np.array(n)
curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='orange')
plt.title('Сортировка слиянием')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.plot(n, np.polyval(curveRandom, n), label = 'Случайный массив', color='orange')

#-----------------------------------------------------------------------
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

fastSortsStep = 10000
fastSortsLow = 50000
fastSortsHigh = 110000
timeResultForRandom = []
n = []
for i in range(fastSortsLow, fastSortsHigh, fastSortsStep):
    n.append(i)
    random.seed(randomSeed[(i - fastSortsLow) // fastSortsStep])
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        quicksort(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))

n = np.array(n)
curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='green')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.plot(n, np.polyval(curveRandom, n), label = 'Быстрая сортировка', color='green')

#----------------------------------------------------------------------------

def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])
        heapify(arr, n, largest)
def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])
        heapify(arr, i, 0)

timeResultForRandom = []
n = []
for i in range(fastSortsLow, fastSortsHigh, fastSortsStep):
    n.append(i)
    random.seed(randomSeed[(i - fastSortsLow) // fastSortsStep])
    randomArray = [random.randint(1, i) for _ in range(i)]
    timeResultForOne = []
    for j in range(10):
        beingSortedArray = randomArray.copy()
        time_start = time.time()
        heap_sort(beingSortedArray)
        time_end = time.time()
        timeResultForOne.append(time_end - time_start)
    timeResultForRandom.append(round((sum(timeResultForOne)/len(timeResultForOne)), 4))

n = np.array(n)
curveRandom = np.polyfit(n, timeResultForRandom, 2)
plt.scatter(n, timeResultForRandom, color='yellow')
plt.xlabel('Размер входных данных (n)')
plt.ylabel('Время выполнения (с)')
plt.plot(n, np.polyval(curveRandom, n), label = 'Пирамидальная сортировка', color='yellow')

plt.title('Неквадратичные сортировки')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()