import numpy as np
import matplotlib.pyplot as plt
n = np.linspace(0,100, 1000)
# t_best = n**2 + 2*n - 3
# t_avg = 5/4 * n**2 + 7/4 *n - 3
# t_worst = 3/2 * n**2 + 3/2 *n - 3
#
# plt.plot(n, t_best, label='Лучший случай')
# plt.plot(n, t_avg, label='Средний случай')
# plt.plot(n, t_worst, label='Худший случай')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Сортировка выбором')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
#
# #--------------------------------------------------------------
#
# t_best = 5*n - 4
# t_avg = 3/4 * n**2 + 15/4 *n - 7/2
# t_worst = 3/2 * n**2 + 7/2 *n - 4
#
# plt.plot(n, t_best, label='Лучший случай')
# plt.plot(n, t_avg, label='Средний случай')
# plt.plot(n, t_worst, label='Худший случай')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Сортировка вставками')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
#
# #--------------------------------------------------------------
#
# t_best = 3/2 * n**2 + 5/2 *n
# t_avg = 7/4 * n**2 + 11/4 *n
# t_worst = 2 * n**2 + 3 *n
#
# plt.plot(n, t_best, label='Лучший случай')
# plt.plot(n, t_avg, label='Средний случай')
# plt.plot(n, t_worst, label='Худший случай')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Сортировка пузырьком')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
# #--------------------------------------------------------------
#
# t = n*np.log(n)
#
# plt.plot(n, t,  label='Лучший, средний и худший случай')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Сортировка слиянием')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
# #--------------------------------------------------------------
#
# t_best = n*np.log(n)
# t_avg = n**(3/2)
# t_worst = n**2
#
# plt.plot(n, t_best, label='Лучший случай')
# plt.plot(n, t_avg, label='Средний случай')
# plt.plot(n, t_worst, label='Худший случай')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Сортировка Шелла (последовательность Шелла)')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
# #--------------------------------------------------------------
#
# t_best = n*np.log(n)
# t = n**(3/2)
#
# plt.plot(n, t_best, label='Лучший случай')
# plt.plot(n, t, label='Средний и худший случаи')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Сортировка Шелла (последовательность Хиббарда)')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
# #--------------------------------------------------------------
#
# t_best = n*np.log(n)
# t = n*(np.log(n)**2)
#
#
# plt.plot(n, t_best, label='Лучший случай')
# plt.plot(n, t, label='Средний и худший случаи')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Сортировка Шелла (последовательность Пратта)')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
# #--------------------------------------------------------------
#
# t = n*np.log(n)
# t_worst = n**2
#
# plt.plot(n, t, label='Лучший и средний случай')
# plt.plot(n, t_worst, label='Худший случай')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Быстрая сортировка')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()
#
# #--------------------------------------------------------------
#
# t = n*np.log(n)
#
# plt.plot(n, t, label='Все случаи')
#
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
#
#
# plt.title('Пирамидальная сортировка')
# plt.xlabel('Размер массива (n)')
# plt.ylabel('Количество операций')
# plt.legend()
#
# plt.show()

#--------------------------------------------------------------

t_ss = 5/4 * n**2 + 7/4 *n - 3
t_is = 3/4 * n**2 + 15/4 *n - 7/2
t_bs = 7/4 * n**2 + 11/4 *n
t_ms = n*np.log(n)
t_shell_s = n**(3/2)
t_shell_h = n**(3/2)
t_shell_p = n*(np.log(n)**2)
t_qs = n*np.log(n)
t_h = n*np.log(n)

plt.plot(n, t_ss, color = '#3357FF', label='Выбором')
plt.plot(n, t_is, color = '#8A2BE2', label='Вставками')
plt.plot(n, t_bs, color = '#FFD700', label='Пузырьком')
plt.plot(n, t_ms + 100, color = '#FF5733', label='Слиянием')
plt.plot(n, t_shell_s + 100, color = '#33FFF3', label='Шелла (последовательность Шелла)')
plt.plot(n, t_shell_h, color = '#006400', label='Шелла (последовательность Хиббарда)')
plt.plot(n, t_shell_p, color = '#FF33A1', label='Шелла (последовательность Пратта)')
plt.plot(n, t_qs - 100, color = '#33FF57', label='Быстрая')
plt.plot(n, t_h, color = '#F3FF33', label='Пирамидальная')

plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.title('Средние случаи сортировок')
plt.xlabel('Размер массива (n)')
plt.ylabel('Количество операций')
plt.legend()

plt.show()