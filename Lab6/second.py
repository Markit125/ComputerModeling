import numpy as np
from scipy.optimize import linprog

# Столбцы: Квартира 1, Квартира 2
payoff_matrix = np.array([
    [12, 3],
    [7,  9]
])

print("Матрица слышимости (выигрышей для Разведчика):")
print(payoff_matrix)
print("-" * 50)

# --- Решение для Разведчика (Игрок-Строка, Максимизатор) ---
# Переменные: p1 (вероятность Устройства 1), p2 (вероятность Устройства 2), V (значение игры)
# Цель: Максимизировать V, что эквивалентно минимизации -V

# Коэффициенты целевой функции (для минимизации -V)
# c = [0, 0, -1] для [p1, p2, V]
c_fox = np.array([0, 0, -1])

# Ограничения-неравенства A_ub @ x <= b_ub
# (12*p1 + 7*p2 >= V)  =>  (-12*p1 - 7*p2 + V <= 0)
# (3*p1 + 9*p2 >= V)   =>  (-3*p1 - 9*p2 + V <= 0)
A_ub_fox = np.array([
    [-payoff_matrix[0, 0], -payoff_matrix[1, 0], 1],  # -12p1 - 7p2 + V <= 0
    [-payoff_matrix[0, 1], -payoff_matrix[1, 1], 1]   # -3p1 - 9p2 + V <= 0
])
b_ub_fox = np.array([0, 0])

# Ограничения-равенства A_eq @ x == b_eq
# p1 + p2 = 1
A_eq_fox = np.array([[1, 1, 0]])
b_eq_fox = np.array([1])

# Границы для переменных: 0 <= p1 <= 1, 0 <= p2 <= 1, V может быть любым (но >=0 в данном контексте)
bounds_fox = [(0, 1), (0, 1), (None, None)] # p1, p2 в [0,1], V без верхних границ (по факту V >=0)

result_fox = linprog(c_fox, A_ub=A_ub_fox, b_ub=b_ub_fox,
                     A_eq=A_eq_fox, b_eq=b_eq_fox, bounds=bounds_fox,
                     method='highs') # 'highs' - рекомендуемый метод для linprog

if result_fox.success:
    optimal_p1 = result_fox.x[0]
    optimal_p2 = result_fox.x[1]
    value_of_game_fox = -result_fox.fun # negate because we minimized -V
    print("Результаты для Разведчика (Оптимальная стратегия для максимизации слышимости):")
    print(f"Вероятность использования Устройства 1 (p1): {optimal_p1:.4f}")
    print(f"Вероятность использования Устройства 2 (p2): {optimal_p2:.4f}")
    print(f"Ожидаемая слышимость (Значение игры V): {value_of_game_fox:.4f}")
else:
    print("Не удалось найти оптимальное решение для Разведчика:", result_fox.message)

print("-" * 50)

# --- Решение для Шпиона (Игрок-Столбец, Минимизатор) ---
# Переменные: q1 (вероятность Квартиры 1), q2 (вероятность Квартиры 2), V (значение игры)
# Цель: Минимизировать V

# Коэффициенты целевой функции (для минимизации V)
# c = [0, 0, 1] для [q1, q2, V]
c_spy = np.array([0, 0, 1])

# Ограничения-неравенства A_ub @ x <= b_ub
# (12*q1 + 3*q2 <= V)   => (12*q1 + 3*q2 - V <= 0)
# (7*q1 + 9*q2 <= V)    => (7*q1 + 9*q2 - V <= 0)
A_ub_spy = np.array([
    [payoff_matrix[0, 0], payoff_matrix[0, 1], -1],  # 12q1 + 3q2 - V <= 0
    [payoff_matrix[1, 0], payoff_matrix[1, 1], -1]   # 7q1 + 9q2 - V <= 0
])
b_ub_spy = np.array([0, 0])

# Ограничения-равенства A_eq @ x == b_eq
# q1 + q2 = 1
A_eq_spy = np.array([[1, 1, 0]])
b_eq_spy = np.array([1])

# Границы для переменных: 0 <= q1 <= 1, 0 <= q2 <= 1, V может быть любым (но >=0)
bounds_spy = [(0, 1), (0, 1), (None, None)] # q1, q2 в [0,1], V без верхних границ (по факту V >=0)

result_spy = linprog(c_spy, A_ub=A_ub_spy, b_ub=b_ub_spy,
                     A_eq=A_eq_spy, b_eq=b_eq_spy, bounds=bounds_spy,
                     method='highs')

if result_spy.success:
    optimal_q1 = result_spy.x[0]
    optimal_q2 = result_spy.x[1]
    value_of_game_spy = result_spy.fun # here we minimized V directly
    print("Результаты для Шпиона (Оптимальная стратегия для минимизации слышимости):")
    print(f"Вероятность выбора Квартиры 1 (q1): {optimal_q1:.4f}")
    print(f"Вероятность выбора Квартиры 2 (q2): {optimal_q2:.4f}")
    print(f"Ожидаемая слышимость (Значение игры V): {value_of_game_spy:.4f}")
else:
    print("Не удалось найти оптимальное решение для Шпиона:", result_spy.message)

print("-" * 50)
print("Проверка: Значения игры для Разведчика и Шпиона должны совпадать.")