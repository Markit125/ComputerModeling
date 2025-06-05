import random

def generate_matrix(rows, cols, min_val, max_val):
    return [[random.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]


def find_saddle_point(payoff_matrix) -> (tuple, bool):
    if not payoff_matrix or not payoff_matrix[0]:
        return (None, False)

    num_rows = len(payoff_matrix)
    num_cols = len(payoff_matrix[0])

    row_mins = [min(row) for row in payoff_matrix]

    col_maxs = []
    for j in range(num_cols):
        col_values = [payoff_matrix[i][j] for i in range(num_rows)]
        col_maxs.append(max(col_values))

    for i in range(num_rows):
        for j in range(num_cols):
            element = payoff_matrix[i][j]
            if element == row_mins[i] and element == col_maxs[j]:
                return ((i, j, element), True)

    return (None, False)


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end=" ")
        print()


print("--- Пример 1: Матрица с седловой точкой ---")
matrix1 = [
    [3, 5, 1],
    [2, 4, 0],
    [6, 7, 2]
]
result1, found1 = find_saddle_point(matrix1)

print("Матрица:")
print_matrix(matrix1)
print()

if found1:
    print(f"Седловая точка найдена: Координаты ({result1[0]}, {result1[1]}), Значение игры: {result1[2]}")
else:
    print("Седловая точка не найдена.")

print("\n--- Пример 2: Матрица без седловой точки ---")
matrix2 = [
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
]
result2, found2 = find_saddle_point(matrix2)

print("Матрица:")
print_matrix(matrix2)
print()

if found2:
    print(f"Седловая точка найдена: Координаты ({result2[0]}, {result2[1]}), Значение игры: {result2[2]}")
else:
    print("Седловая точка не найдена.")

print("\n--- Пример 3: Пустая матрица ---")
matrix4 = []
result4, found4 = find_saddle_point(matrix4)

print("Матрица:")
print_matrix(matrix4)
print()

if found4:
    print(f"Седловая точка найдена: Координаты ({result4[0]}, {result4[1]}), Значение игры: {result4[2]}")
else:
    print("Седловая точка не найдена.")


print("--- Тест 1: Большая матрица с гарантированной седловой точкой ---")
rows1 = 12
cols1 = 12
matrix_with_saddle = generate_matrix(rows1, cols1, 100, 1000)

saddle_row1 = random.randint(0, rows1 - 1)
saddle_col1 = random.randint(0, cols1 - 1)
saddle_value1 = 50

for j in range(cols1):
    if j == saddle_col1:
        matrix_with_saddle[saddle_row1][j] = saddle_value1
    else:
        matrix_with_saddle[saddle_row1][j] = random.randint(saddle_value1 + 1, 1000)

for i in range(rows1):
    if i == saddle_row1:
        matrix_with_saddle[i][saddle_col1] = saddle_value1
    else:
        matrix_with_saddle[i][saddle_col1] = random.randint(10, saddle_value1 - 1)

for i in range(rows1):
    for j in range(cols1):
        if matrix_with_saddle[i][j] < 0:
            matrix_with_saddle[i][j] = random.randint(0, 1000)

print(f"Генерируем матрицу {rows1}x{cols1} с гарантированной седловой точкой.")
print("Матрица (фрагмент):")
print_matrix(matrix_with_saddle)
print()

result1, found1 = find_saddle_point(matrix_with_saddle)
if found1:
    print(f"Седловая точка найдена: Координаты ({result1[0]}, {result1[1]}), Значение игры: {result1[2]}")
    if result1[0] == saddle_row1 and result1[1] == saddle_col1 and result1[2] == saddle_value1:
        print("Координаты и значение седловой точки совпадают с ожидаемыми.")
    else:
        print("ВНИМАНИЕ: Найденная седловая точка отличается от вставленной.")
else:
    print("Седловая точка не найдена.")

print("\n--- Тест 2: Большая матрица без седловой точки ---")

rows2 = 24
cols2 = 24
matrix_no_saddle = generate_matrix(rows2, cols2, -100, 100)

print(f"Генерируем матрицу {rows2}x{cols2} без гарантированной седловой точки.")
print("Матрица (фрагмент):")
print_matrix(matrix_no_saddle)
print()

result2, found2 = find_saddle_point(matrix_no_saddle)
if found2:
    print(f"Седловая точка найдена: Координаты ({result2[0]}, {result2[1]}), Значение игры: {result2[2]} (ОШИБКА: Не должно быть найдено).")
else:
    print("Седловая точка не найдена.")