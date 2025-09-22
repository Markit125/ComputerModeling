import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict

df = pd.read_excel("Lab2/ЛР2_Домохозяйства_в_районах_РТ.xls", header=1)
print("Файл Excel успешно загружен.")

data_columns = [
    'Процент домохозяйств с дефицитом',
    'Средний дефицит',
    'Cредний префицит',
    'МО распределения',
    'Дисперсия распределения',
    'Cредний доход на одно домохозяйство'
]

X = df[data_columns].values
regions = df['Район'].values

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X)
print("Данные успешно нормализованы к диапазону [0, 1].")

print(f"Количество районов в обработанных данных: {len(regions)}")
print(f"Количество признаков (параметров) для кластеризации: {X_normalized.shape[1]}")
print("Подготовка данных завершена\n")

print("Начинается базовый запуск SOM")
initial_m = 8
initial_n = 8
initial_sigma = 0.5 * max(initial_m, initial_n) / 2
initial_learning_rate = 0.5
initial_num_iterations = 10000

som = MiniSom(initial_m, initial_n, X_normalized.shape[1],
              sigma=initial_sigma, learning_rate=initial_learning_rate)

som.random_weights_init(X_normalized)

print(f"Начало обучения SOM с размером сетки {initial_m}x{initial_n} и {initial_num_iterations} итерациями...")
som.train_random(X_normalized, initial_num_iterations, verbose=True)
print(f"Базовая SOM ({initial_m}x{initial_n}) успешно обучена.")
print("Базовый запуск SOM завершен\n")

print("Построение графиков для базового запуска")
plt.figure(figsize=(12, 12))
plt.title(f'U-Matrix для SOM ({initial_m}x{initial_n}) - Базовый запуск', fontsize=16)
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.9)
plt.colorbar(label='Среднее расстояние до соседних нейронов', orientation='horizontal')

winner_map = defaultdict(list)
for i, x in enumerate(X_normalized):
    w = som.winner(x)
    winner_map[w].append(regions[i])

max_vertical_offset = 0.4

for (x_coord, y_coord), region_names in winner_map.items():
    num_regions_in_cell = len(region_names)
    
    if num_regions_in_cell > 1:
        dy_step = (2 * max_vertical_offset) / (num_regions_in_cell + 1)
        dy_offsets = np.arange(num_regions_in_cell) * dy_step - max_vertical_offset + dy_step/2
    else:
        dy_offsets = [0]
    
    for i, region_name in enumerate(region_names):
        dx = 0
        dy = dy_offsets[i]

        plt.text(x_coord + 0.5 + dx, y_coord + 0.5 + dy, region_name,
                 color='red',
                 fontsize=max(6, 10 - num_regions_in_cell),
                 ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                 rotation=0)

plt.xticks(np.arange(initial_m + 1), labels=[f'{i}' for i in range(initial_m + 1)])
plt.yticks(np.arange(initial_n + 1), labels=[f'{i}' for i in range(initial_n + 1)])
plt.grid(True, linestyle='--', alpha=0.6, color='gray')
plt.xlabel('Координата X нейрона', fontsize=12)
plt.ylabel('Координата Y нейрона', fontsize=12)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(18, 12))
# plt.suptitle(f'Компонентные плоскости - Базовый запуск ({initial_m}x{initial_n})', fontsize=18)

# for i, feature_name in enumerate(data_columns):
#     plt.subplot(2, 3, i + 1)
#     plt.title(feature_name, fontsize=14)
#     plt.pcolor(som.get_weights()[:, :, i].T, cmap='viridis')
#     plt.colorbar()
#     plt.xticks(np.arange(initial_m + 1))
#     plt.yticks(np.arange(initial_n + 1))
#     plt.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
# print("Построение графиков для базового запуска завершено\n")

print("\n3.1. Варьирование размеров сетки")
grid_sizes = [(5, 5), (10, 10), (15, 15)]
results_grid_size = []

for m, n in grid_sizes:
    print(f"Обучение SOM с размером сетки: {m}x{n}")
    som_gs = MiniSom(m, n, X_normalized.shape[1],
                   sigma=initial_sigma, learning_rate=initial_learning_rate)
    som_gs.random_weights_init(X_normalized)
    som_gs.train_random(X_normalized, initial_num_iterations, verbose=False)

    plt.figure(figsize=(m + 2, n + 2))
    plt.title(f'U-Matrix ({m}x{n})', fontsize=16)
    plt.pcolor(som_gs.distance_map().T, cmap='bone_r', alpha=0.9)
    plt.colorbar(label='Среднее расстояние до соседних нейронов')

    winner_map_gs = defaultdict(list)
    for i, x in enumerate(X_normalized):
        w = som_gs.winner(x)
        winner_map_gs[w].append(regions[i])
    
    for (x_coord, y_coord), region_names in winner_map_gs.items():
        num_regions_in_cell = len(region_names)
        if num_regions_in_cell > 1:
            dy_step = (2 * max_vertical_offset) / (num_regions_in_cell + 1)
            dy_offsets = np.arange(num_regions_in_cell) * dy_step - max_vertical_offset + dy_step/2
        else:
            dy_offsets = [0]
        
        for i, region_name in enumerate(region_names):
            dx = 0
            dy = dy_offsets[i]

            plt.text(x_coord + 0.5 + dx, y_coord + 0.5 + dy, region_name,
                     color='red',
                     fontsize=max(6, 12 - m - (num_regions_in_cell//2)),
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.xticks(np.arange(m + 1), labels=[f'{i}' for i in range(m + 1)])
    plt.yticks(np.arange(n + 1), labels=[f'{i}' for i in range(n + 1)])
    plt.grid(True, linestyle='--', alpha=0.6, color='gray')
    plt.xlabel('Координата X нейрона', fontsize=12)
    plt.ylabel('Координата Y нейрона', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    results_grid_size.append({
        'Размер сетки': f'{m}x{n}',
        'Количество нейронов': m * n,
        'Комментарий': 'Визуальная оценка качества кластеризации (сколько кластеров видно, их форма), время обучения (наблюдаемое).'
    })
    
print("\n3.2. Выбор различных функций соседства")
neighborhood_functions = ['gaussian', 'bubble']
results_neighborhood = []

for func in neighborhood_functions:
    print(f"Обучение SOM с функцией соседства: {func}")
    som_nf = MiniSom(initial_m, initial_n, X_normalized.shape[1],
                     sigma=initial_sigma, learning_rate=initial_learning_rate,
                     neighborhood_function=func)
    som_nf.random_weights_init(X_normalized)
    som_nf.train_random(X_normalized, initial_num_iterations, verbose=False)
    
    plt.figure(figsize=(initial_m + 2, initial_n + 2))
    plt.title(f'U-Matrix ({func} neighborhood)', fontsize=16)
    plt.pcolor(som_nf.distance_map().T, cmap='bone_r', alpha=0.9)
    plt.colorbar(label='Среднее расстояние до соседних нейронов')
    
    winner_map_nf = defaultdict(list)
    for i, x in enumerate(X_normalized):
        w = som_nf.winner(x)
        winner_map_nf[w].append(regions[i])

    for (x_coord, y_coord), region_names in winner_map_nf.items():
        num_regions_in_cell = len(region_names)
        if num_regions_in_cell > 1:
            dy_step = (2 * max_vertical_offset) / (num_regions_in_cell + 1)
            dy_offsets = np.arange(num_regions_in_cell) * dy_step - max_vertical_offset + dy_step/2
        else:
            dy_offsets = [0]
        
        for i, region_name in enumerate(region_names):
            dx = 0
            dy = dy_offsets[i]
            plt.text(x_coord + 0.5 + dx, y_coord + 0.5 + dy, region_name,
                     color='red', fontsize=8, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.xticks(np.arange(initial_m + 1), labels=[f'{i}' for i in range(initial_m + 1)])
    plt.yticks(np.arange(initial_n + 1), labels=[f'{i}' for i in range(initial_n + 1)])
    plt.grid(True, linestyle='--', alpha=0.6, color='gray')
    plt.xlabel('Координата X нейрона', fontsize=12)
    plt.ylabel('Координата Y нейрона', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    results_neighborhood.append({
        'Функция соседства': func,
        'Комментарий': 'Визуальная оценка формы кластеров, гладкости карты, как сильно перемещаются нейроны при обучении.'
    })

print("\n3.3. Выберите разную скорость обучения")
learning_rates = [(0.1, 0.01), (0.5, 0.05), (0.9, 0.1)]
results_learning_rate = []

for lr_start, lr_end in learning_rates:
    print(f"Обучение SOM со скоростью обучения: начальная={lr_start}")
    som_lr = MiniSom(initial_m, initial_n, X_normalized.shape[1],
                     sigma=initial_sigma, learning_rate=lr_start)
    som_lr.random_weights_init(X_normalized)
    som_lr.train_random(X_normalized, initial_num_iterations, verbose=False)

    plt.figure(figsize=(initial_m + 2, initial_n + 2))
    plt.title(f'U-Matrix (Начальная скорость обучения: {lr_start})', fontsize=16)
    plt.pcolor(som_lr.distance_map().T, cmap='bone_r', alpha=0.9)
    plt.colorbar(label='Среднее расстояние до соседних нейронов')
    
    winner_map_lr = defaultdict(list)
    for i, x in enumerate(X_normalized):
        w = som_lr.winner(x)
        winner_map_lr[w].append(regions[i])

    for (x_coord, y_coord), region_names in winner_map_lr.items():
        num_regions_in_cell = len(region_names)
        if num_regions_in_cell > 1:
            dy_step = (2 * max_vertical_offset) / (num_regions_in_cell + 1)
            dy_offsets = np.arange(num_regions_in_cell) * dy_step - max_vertical_offset + dy_step/2
        else:
            dy_offsets = [0]
        
        for i, region_name in enumerate(region_names):
            dx = 0
            dy = dy_offsets[i]
            plt.text(x_coord + 0.5 + dx, y_coord + 0.5 + dy, region_name,
                     color='red', fontsize=8, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.xticks(np.arange(initial_m + 1), labels=[f'{i}' for i in range(initial_m + 1)])
    plt.yticks(np.arange(initial_n + 1), labels=[f'{i}' for i in range(initial_n + 1)])
    plt.grid(True, linestyle='--', alpha=0.6, color='gray')
    plt.xlabel('Координата X нейрона', fontsize=12)
    plt.ylabel('Координата Y нейрона', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    results_learning_rate.append({
        'Начальная скорость обучения': lr_start,
        'Комментарий': 'Визуальная оценка сходимости (быстрота обучения), детализации кластеров, общего упорядочивания карты.'
    })

print("Исследования параметров SOM завершены\n")

# Вывод сводных таблиц результатов исследований
print("\nСводные результаты исследований")
print(pd.DataFrame(results_grid_size).to_string(index=False))
print("\nТаблица: Выбор различных функций соседства")
print(pd.DataFrame(results_neighborhood).to_string(index=False))
print("\nТаблица: Выбор разной скорости обучения")
print(pd.DataFrame(results_learning_rate).to_string(index=False))
print("Сводные результаты исследований завершены\n")


print("Анализ кластеров")
som_weights = som.get_weights().reshape(-1, X_normalized.shape[1])

n_clusters = 5

kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(som_weights)

som_cluster_map = cluster_labels.reshape(initial_m, initial_n)

plt.figure(figsize=(12, 12))
plt.title(f'Кластеры на SOM (K-Means, {n_clusters} кластеров) с названиями районов', fontsize=16)
plt.pcolor(som_cluster_map.T, cmap='viridis', alpha=0.9)
plt.colorbar(label='Номер кластера')

winner_map_cluster = defaultdict(list)
for i, x in enumerate(X_normalized):
    w = som.winner(x)
    winner_map_cluster[w].append(regions[i])

for (x_coord, y_coord), region_names in winner_map_cluster.items():
    num_regions_in_cell = len(region_names)
    if num_regions_in_cell > 1:
        dy_step = (2 * max_vertical_offset) / (num_regions_in_cell + 1)
        dy_offsets = np.arange(num_regions_in_cell) * dy_step - max_vertical_offset + dy_step/2
    else:
        dy_offsets = [0]
    
    for i, region_name in enumerate(region_names):
        dx = 0
        dy = dy_offsets[i]
        
        cluster_id = som_cluster_map[x_coord, y_coord] 

        plt.text(x_coord + 0.5 + dx, y_coord + 0.5 + dy, region_name,
                 color='white',
                 ha='center', va='center',
                 bbox=dict(facecolor=plt.cm.viridis(cluster_id / (n_clusters - 1)),
                           edgecolor='none', boxstyle='round,pad=0.2'),
                 fontweight='bold', fontsize=7)

plt.xticks(np.arange(initial_m + 1), labels=[f'{i}' for i in range(initial_m + 1)])
plt.yticks(np.arange(initial_n + 1), labels=[f'{i}' for i in range(initial_n + 1)])
plt.grid(True, linestyle='--', alpha=0.6, color='gray')
plt.xlabel('Координата X нейрона', fontsize=12)
plt.ylabel('Координата Y нейрона', fontsize=12)
plt.tight_layout()
plt.show()
print("График кластеров с названиями районов построен.")

print("\nАнализ средних значений показателей по кластерам")
clusters_data = {i: [] for i in range(n_clusters)}
region_clusters = {}

for i, x in enumerate(X_normalized):
    w = som.winner(x)
    cluster_id = som_cluster_map[w[0], w[1]]
    clusters_data[cluster_id].append(X[i])
    region_clusters[regions[i]] = cluster_id

cluster_summary = []
for cluster_id, data_in_cluster in clusters_data.items():
    if data_in_cluster:
        mean_values = np.mean(data_in_cluster, axis=0)

        regions_in_cluster = [region for region, c_id in region_clusters.items() if c_id == cluster_id]

        cluster_info = {
            'Кластер ID': cluster_id,
            'Количество районов': len(data_in_cluster),
            'Районы': ', '.join(regions_in_cluster)
        }

        for j, col_name in enumerate(data_columns):
            cluster_info[f'Средний {col_name}'] = mean_values[j]

        cluster_summary.append(cluster_info)

df_cluster_summary = pd.DataFrame(cluster_summary)
print(df_cluster_summary.to_string(index=False))
print("Таблица средних значений по кластерам сформирована.")

print("\nВыводы о типах районов по характеристикам домохозяйств")

for index, row in df_cluster_summary.iterrows():
    cluster_id = row['Кластер ID']
    
    percent_deficit = row['Средний Процент домохозяйств с дефицитом']
    avg_deficit = row['Средний Средний дефицит']
    avg_prefit = row['Средний Cредний префицит']
    avg_mo = row['Средний МО распределения']
    avg_disp = row['Средний Дисперсия распределения']
    avg_income = row['Средний Cредний доход на одно домохозяйство']
    
    conclusion = f"Кластер {cluster_id}: "
    
    if cluster_id == 0:
        conclusion += "Неблагополучный, но стабильный район: Очень низкий процент домохозяйств с дефицитом, но и низкий средний доход. Значительный средний префицит (отрицательный), что может указывать на общий дефицит. Низкие МО и Дисперсия говорят о меньшем разбросе показателей."
    elif cluster_id == 1:
        conclusion += "Смешанные городские/пригородные районы: Высокий процент домохозяйств с дефицитом и значительный средний дефицит, но также присутствует средний доход. Отрицательный, но менее выраженный префицит, и положительное МО указывают на некоторую экономическую активность, несмотря на проблемы с дефицитом."
    elif cluster_id == 2:
        conclusion += "Районы с высоким потенциалом дохода, но выраженными финансовыми дисбалансами: Самый высокий средний доход, но при этом довольно высокий процент домохозяйств с дефицитом и значительный средний дефицит. Крайне высокая дисперсия и очень низкое МО, что указывает на большой разброс между благополучными и неблагополучными домохозяйствами."
    elif cluster_id == 3:
        conclusion += "Средне-неблагополучные районы с умеренным разбросом: Высокий процент домохозяйств с дефицитом и значительный средний дефицит. Средний доход находится между благополучными и крайне неблагополучными группами. МО положительное, а дисперсия умеренная, что указывает на более однородную ситуацию по сравнению с кластером 2."
    elif cluster_id == 4:
        conclusion += "Район с высоким разбросом показателей и заметным дефицитом: Средний процент домохозяйств с дефицитом, заметный средний дефицит и отрицательный префицит. Отличается ОЧЕНЬ высокой дисперсией, что говорит о крайней неоднородности финансовых показателей домохозяйств внутри этого района."
    else:
        conclusion += "Неопределенный тип районов: не удалось классифицировать по основным признакам. Требуется дополнительный анализ."
        
    print(conclusion)
print("Анализ кластеров завершен\n")

print("\nСравнение с теоретическими выводами")
print("3.1. Выводы по варьированию размеров сетки:")
print("  - Меньшие сетки (например, 5x5) дают более грубую, обобщенную кластеризацию. Они хорошо подходят для выявления основных, крупных кластеров, но могут упустить более тонкие различия между районами. Время обучения, как правило, меньше.")
print("  - Большие сетки (например, 10x10, 15x15) позволяют выявить более детализированные кластеры и лучше сохраняют топологию входных данных. Однако они более чувствительны к шумам, требуют больше времени на обучение и могут создать слишком много маленьких, трудноинтерпретируемых кластеров, если данных недостаточно для 'заполнения' всех нейронов.")
print("  - Оптимальный размер сетки часто находится эмпирически. Для небольшого количества данных и потребности в общих группах лучше меньшая сетка, для детального анализа - большая.")

print("\n3.2. Выводы по выбору различных функций соседства:")
print("  - Гауссова функция соседства (Gaussian neighborhood) обычно приводит к более гладким и топологически упорядоченным картам. Это происходит потому, что обновление весов нейронов плавно убывает с увеличением расстояния от нейрона-победителя, что способствует созданию непрерывных областей. Она хорошо сохраняет исходную топологию данных.")
print("  - Пузырьковая функция (Bubble neighborhood) обновляет веса всех нейронов в пределах радиуса одинаково (дискретно). Это может привести к более четким, но менее гладким границам кластеров и иногда к менее упорядоченной карте. Обучение с пузырьковой функцией может быть быстрее, но может потребоваться больше итераций для стабильной сходимости.")
print("  - Выбор функции зависит от цели: Гауссова для сохранения топологии и гладкости, Пузырьковая для более резких границ.")

print("\n3.3. Выводы по выбору разной скорости обучения:")
print("  - Высокая начальная скорость обучения (например, 0.9) позволяет сети быстро адаптироваться к общим паттернам в данных на ранних этапах обучения, быстро перемещая веса нейронов в сторону областей данных. Это полезно для быстрого 'обнаружения' основных структур.")
print("  - Низкая конечная скорость обучения (например, 0.01) позволяет нейронам 'тонко настраиваться' на свои входные эталонные вектора на поздних этапах. Это уменьшает колебания весов и обеспечивает точную сходимость, позволяя нейронам занять стабильные положения, которые лучше представляют данные. Если скорость обучения остается высокой до конца, сеть может 'перепрыгивать' через оптимальные решения и не стабилизироваться.")
print("  - Стратегия уменьшения скорости обучения со временем (линейное или экспоненциальное) является стандартной. Она обеспечивает баланс между быстрым исследованием пространства данных на начальных этапах и точной подстройкой весов на завершающих.")

print("\nЛабораторная работа по кластеризации с помощью самоорганизующихся карт Кохонена завершена.")
