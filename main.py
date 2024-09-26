import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Завантаження даних з файлу f9.txt
data = np.loadtxt('/Users/polyamelnik/Desktop/f9.txt')

# Параметри
dt = 0.01
T = 5
N = len(data)
dlt = T / N
t = np.arange(0, T+dt, dt)

# Дискретне перетворення Фур’є
def df(x):
    N = len(x)
    df_result = np.zeros(N, dtype=complex)
    for k in range(N):
        for m in range(N):
            df_result[k] += x[m] * np.exp(-2j * np.pi * k * m / N)
        df_result[k] /= N
    return df_result

c_y = df(data)

# Пошук частоти
fr = np.fft.fftfreq(N, dlt)
mod_c_y = np.abs(c_y)

half_N = N // 2

# Пошук локальних максимумів
peaks, _ = find_peaks(mod_c_y[:half_N])

# Визначаємо частоти для локальних максимумів
significant_fr = fr[peaks]

# Відкидаємо частоти, близькі до нуля
epsilon = 1
filtered_fr = significant_fr[np.abs(significant_fr) > epsilon]

# Метод найменших квадратiв
# Функція для обчислення синусів
def sinf(t, f):
    return np.sin(2 * np.pi * f * t)
# Побудова матриці A
def matrix_A(t, filtered_fr):
    N = len(t)
    A = np.zeros((5, 5))

    A[0, 0] = np.sum(t ** 6)
    A[0, 1] = np.sum(t ** 5)
    A[0, 2] = np.sum(t ** 4)
    A[0, 3] = np.sum(sinf(t, filtered_fr[0]) * t ** 3)
    A[0, 4] = np.sum(t ** 3)

    A[1, 0] = np.sum(t ** 5)
    A[1, 1] = np.sum(t ** 4)
    A[1, 2] = np.sum(t ** 3)
    A[1, 3] = np.sum(sinf(t, filtered_fr[0]) * t ** 2)
    A[1, 4] = np.sum(t ** 2)

    A[2, 0] = np.sum(t ** 4)
    A[2, 1] = np.sum(t ** 3)
    A[2, 2] = np.sum(t ** 2)
    A[2, 3] = np.sum(sinf(t, filtered_fr[0]) * t)
    A[2, 4] = np.sum(t)

    A[3, 0] = np.sum(sinf(t, filtered_fr[0]) * t ** 3)
    A[3, 1] = np.sum(sinf(t, filtered_fr[0]) * t ** 2)
    A[3, 2] = np.sum(sinf(t, filtered_fr[0]) * t)
    A[3, 3] = np.sum(sinf(t, filtered_fr[0]) ** 2)
    A[3, 4] = np.sum(N * sinf(t, filtered_fr[0]))

    A[4, 0] = np.sum(t ** 3)
    A[4, 1] = np.sum(t ** 2)
    A[4, 2] = np.sum(t)
    A[4, 3] = np.sum(N * sinf(t, filtered_fr[0]))
    A[4, 4] = N

    return A
# Побудова вектора c
def vector_c(t, y, filtered_fr):
    c = np.array([
        np.sum(y * t ** 3),
        np.sum(y * t ** 2),
        np.sum(y * t),
        np.sum(y * sinf(t, filtered_fr[0])),
        np.sum(y)
    ])
    return c
# Визначення невідомих параметрів
def least_squares_solution(t, y, filtered_fr):
    A = matrix_A(t, filtered_fr)
    c = vector_c(t, y, filtered_fr)
    x = np.linalg.inv(A).dot(c)
    return x

result = least_squares_solution(t, data, filtered_fr)
optparams = np.round(result).astype(int)

print("Суттєві частоти:", filtered_fr)
print("a:", optparams)
def model_equation(params, fr):
    equation = f"y(t) = {params[0]} * t^3 + {params[1]} * t^2 + {params[2]} * t"
    equation += f" + {params[3]} * sin(2π * {fr[0]} * t)"
    equation += f" + {params[4]}"
    print("Рівняння моделі:", equation)
model_equation(optparams, filtered_fr)

# Графік спостережень залежно від часу
plt.figure(figsize=(10, 5))
plt.plot(t, data)
plt.title('Графік спостережень y(t) залежно від часу')
plt.xlabel('Час (секунди)')
plt.ylabel('Спостереження y(t)')
plt.grid(True)
plt.show()

# Графік модуля перетворення Фур'є(до/після виключення симетрії)
plt.figure()
plt.plot(t, np.abs(c_y))
plt.grid()
plt.title('Модуль перетворення Фур\'є')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(fr[:half_N], mod_c_y[:half_N])
plt.plot(fr[peaks], mod_c_y[peaks], 'x')
plt.title('Модуль перетворення Фур\'є')
plt.grid(True)
plt.show()

# Графік моделі з знайденими параметрами
def model(params, t, f):
    return (params[0] * t ** 3 + params[1] * t ** 2 + params[2] * t + params[3] * np.sin(2 * np.pi * f[0] * t)
            + params[4])

fitted_data = model(optparams, t, filtered_fr)

plt.figure(figsize=(10, 5))
plt.plot(t, fitted_data)
plt.title('Графік моделі')
plt.xlabel('Час (секунди)')
plt.ylabel('y(t)')
plt.grid(True)
plt.show()

# Обчислення квадратичної похибки
mse = np.mean((data - fitted_data) ** 2)
print(f"Error_value: {mse}")