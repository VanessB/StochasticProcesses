import numpy as np


def generate_wiener(T: np.array, realizations_number: int=5) -> np.ndarray:
    """
    Сгенерировать realizations_number реализаций винеровского процесса правильным методом.
    
    Параметры
    ---------
    T : np.array
        Сетка по времени.
    realizations_number : int
        Число реализаций, которое требуется сгенерировать.
    """
    
    T = np.sort(T)
    assert T[0] >= 0.0 # В более совершенной реализации стартовать можно позже нуля.
    realizations = np.zeros((realizations_number, T.shape[0]))
    
    # Для каждого очередного момента времени производится независимая генерация произошедших
    # на пройденном интервале скачков согласно нормальному распределению с нулевым средним и дисперсией \Delta t.
    from scipy.stats import norm
    step_rv = norm()
    
    realizations[:,0] = T[0] * step_rv.rvs((realizations_number)) # Если уж совсем честно генерировать, надо учесть, что T может начинаться не с нуля.
    for index in range(T.shape[0] - 1):
        realizations[:,index + 1] = realizations[:,index] + np.sqrt(T[index+1] - T[index])*step_rv.rvs((realizations_number))
    
    return realizations


def generate_gaussian(T: np.array, mean_func: callable, correlation_func: callable, realizations_number: int=5) -> np.ndarray:
    """
    Сгенерировать realizations_number реализаций гауссовского случайного процесса
    с функцией среднего mean_func и корреляционной функцией correlation_func.
    
    Параметры
    ---------
    T : np.array
        Сетка по времени.
    mean_func: callable
        Функция среднего.
        На вход получает массив T.
        Выдаёт массив матожиданий соответствующих сечений.
    correlation_func: callable
        Корреляционная функция.
        На вход получает массив T.
        Выдаёт ковариационную матрицу соответствующего вектора сечений.
    realizations_number : int
        Число реализаций, которое требуется сгенерировать.
    """
    
    realizations = np.zeros((realizations_number, T.shape[0]))
    
    # Поскольку неизвестно, является ли процесс случайным процессом с независимыми приращениями (по всей видимости, нет),
    # его требуется моделировать "в лоб", через конечномерные распределения.
    
    # Параметры конечномерного распределения.
    mean = mean_func(T)
    cov_matrix = correlation_func(T)
    
    from scipy.stats import multivariate_normal
    realizations = multivariate_normal(cov=cov_matrix).rvs((realizations_number)) + mean
    
    return realizations