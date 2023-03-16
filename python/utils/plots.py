import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_realizations(T: np.array, realizations: np.ndarray,
                      is_discrete: bool=False, title: str="Реализации случайного процесса", **kwargs):
    """
    Отрисовка реализаций слчайного процесса.
    
    Параметры
    ---------
    T : np.array
        Сетка по времени.
    realizations : np.ndarray
        Данные о реализациях (ось 0: реализации, ось 1: точки выбранной реализации).
    is_discrete : bool
        Рисовать ли графики как траектории процесса с дискретными сечениями.
    title : str
        Заголовок графика.
    """
    
    fig, ax = plt.subplots()

    fig.set_figwidth(kwargs["width"] if "width" in kwargs else 16)
    fig.set_figheight(kwargs["height"] if "height" in kwargs else 10)

    # Сетка.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax.set_title(title)
    ax.set_xlabel("$ t $")
    ax.set_ylabel("$ X(\\omega) $")
    
    ax.minorticks_on()
    
    if is_discrete:
        for realization in realizations:
            ax.step(T, realization)
    else:
        for realization in realizations:
            ax.plot(T, realization)

    plt.show();
    
    

def plot_realizations_heatmap(T, realizations, bandwidth=1.06, grid_size=41, title="Тепловая карта реализаций случайного процесса", **kwargs):
    """
    Отрисовка реализаций слчайного процесса.
    
    Параметры
    ---------
    T : np.array
        Сетка по времени.
    realizations : np.array
        Данные о реализациях (ось 0: реализации, ось 1: точки выбранной реализации).
    title : str
        Заголовок графика.
    """
    
    fig, ax = plt.subplots()

    fig.set_figwidth(kwargs["width"] if "width" in kwargs else 16)
    fig.set_figheight(kwargs["height"] if "height" in kwargs else 10)

    # Сетка.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax.set_title(title)
    ax.set_xlabel("$ t $")
    ax.set_ylabel("$ X(\\omega) $")
    
    ax.minorticks_on()
    
    # Разброс реализаций.
    global_min = np.min(realizations)
    global_max = np.max(realizations)
    global_delta = global_max - global_min
    
    local_std = np.std(realizations, axis=0) + 1e-6
    
    # Сетка по оси значений сечений.
    grid_X = np.linspace(global_min - 0.01 * global_delta, global_max + 0.01 * global_delta, grid_size)
    
    # Оценка плотности методом KDE с гауссовым ядром.
    sigma = bandwidth * local_std / realizations.shape[0]**0.2
    density = np.mean(np.exp(-0.5 * (realizations[:,:,None] - grid_X[None,None,:])**2 / (sigma**2)[None,:,None]), axis=0)# / (sigma[None,:,None] * np.sqrt(2.0 * np.pi)), axis=0)
    
    ax.contourf(T, grid_X, density.T, levels=(kwargs["levels"] if "levels" in kwargs else 10))

    plt.show();

    
    
def plot_slices(states, slices, is_discrete=False, title="Сечения случайного процесса", **kwargs):
    """
    Отрисовка сечений слчайного процесса.
    
    Параметры
    ---------
    states : np.array
        Сетка по состояниям.
    slices : scipy.stats random variable
        Случайные величины, соответствующие сечениям.
    is_discrete : bool
        Являются ли сечения дискретными случайными величинами?
        В противном случае считается, что сечения - непрерывные случайные величины.
    title : str
        Заголовок графика.
    """
    
    fig, ax = plt.subplots()

    fig.set_figwidth(kwargs["width"] if "width" in kwargs else 16)
    fig.set_figheight(kwargs["height"] if "height" in kwargs else 10)

    # Сетка.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    ax.set_title(title)
    ax.set_xlabel("$ x $")
    
    ax.minorticks_on()
    
    if is_discrete:
        # Дискретная случайная величина, строим график функции веротности.
        ax.set_ylabel("$ p_{X_t}(x) $")
        
        for slice_rv, slice_label in slices:
            nonzero_states = states[slice_rv.pmf(states) > 0]
            pmf = slice_rv.pmf(nonzero_states)
            color = next(ax._get_lines.prop_cycler)['color']
            
            ax.plot(nonzero_states, pmf, "o", color=color, ms=8, label=slice_label)
            ax.vlines(nonzero_states, 0, pmf, color=color, lw=5, alpha=0.5)
    
    else:
        # Непрерывная случайная величина, строим график плотности вероятности.
        ax.set_ylabel("$ \\rho_{X_t}(x) $")
        
        for slice_rv, slice_label in slices:
            pdf = slice_rv.pdf(states)
            ax.plot(states, pdf, label=slice_label)
            ax.fill_between(states, pdf, alpha=0.2)
        
    ax.legend(loc='upper left')

    plt.show();
    
    

def plot_correlation_function(T, realizations, title=None, normalize=False, **kwargs):
    """
    Отрисовка реализаций слчайного процесса.
    
    Параметры
    ---------
    T : np.array
        Сетка по времени.
    realizations : np.array
        Данные о реализациях (ось 0: реализации, ось 1: точки выбранной реализации).
    title : str
        Заголовок графика.
    """
    
    fig, ax = plt.subplots()

    fig.set_figwidth(kwargs["width"] if "width" in kwargs else 10)
    fig.set_figheight(kwargs["height"] if "height" in kwargs else 10)

    # Сетка.
    ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
    ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')

    if title is None:
        title = "Функция коэффициента корреляции случайного процесса" if normalize else "Корреляционная функция случайного процесса"
        
    ax.set_title(title)
    ax.set_xlabel("$ t_1 $ (номер)")
    ax.set_ylabel("$ t_2 $ (номер)")
    
    ax.minorticks_on()
    
    correlation_fun = np.cov(realizations, rowvar=False)
    if normalize:
        dispersion_fun = np.diag(correlation_fun)
        correlation_fun /= np.sqrt(np.outer(dispersion_fun, dispersion_fun))
    
    ax.imshow(correlation_fun, interpolation='bilinear', origin="lower")
    #ax.contourf(T, T, correlation_fun)

    plt.show();