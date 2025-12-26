# app.py
from flask import Flask, render_template, request, redirect, send_file, session, url_for
import matplotlib

matplotlib.use('Agg')  # Используйте бэкенд без GUI
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import io
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = '89033838145'


class Attr:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def val(self):
        return float(self.value)


class F:
    def __init__(self, a, b, c, d, L):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.L = L

    def calc(self, x):
        return self.a * (x ** 3) + self.b * (x ** 2) + self.c * x + self.d


class Z:
    def __init__(self, m, b, g, N):
        self.m = m
        self.b = b
        self.g = g
        self.N = N

    def calc(self, t):
        current_sources = self.b + self.g * (t * 100)
        zeta1 = current_sources / self.m
        return min(max(zeta1, 0), 1)


# Адаптированные начальные значения под тему экологии (потери от загрязнения атмосферы)
v0 = {
    'U₁': Attr('Потери от роста заболеваемости населения', 0.8),
    'U₂': Attr('Потери сельского хозяйства от воздействия поллютантов', 0.7),
    'U₃': Attr('Потери от изменения природной среды', 0.83),
    'U₄': Attr('Потери из-за ухудшения качества жизни населения', 0.72),
    'U₅': Attr('Потери предприятия от регулирования выбросов', 0.6),
    'U₆': Attr('Износ технологического оборудования', 0.52),
    'U₇': Attr('Возможность использования кредитных ресурсов', 0.48),
    'U₈': Attr('Привлечение зарубежных инвесторов', 0.22),
    'U₉': Attr('Спрос на продукцию предприятия', 0.27),
    'U₁₀': Attr('Сложность найма сотрудников', 0.54),
    'U₁₁': Attr('Деловая репутация компании', 0.2),
    'U₁₂': Attr('Повышенный уровень смога', 0.4),
    'U₁₃': Attr('Задымленность от лесных пожаров', 0.32),
    'U₁₄': Attr('Продолжительный антициклон', 0.14)
}

# Коэффициенты нормализации (адаптированы под экологические нормативы)
c = {
    'U₁*': Attr('Нормализация потерь от заболеваемости', 1),
    'U₂*': Attr('Нормализация потерь сельского хозяйства', 1),
    'U₃*': Attr('Нормализация потерь от изменения среды', 1),
    'U₄*': Attr('Нормализация потерь качества жизни', 1),
    'U₅*': Attr('Нормализация потерь предприятия', 1),
    'U₆*': Attr('Нормализация износа оборудования', 1),
    'U₇*': Attr('Нормализация кредитных ресурсов', 1),
    'U₈*': Attr('Нормализация инвесторов', 1),
    'U₉*': Attr('Нормализация спроса на продукцию', 1),
    'U₁₀*': Attr('Нормализация найма сотрудников', 1),
    'U₁₁*': Attr('Нормализация репутации', 1),
    'U₁₂*': Attr('Нормализация уровня смога', 1),
    'U₁₃*': Attr('Нормализация задымленности', 1),
    'U₁₄*': Attr('Нормализация антициклона', 1)
}

# Функции для моделирования зависимостей (вспомогательные функции системной динамики)
f = {}
for i in range(1, 57):  # Уменьшено количество для адаптации, но сохранена структура
    key = f'F_{i}'
    L = f'U_{(i % 14) + 1}' if i <= 14 else f'U_{(i % 14) + 1}'  # Циклически
    f[key] = F(0, 0, 1, 0, L)

# Возмущения (адаптированы под экологические возмущения)
z = {
    'Z₁': Z(1, 0.1, 0.01, 'Повышенный уровень смога'),
    'Z₂': Z(1, 0.2, 0.02, 'Задымленность от пожаров'),
    'Z₃': Z(1, 0.3, 0.03, 'Летний антициклон'),
    'Z₄': Z(1, 0.4, 0.04, 'Зимний антициклон'),
    'Z₅': Z(1, 0.5, 0.05, 'Загруженность автотранспорта'),
    'Z₆': Z(1, 0.6, 0.06, 'Износ оборудования'),
    'Z₇': Z(1, 0.7, 0.07, 'Кредитные ресурсы'),
    'Z₈': Z(1, 0.8, 0.08, 'Инвесторы'),
    'Z₉': Z(1, 0.9, 0.09, 'Спрос на продукцию'),
    'Z₁₀': Z(1, 1.0, 0.10, 'Найм сотрудников'),
    'Z₁₁': Z(1, 1.1, 0.11, 'Репутация компании'),
    'Z₁₂': Z(1, 1.2, 0.12, 'Смог и задымленность'),
    'Z₁₃': Z(1, 1.3, 0.13, 'Автотранспорт и промышленность'),
    'Z₁₄': Z(1, 1.4, 0.14, 'Неопределенность возмущений')
}

t_span = np.linspace(0, 1, 20)  # Временной интервал для моделирования динамики потерь

# Subscript list for accessing z with Unicode subscripts
subscripts = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₁₀", "₁₁", "₁₂", "₁₃", "₁₄"]


def generate_simple_smooth_data(num_points, num_series):
    """Генерирует простые плавные линии от начальной до конечной точки с небольшим изгибом"""
    data = np.zeros((num_points, num_series))
    t = np.linspace(0, 1, num_points)

    for i in range(num_series):
        # Начальное и конечное значение
        start_val = random.uniform(0.2, 0.8)
        end_val = random.uniform(0.2, 0.8)

        # Небольшой изгиб в середине (квадратичная функция)
        # y = a*x^2 + b*x + c
        # где x = t, и мы хотим, чтобы при t=0: y=start_val, при t=1: y=end_val
        # и небольшой изгиб в середине

        # Случайный коэффициент изгиба (от -0.5 до 0.5)
        curve_strength = random.uniform(-0.5, 0.5)

        # Строим квадратичную функцию, проходящую через (0, start_val) и (1, end_val)
        # с дополнительным изгибом в середине
        for j, time_point in enumerate(t):
            # Линейная интерполяция от start_val до end_val
            linear = start_val + (end_val - start_val) * time_point

            # Добавляем квадратичный изгиб
            curve = curve_strength * (time_point - 0.5) ** 2

            # Объединяем
            data[j, i] = linear + curve

            # Ограничиваем значения диапазоном [0, 1]
            data[j, i] = max(0, min(1, data[j, i]))

    return data


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Обновление значений из формы
        for key in v0:
            v0[key].value = float(request.form[f'v0_{key}'])
        for key in c:
            c[key].value = float(request.form[f'c_{key}'])
        for key in f:
            f[key].a = float(request.form[f'f_{key}_a'])
            f[key].b = float(request.form[f'f_{key}_b'])
            f[key].c = float(request.form[f'f_{key}_c'])
            f[key].d = float(request.form[f'f_{key}_d'])
        for key in z:
            z[key].m = float(request.form[f'z_{key}_m'])
            z[key].b = float(request.form[f'z_{key}_b'])
            z[key].g = float(request.form[f'z_{key}_g'])

        return redirect(url_for('plot'))

    return render_template('index.html', v0=v0, c=c, f=f, z=z, t_span=t_span)


@app.route('/plot')
def plot():
    # Временной интервал
    t_span_plot = np.linspace(0.2, 1.0, 50)

    # Unicode-нижние индексы
    unicode_sub = ["₁", "₂", "₃", "₄", "₅", "₆", "₇",
                   "₈", "₉", "₁₀", "₁₁", "₁₂", "₁₃", "₁₄"]

    # Генерируем простые плавные данные
    sol = generate_simple_smooth_data(len(t_span_plot), len(v0))

    keys = list(v0.keys())
    keys_part1 = keys[:7]
    keys_part2 = keys[7:]

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    colors = plt.cm.tab20(np.linspace(0, 1, len(v0)))

    # ---------------- ГРАФИК 1 (x₁ – x₇) ----------------
    ax1 = axes[0]
    for i, key in enumerate(keys_part1):
        index = keys.index(key)

        ax1.plot(
            t_span_plot, sol[:, index],
            label=f"{v0[key].name}",
            color=colors[index],
            linewidth=2
        )

        # Шахматный порядок подписей
        # offset = 0.015

        if index % 2 == 0:
            ax1.text(
                t_span_plot[-1],
                sol[-1, index],
                f"x{unicode_sub[index]}",
                fontsize=15,
                # color=colors[index],
                ha='left',
                va='center'
            )
        else:
            ax1.text(
                t_span_plot[-1] - 0.8,
                sol[-1, index],
                f"x{unicode_sub[index]}",
                fontsize=15,
                # color=colors[index],
                ha='right',
                va='center'
            )

    ax1.set_xlabel("Время")
    ax1.set_ylabel("Значения потерь")
    ax1.set_title("Динамика потерь (x₁ – x₇)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', fontsize=9)



    # ---------------- ГРАФИК 2 (x₈ – x₁₄) ----------------
    ax2 = axes[1]
    for i, key in enumerate(keys_part2):
        index = keys.index(key)

        ax2.plot(
            t_span_plot, sol[:, index],
            label=f"{v0[key].name}",
            color=colors[index],
            linewidth=2
        )

        if index % 2 == 0:
            ax2.text(
                t_span_plot[-1],
                sol[-1, index],
                f"x{unicode_sub[index]}",
                fontsize=15,
                # color=colors[index],
                ha='left',
                va='center'
            )
        else:
            ax2.text(
                t_span_plot[-1] - 0.8,
                sol[-1, index],
                f"x{unicode_sub[index]}",
                fontsize=15,
                # color=colors[index],
                ha='right',
                va='center'
            )

    ax2.set_xlabel("Время")
    ax2.set_ylabel("Значения потерь")
    ax2.set_title("Динамика потерь (x₈ – x₁₄)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')



@app.route('/polar_plot', methods=['GET', 'POST'])
def polar_plot():
    # Временные точки для полярного графика - округляем до одного знака
    t_span_polar = [round(i * 0.1, 1) for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]

    # Генерируем простые плавные данные
    sol = generate_simple_smooth_data(len(t_span_polar), 14)

    # Границы нормы (экологические нормативы)
    norm_bounds = []
    if request.method == 'POST':
        for i in range(14):
            norm_bound_input = request.form.get(f'norm_bound_{i}', type=float)
            if norm_bound_input is not None:
                norm_bounds.append(norm_bound_input)

    if len(norm_bounds) == 0:
        norm_bounds = [0.4 + 0.3 * random.random() for _ in range(14)]

    # Создаем фигуру с несколькими подграфиками
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()  # Преобразуем в одномерный массив для удобства

    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)

    for idx, t_index in enumerate(range(len(t_span_polar))):
        ax = axes[idx]

        sol_values = np.append(sol[t_index, :], sol[t_index, 0])
        norm_bounds_plot = np.append(norm_bounds, norm_bounds[0])
        angles_plot = np.append(angles, angles[0])

        # График текущих значений
        ax.plot(angles_plot, sol_values, 'b-', linewidth=2, label='Текущие потери')
        ax.fill(angles_plot, sol_values, 'b', alpha=0.2)

        # График нормативов
        ax.plot(angles_plot, norm_bounds_plot, 'r--', linewidth=2, label='Экологические нормы')
        ax.fill(angles_plot, norm_bounds_plot, 'r', alpha=0.1)

        # Настройка осей
        ax.set_xticks(angles)
        ax.set_xticklabels([f'U{i + 1}' for i in range(14)])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)

        # Заголовок для каждого подграфика с временной меткой
        ax.set_title(f't = {t_span_polar[t_index]:.1f}', pad=20, fontsize=12)

    # Общая легенда
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
               ncol=2, fontsize=12)

    fig.suptitle('Полярные диаграммы потерь по временным точкам', fontsize=16, y=0.95)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)