import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import HTML
from matplotlib.ticker import StrMethodFormatter
import io
import base64

def calculate_descriptive_statistic(column: pd.Series):
    statistic_dict = {
        'mean': column.mean().round(2),
        'median': column.quantile(0.5),
        'std': column.std().round(2)
    }

    min_max_dict = {
        'min': column.min(),
        'max': column.max()
    }

    quartiles_dict = {
        '25%': column.quantile(0.25),
        '50%': column.quantile(0.5),
        '75%': column.quantile(0.75),
        'IQR': column.quantile(0.75) - column.quantile(0.25)

    }

    statistic_df = pd.Series(statistic_dict).to_frame(name='Descriptive')
    min_max_df = pd.Series(min_max_dict).to_frame(name='min max')
    quartiles_df = pd.Series(quartiles_dict).to_frame(name='Quartiles')

    result = {'name': 'Descriptive statistic', 'data_frames': [statistic_df, min_max_df, quartiles_df]}

    return result


def create_data_review_dfs(column: pd.Series):
    # Информация о количестве и типе данных
    total_count = column.size
    unique_val = column.nunique()
    nan_val = column.isna().sum()
    zeroes_val = (column == 0).sum()
    negative_val = (column < 0).sum()

    counts_dict = {
        'dtype': {'': column.dtype, 'types info': ''},
        'count': {'': total_count, 'types info': ''},
        'unique': {'': unique_val, 'types info': f'{(unique_val / total_count) * 100:.2f}%'},
        'nan': {'': nan_val, 'types info': f'{(nan_val / total_count) * 100:.2f}%'},
        'zeroes': {'': zeroes_val, 'types info': f'{(zeroes_val / total_count) * 100:.2f}%'},
        'negative': {'': negative_val, 'types info': f'{(negative_val / total_count) * 100:.2f}%'}
    }

    # Создаем df из dict
    head_df = column.head(6).to_frame(name='Head')
    tail_df = column.tail(6).to_frame(name='Tail')

    counts_df = pd.DataFrame.from_dict(counts_dict, orient='index')

    result = {'name': 'Data overview', 'data_frames': [counts_df, head_df, tail_df]}

    return result



def find_optimal_bins_amount(column: pd.Series):
    # Используем модифицированную версию формулы rice
    formula_amount = 3 * int(column.size ** (1 / 3))

    # Если формула выдала слишком большое значение, ограничиваемся 150 бинами
    bins_amount = min(formula_amount, 150)
    return bins_amount


def turn_fig_to_html(fig: plt.Figure):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_b64


def create_plot(column: pd.Series):
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    # ---Boxplot ---
    sns.boxplot(x=column, ax=axs[0], color='orange')
    axs[0].set_ylabel(None)
    axs[0].set_xlabel(column.name)
    axs[0].set_title('Box plot')

    # ---Histplot ---
    bins_amount = find_optimal_bins_amount(column)
    sns.histplot(x=column, ax=axs[1], bins=bins_amount)
    axs[1].set_ylabel(None)
    axs[1].set_xlabel(column.name)
    axs[1].set_title('Histogram')

    # ---KDE plot---
    sns.kdeplot(x=column, fill=True, ax=axs[2], color='orange')
    axs[2].set_ylabel(None)
    axs[2].set_title('Kernel Density Estimation (kde)')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.subplots_adjust(hspace=1)

    return fig


def check_data(column: pd.Series):
    if not isinstance(column, pd.Series):
        raise TypeError("The input data must be a pandas Series object")

    if column.empty:
        raise TypeError(f"Сannot process empty column {column.name}")

    if not np.issubdtype(column.dtype, np.number):
        raise TypeError(
            f"The column must contain only numbers and NaN. The column '{column.name}' has a data type: {column.dtype}")


def create_html(col_name, info_blocks, plot_html):
    html = f'<h1>{col_name}</h1>'

    # --- Информационные блоки ---
    for block in info_blocks:
        html += f'<h3>{block['name']}</h3>'
        html += '<div style="display:flex; gap: 40px;">'
        for df in block['data_frames']:
            html += f'<div>{df.to_html()}</div>'
        html += '</div>'

    # --- Графики ---
    plot_html = f'<img src="data:image/png;base64,{plot_html}">'

    html += '<hr>'
    html += f'<div>{plot_html}</div>'

    return html


def show_quant_data_info(column: pd.Series):
    # Проверка данных
    check_data(column)

    # Создание dataframes из данных
    basic_data = create_data_review_dfs(column)
    statisic_data = calculate_descriptive_statistic(column)
    info_blocks = [basic_data, statisic_data]

    # Создаем график и его изображение
    plot = create_plot(column)
    plot_html = turn_fig_to_html(plot)

    # Создаем HTML
    html = create_html(column.name, info_blocks, plot_html)

    display(HTML(html))
