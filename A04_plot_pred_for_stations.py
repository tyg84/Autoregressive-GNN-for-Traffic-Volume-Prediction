import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def plot_traffic_comparison(time, actual, predicted, model_name,
                            actual_linewidth=2, actual_linestyle='-', actual_color='blue',
                            predicted_linewidth=2, predicted_linestyle='--', predicted_color='orange',
                            fontsize=12, xticks_interval=None,
                            xlim=None, ylim=None, text_position=(0.5, 0.9), save_fig=0, save_img_name=None):
    """
    Plot the comparison between actual and predicted traffic volumes over time.

    Parameters:
    - time: Array-like, Time series for the x-axis.
    - actual: Array-like, Actual traffic volumes.
    - predicted: Array-like, Predicted traffic volumes.
    - model_name: str, Name of the model to display on the plot.
    - actual_linewidth: float, Line width for the actual data.
    - actual_linestyle: str, Line style for the actual data.
    - actual_color: str, Color for the actual data line.
    - predicted_linewidth: float, Line width for the predicted data.
    - predicted_linestyle: str, Line style for the predicted data.
    - predicted_color: str, Color for the predicted data line.
    - xlim: tuple, Limit for the x-axis (min, max).
    - ylim: tuple, Limit for the y-axis (min, max).
    - text_position: tuple, Position (x, y) to place the model name text (in axes fraction).
    """

    plt.figure(figsize=(10, 6))

    # Plotting actual and predicted data with specified colors
    plt.plot(time, actual, label='Actual', linewidth=actual_linewidth, linestyle=actual_linestyle, color=actual_color)
    plt.plot(time, predicted, label='Predicted', linewidth=predicted_linewidth, linestyle=predicted_linestyle,
             color=predicted_color)

    # Labels and title
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('Traffic Volume', fontsize=fontsize)
    if xticks_interval:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=int(xticks_interval[:-1])))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Set tick label size for both axes
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # Adding model name as text
    # plt.text(text_position[0], text_position[1], model_name, transform=plt.gca().transAxes, fontsize=fontsize,
    #          bbox=dict(facecolor='white', alpha=0.5))
    plt.text(text_position[0], text_position[1], model_name, transform=plt.gca().transAxes,
             size=fontsize*1.2,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       )
             )

    # Setting limits
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Show legend
    plt.legend(fontsize=fontsize, loc='upper left')

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    if save_fig and save_img_name:
        plt.savefig(save_img_name, dpi=200)
    else:
        plt.show()




def plot_eval_at_stations(file_name, model_name, save_img_name, time_period = None):
    data = pd.read_csv(file_name)
    data['timestamps'] = pd.to_datetime(data['timestamps'])
    if time_period:
        data = data.loc[
            (data['timestamps']>=time_period[0]) &
            (data['timestamps']<=time_period[1])]
    time = pd.to_datetime(data['timestamps'])
    actual = data['actual']
    predicted = data['pred']
    if '03016V805614' in file_name:
        ylim = max(max(actual),max(predicted))*1.2
    else:
        ylim = max(max(actual), max(predicted)) * 1.2
    plot_traffic_comparison(time, actual, predicted, model_name=model_name,
                            actual_linewidth=2, actual_linestyle='-', actual_color='black',
                            predicted_linewidth=2, predicted_linestyle='--', predicted_color='red',
                            xlim=None, ylim=[0, ylim], text_position=(0.7, 0.9),
                            save_fig=1, save_img_name=save_img_name, fontsize=16, xticks_interval='5D')


if __name__ == '__main__':
    model_list = {"baseline": "DNN", 'gnn': "GNN", 'autoregressive_gnn': "Autoregressive GNN"}
    station_list = ['78845V804838', ]

    time_period_dict = {'03016V805614': ['2022-11-20', '2022-12-15'],
                   '78845V804838': ['2022-02-15', '2022-03-10']}
    for model, model_name in model_list.items():
        for station in station_list:
            time_period = time_period_dict[station]
            filename = f'output/{model}_pred_station_{station}.csv'
            save_img_name = f'img/{model}_{station}.jpg'
            plot_eval_at_stations(filename, model_name, save_img_name, time_period)
