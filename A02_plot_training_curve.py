import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_loss_curve(models_losses, model_names, save_fig):



    linewidth = 2
    fontsize = 12
    ###############

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define some linestyles and colors for variation
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_losses)))

    for i, (train_loss, val_loss) in enumerate(models_losses):
        iterations = np.array(range(len(train_loss))) + 1
        # Plot training loss
        plt.plot(train_loss, linestyle=linestyles[i % len(linestyles)], color=colors[i],
                 label=f'{model_names[i]} Train', linewidth=linewidth)
        # Plot validation loss
        plt.plot(val_loss, linestyle=linestyles[i % len(linestyles)], color=colors[i], alpha=0.7,
                 label=f'{model_names[i]} Validation', linewidth=linewidth)

    # Labels and title
    plt.xlabel('Training Iterations', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    # plt.title('Training and Validation Loss Comparison', fontsize=fontsize + 2)

    # Y-axis cut-off for values between 200-300 and 25-40
    ax.set_ylim(25, 65)
    # ax.set_yticks(list(range(25, 41, 5)) + list(range(200, 301, 20)))
    # ax.set_yticklabels(list(range(25, 41, 5)) + list(range(200, 301, 20)))

    # Labels and title
    ax.set_xlabel('Training Iterations', fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    # ax.set_title('Training and Validation Loss Comparison', fontsize=fontsize + 2)

    # Customizing the font size of the ticks
    ax.tick_params(axis='both', which='major', labelsize=fontsize)


    # plt.xlim(200, 420)
    # plt.ylim(25, 40)
    # Add legend
    plt.legend(fontsize=fontsize)

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'img/train_val_loss.jpg', dpi=200)
    else:
        plt.show()


if __name__ == '__main__':
    model_list = { "baseline": "DNN", 'gnn': "GNN", 'autoregressive_gnn':"Autoregressive GNN"}
    models_losses = []
    models_names = []

    save_fig = 1
    for model, model_name in model_list.items():
        loss = pd.read_csv(f'output/{model}_loss.csv')
        training_loss = loss['train_loss']
        validation_loss = loss['val_loss']
        models_losses.append((training_loss, validation_loss))
        models_names.append(model_name)

    plot_loss_curve(models_losses, models_names, save_fig)