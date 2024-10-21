from os.path import join
from torch_geometric.seed import seed_everything

# Seed random generators (python, torch, numpy)
seed_everything(0)

# Paths
data_path = "data"
figs_path = "figs"
docs_path = "docs"
output_path = "output"
checkpoints_path = "checkpoints"

# Filenames
data_file_zip = join(data_path, "traffic_data.zip")
data_file_pkl = join(data_path, "traffic_data.pkl")
stations_data_file = join(data_path, "traffic_stations.csv")
summary_table_file = join(docs_path, "data_summary_table.md")
time_series_file = join(data_path, "time_series_data.pkl")
train_data_file = join(data_path, "time_series_train.pkl")
val_data_file = join(data_path, "time_series_val.pkl")
test_data_file = join(data_path, "time_series_test.pkl")
graph_file = join(data_path, "graph.pkl")
stations_included_file = join(data_path, "stations_included.csv")

# Pre-processing
min_number_of_observations = 1500   # Drop stations having too few observations
val_fraction = 0.10                 # Fraction of data to use for validation data
test_fraction = 0.20                # Fraction of data to use for test data
normalize_data = None               # "minmax" : scale to [0,1], "normal" : use z-scores, None : no normalization

# Training 
num_workers = 8                     # Number of workers to use with dataloader.
# device = 'cpu'
# Baseline model config
config_baseline = {}
config_baseline["name"] = "Baseline"
config_baseline["batch_size"] = 128 
config_baseline["lr"] = 0.001
config_baseline["epochs"] = 100
config_baseline["val_per_epoch"] = 4
config_baseline["checkpoint_file"] = join(checkpoints_path, "baseline.pth") 
config_baseline["prediction_plot_dir"] = join(figs_path, "baseline_predictions")
config_baseline["loss_plot_file"] = join(figs_path, "baseline_loss_plot.png")
config_baseline["loss_save_dir"] = join(output_path, "baseline_loss.csv")
config_baseline["earlystop_limit"] = 20 


# GNN model config
config_gnn = {}
config_gnn["name"] = "GNN"
config_gnn["batch_size"] = 128
config_gnn["lr"] = 0.001
config_gnn["epochs"] = 100
config_gnn["val_per_epoch"] = 4
config_gnn["checkpoint_file"] = join(checkpoints_path, "gnn.pth") 
config_gnn["prediction_plot_dir"] = join(figs_path, "gnn_predictions")
config_gnn["loss_plot_file"] = join(figs_path, "gnn_loss_plot.png")
config_gnn["loss_save_dir"] = join(output_path, "gnn_loss.csv")
config_gnn["earlystop_limit"] = 20

# GNN model with no edge model
config_gnn_ne = {}
config_gnn_ne["name"] = "GNN_NE"
config_gnn_ne["batch_size"] = 128 
config_gnn_ne["lr"] = 0.001
config_gnn_ne["epochs"] = 100
config_gnn_ne["val_per_epoch"] = 4
config_gnn_ne["checkpoint_file"] = join(checkpoints_path, "gnn_ne.pth") 
config_gnn_ne["prediction_plot_dir"] = join(figs_path, "gnn_ne_predictions")
config_gnn_ne["loss_plot_file"] = join(figs_path, "gnn_ne_loss_plot.png")
config_gnn_ne["loss_save_dir"] = join(output_path, "gnn_ne_loss.csv")
config_gnn_ne["earlystop_limit"] = 20 

# GNN model with kNN generated graph
config_gnn_knn = {}
config_gnn_knn["name"] = "GNN_KNN"
config_gnn_knn["batch_size"] = 128 
config_gnn_knn["lr"] = 0.001
config_gnn_knn["epochs"] = 100
config_gnn_knn["val_per_epoch"] = 4
config_gnn_knn["checkpoint_file"] = join(checkpoints_path, "gnn_knn.pth") 
config_gnn_knn["prediction_plot_dir"] = join(figs_path, "gnn_knn_predictions")
config_gnn_knn["loss_plot_file"] = join(figs_path, "gnn_knn_loss_plot.png")
config_gnn_knn["loss_save_dir"] = join(output_path, "gnn_knn_loss.csv")
config_gnn_knn["earlystop_limit"] = 20 


# GNN model with kNN generated graph
config_autoregressive_gnn = {}
config_autoregressive_gnn["name"] = "AUTOREGRESSIVE_GNN"
config_autoregressive_gnn["batch_size"] = 128
config_autoregressive_gnn["lr"] = 0.001
config_autoregressive_gnn["epochs"] = 100
config_autoregressive_gnn["val_per_epoch"] = 4
config_autoregressive_gnn["checkpoint_file"] = join(checkpoints_path, "autoregressive_gnn.pth")
config_autoregressive_gnn["prediction_plot_dir"] = join(figs_path, "autoregressive_gnn_predictions")
config_autoregressive_gnn["loss_plot_file"] = join(figs_path, "autoregressive_gnn_loss_plot.png")
config_autoregressive_gnn["loss_save_dir"] = join(output_path, "autoregressive_gnn_loss.csv")
config_autoregressive_gnn["earlystop_limit"] = 20

# List of all models
configs = [config_baseline, config_gnn, config_gnn_ne, config_gnn_knn, config_autoregressive_gnn]
