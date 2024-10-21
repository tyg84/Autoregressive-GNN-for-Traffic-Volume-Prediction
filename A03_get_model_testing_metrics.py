import torch
import torch.nn as nn
from os.path import isfile

from config import *
from utils import BaselineTrainer, GNNTrainer
from models import BaseLineModel, GNNModel
from utils import TrafficVolumeDataLoader, TrafficVolumeGraphDataLoader, create_edge_index_and_features

from utils.dataloader import TrafficVolumeAutoRegressiveGNNDataLoader
from models.gnn import AutoRegressiveGNNModel

import pickle

if __name__ == "__main__":
    # check cuda version: nvidia-smi
    print(f'GPU Count {torch.cuda.device_count()}')
    print(f'CUDA Version {torch.version.cuda}')
    configs = [config_baseline, config_gnn, config_autoregressive_gnn]
    config_id = 0
    for config_id in range(len(configs)):
        config = configs[config_id]
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        holiday_file = 'data/norway_holidays.pkl'
        with open(holiday_file, 'rb') as file:
            holiday_dict = pickle.load(file)
        holiday_records = sorted(set(holiday_dict.values())) #drop last

        model_built_config = {}
        model_built_config['graph_feature_dim'] = len(holiday_records) + 3 # 3 is the dim of [month, day, hr]
        model_built_config['node_feature_dim'] = 3

        print("Loaded model configuration:")
        for key, value in config.items():
            print(f"\t* {key}: {value}")
        name = config["name"]
        lr = config["lr"]
        batch_size = config["batch_size"]
        loss_function = nn.L1Loss()

        if name == "GNN":
            # Graph NN with edge, node and graph models (using pre-defined graph)
            model = GNNModel()
            # print(model)
            edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file)
            train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
            val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
            test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
            trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
        elif name == "GNN_NE":
            # Graph NN with node and graph model (using pre-defined graph)
            model = GNNModel(use_edge_model=False)
            edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file, compute_edge_features=False)
            train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
            val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
            test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
            trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
        elif name == "GNN_KNN":
            # Graph NN using graph generated with kNN
            model = GNNModel()
            edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file)
            train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
            val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
            test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
            trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
        elif name == "Baseline":
            # Baseline fully connected NN model
            model = BaseLineModel()
            train_dataloader = TrafficVolumeDataLoader(train_data_file, batch_size, num_workers, shuffle=True)
            val_dataloader = TrafficVolumeDataLoader(val_data_file, batch_size, num_workers)
            test_dataloader = TrafficVolumeDataLoader(test_data_file, batch_size, num_workers)
            trainer = BaselineTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
        elif name == "AUTOREGRESSIVE_GNN":
            model = AutoRegressiveGNNModel(model_built_config=model_built_config)
            # print(model)
            edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file)
            train_dataloader = TrafficVolumeAutoRegressiveGNNDataLoader(train_data_file, edge_index, edge_weight,
                                                                        batch_size, num_workers, shuffle=True,
                                                                        holidays=holiday_file,
                                                                        lookback_steps=model_built_config[
                                                                            'node_feature_dim'])
            val_dataloader = TrafficVolumeAutoRegressiveGNNDataLoader(val_data_file, edge_index, edge_weight, batch_size,
                                                                      num_workers, holidays=holiday_file,
                                                                      lookback_steps=model_built_config['node_feature_dim'])
            test_dataloader = TrafficVolumeAutoRegressiveGNNDataLoader(test_data_file, edge_index, edge_weight, batch_size,
                                                                       num_workers, holidays=holiday_file,
                                                                       lookback_steps=model_built_config[
                                                                           'node_feature_dim'])
            trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

        if name != "Baseline":
            print(f'Total num edges: {edge_index.shape[1]}')
        trainer.print_model_size()
        # Evaluate model on test data and compute test loss
        trainer.evaluate()

        # Make some prediction and save plot
        from_index = 10
        all_test_time_index = test_dataloader.dataset.df.index
        length = len(all_test_time_index) - from_index - 1
        trainer.save_prediction_plot(from_index, length)
