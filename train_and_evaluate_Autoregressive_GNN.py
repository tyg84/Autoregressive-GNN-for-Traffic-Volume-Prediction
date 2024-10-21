import torch
import torch.nn as nn
from os.path import isfile
import pickle
from config import *
from utils import BaselineTrainer, GNNTrainer
from models import BaseLineModel, GNNModel
from models.gnn import AutoRegressiveGNNModel
from utils import TrafficVolumeDataLoader, TrafficVolumeGraphDataLoader, create_edge_index_and_features
from utils.dataloader import TrafficVolumeAutoRegressiveGNNDataLoader

if __name__ == "__main__":
    # check cuda version: nvidia-smi
    print(f'GPU Count {torch.cuda.device_count()}')
    print(f'CUDA Version {torch.version.cuda}')

    holiday_file = 'data/norway_holidays.pkl'

    with open(holiday_file, 'rb') as file:
        holiday_dict = pickle.load(file)
    holiday_records = sorted(set(holiday_dict.values())) #drop last

    config = config_autoregressive_gnn

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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


    model = AutoRegressiveGNNModel(model_built_config=model_built_config)
    # print(model)
    edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file)
    train_dataloader = TrafficVolumeAutoRegressiveGNNDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True, holidays=holiday_file, lookback_steps=model_built_config['node_feature_dim'])
    val_dataloader = TrafficVolumeAutoRegressiveGNNDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers, holidays=holiday_file, lookback_steps=model_built_config['node_feature_dim'])
    test_dataloader = TrafficVolumeAutoRegressiveGNNDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers, holidays=holiday_file, lookback_steps=model_built_config['node_feature_dim'])
    trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    print(f'Total num edges: {edge_index.shape[1]}')
    trainer.print_model_size()

    if not isfile(config["checkpoint_file"]):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Uncomment to use learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
        trainer.train(optimizer, scheduler)
        # trainer.train(optimizer)
        trainer.summarize_training()
    else:
        print("Checkpoint file exists. Please delete checkpoint file to re-train model.")

    # Evaluate model on test data and compute test loss
    trainer.evaluate()

    # Make some prediction and save plot
    from_index = 500
    length = 500
    trainer.save_prediction_plot(from_index, length)
