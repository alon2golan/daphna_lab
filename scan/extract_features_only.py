"""
Extract features from a pre-trained SimCLR model without retraining.
This is much faster than running the full training loop.
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_model, get_train_dataset, get_val_dataset, \
                                get_val_dataloader, get_val_transformations
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Extract Features from Pretrained SimCLR Model')
parser.add_argument('--config_env', help='Config file for the environment')
parser.add_argument('--config_exp', help='Config file for the experiment')
parser.add_argument('--model_path', help='Path to the pretrained model', required=True)
parser.add_argument('--seed', type=int, default=1, help='Random seed')

args = parser.parse_args()

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.seed)
    print(colored(p, 'red'))
    
    # Model
    print(colored('Loading pretrained model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    
    # Load pretrained weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    print(colored(f'Loading weights from {args.model_path}', 'blue'))
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.cuda()
    model.eval()  # Set to evaluation mode
    
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    
    # Get train dataset without augmentations for feature extraction
    train_dataset = get_train_dataset(p, val_transforms, split='train')
    train_dataloader = get_val_dataloader(p, train_dataset)
    
    # Get test/val dataset
    val_dataset = get_val_dataset(p, val_transforms) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    
    # Memory Banks
    print(colored('Build MemoryBanks', 'blue'))
    memory_bank_train = MemoryBank(len(train_dataset), 
                                    p['model_kwargs']['features_dim'],
                                    p['num_classes'], 
                                    p['criterion_kwargs']['temperature'])
    memory_bank_train.cuda()
    
    memory_bank_val = MemoryBank(len(val_dataset),
                                  p['model_kwargs']['features_dim'],
                                  p['num_classes'], 
                                  p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()
    
    # Extract train features
    print(colored('Extracting train features...', 'blue'))
    with torch.no_grad():
        fill_memory_bank(train_dataloader, model, memory_bank_train)
    
    # Extract val/test features
    print(colored('Extracting val/test features...', 'blue'))
    with torch.no_grad():
        fill_memory_bank(val_dataloader, model, memory_bank_val)
    
    # Save features
    print(colored('Saving features...', 'blue'))
    train_features = memory_bank_train.pre_lasts.cpu().numpy()
    test_features = memory_bank_val.pre_lasts.cpu().numpy()
    
    np.save(p['pretext_features'], train_features)
    np.save(p['pretext_features'].replace('features', 'test_features'), test_features)
    
    print(colored(f"Train features saved to: {p['pretext_features']}", 'green'))
    print(colored(f"Test features saved to: {p['pretext_features'].replace('features', 'test_features')}", 'green'))
    print(colored(f"Train features shape: {train_features.shape}", 'green'))
    print(colored(f"Test features shape: {test_features.shape}", 'green'))
    
    # Mine nearest neighbors for train set
    print(colored('Mining nearest neighbors (train)...', 'blue'))
    topk = 20
    indices, acc = memory_bank_train.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)
    
    # Mine nearest neighbors for val set
    print(colored('Mining nearest neighbors (val)...', 'blue'))
    topk = 5
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)
    
    print(colored('Done! Features extracted successfully.', 'green'))

if __name__ == '__main__':
    main()
