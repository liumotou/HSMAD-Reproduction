import os
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # cuda
    parser.add_argument('--cuda_device', type=int, default=2, help='Cuda device index')
    
    parser.add_argument('--dataset_root', type=str, default='./dataset', help='Path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='tolokers', help='Name of the dataset')

    # model
    parser.add_argument('--hiddim', type=int, default=64, help='Dimension of the hidden layer')
    parser.add_argument('--gnndp', type=float, default=0.0, help="dropout ratio of gnn")
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the gnn')
    
    parser.add_argument('--num_steps', type=int, default=500, help='Number of total diffusion steps')
    parser.add_argument('--beta_1', type=float, default=1e-4, help='Beta_1 for the diffusion process')
    parser.add_argument('--beta_T', type=float, default=0.02, help='Beta_T for the diffusion process')


    # training 
    parser.add_argument('--batch_size', type=int, default=16384, help='Batch size for training')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')

    parser.add_argument('--step_size', type=int, default=150, help='Step size for the learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for the learning rate scheduler')

    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency for training')

    # log
    parser.add_argument('--checkpoints_dir', type=str, default='./data', help='Directory to save logs')
    parser.add_argument('--loadFilename', type=str, default=None, help='Load model from a .tar file')
    parser.add_argument('--note', type=str, default="", help='note for the directory')

    
    opt = parser.parse_args()
    
    return opt