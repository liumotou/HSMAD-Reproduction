import argparse
import time
from utils import *
import pandas
import os
import warnings
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))
import re
def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--inductive', type=int, default=0)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--manifold_config', type=str, default="H8S8E8", help="Specify manifold configuration (e.g., H8S8E8)")
parser.add_argument('--optimizer', type=str, choices=['adam', 'riemannian_adam'], default='adam', help="Choose the optimizer: adam or riemannian_adam")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the optimizer")
parser.add_argument('--drop_rate', type=float, default=0, help="Dropout rate for the model")
parser.add_argument('--K', type=int, default=3, help="Number of Chebyshev polynomial filters (K)")
parser.add_argument('--use_autoencoder', action='store_true', help="Use autoencoder with curvature reconstruction")
parser.add_argument('--alpha', type=float, default=1.0, help="Weight for the reconstruction loss term")
args = parser.parse_args()

columns = ['name']
new_row = {}
datasets = [ 'chameleon']
# datasets = ['disney', 'books', 'inj_cora', 'enron', 'reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
            # 'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial', 'hetero/amazon', 'hetero/yelp']
models = model_detector_dict.keys()

def get_matching_datasets(prefix, datasets_dir='datasets/'):
    """
    Return a list of dataset names that match the given prefix.
    For example, if prefix is 'actor', this will return ['actor_0', 'actor_1', ...]
    
    Parameters:
    - prefix: The dataset prefix to match (e.g., 'actor', 'cora')
    - datasets_dir: Directory where datasets are stored
    
    Returns:
    - List of dataset names matching the prefix
    """
    all_files = os.listdir(datasets_dir)
    dataset_names = [f for f in all_files if re.match(f'{prefix}_*', f)]
    
    return dataset_names

if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
    print('Evaluated Datasets: ', datasets)


all = []
for data in datasets:
    all.extend(get_matching_datasets(data))


datasets = all
print('Evaluated Datasets: ', datasets)
if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset in datasets:
    for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                   'RecK mean', 'RecK std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        if model in ['CAREGNN', 'H2FD'] and 'hetero' not in dataset_name:
            continue
        time_cost = 0
        train_config = {
            'dataset_name':dataset_name,
            'device': 'cuda',
            'epochs': 200,
            'patience': 100,
            'metric': 'AUPRC',
            'inductive': bool(args.inductive),
            'optimizer': args.optimizer,  # Add optimizer to train config
            'use_autoencoder': args.use_autoencoder,
            'alpha': args.alpha,
            'use_ricci_flow_weights': True}
        data = Dataset(dataset_name)

        
        model_config = {
            'model': model,
            'lr': args.lr,  # Pass learning rate from argparse
            'drop_rate': args.drop_rate} # Pass drop rate from argparse
        if model == "CurvGAD":
            print("Model name is:", model)
            model_config = {
            'model': model,
            'lr': args.lr,  # Pass learning rate from argparse
            'drop_rate': args.drop_rate,
            "K": args.K,
            'manifolds_config_str': args.manifold_config} # Pass drop rate from argparse
    
        
        if dataset_name == 'tsocial':
            model_config['h_feats'] = 16
            # if model in ['GHRN', 'KNNGCN', 'AMNet', 'GT', 'GAT', 'GATv2', 'GATSep', 'PNA']:   # require more than 24G GPU memory
                # continue

        auc_list, pre_list, rec_list = [], [], []
        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            data.split(args.semi_supervised, t)
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            detector = model_detector_dict[model](train_config, model_config, data)
            st = time.time()
            print(detector.model_curv, detector.model_adj_feat)
            test_score = detector.train()
            auc_list.append(test_score['AUROC']), pre_list.append(test_score['AUPRC']), rec_list.append(test_score['RecK'])
            ed = time.time()
            time_cost += ed - st
        del detector, data

        model_result[dataset_name+'-AUROC mean'] = np.mean(auc_list)
        model_result[dataset_name+'-AUROC std'] = np.std(auc_list)
        model_result[dataset_name+'-AUPRC mean'] = np.mean(pre_list)
        model_result[dataset_name+'-AUPRC std'] = np.std(pre_list)
        model_result[dataset_name+'-RecK mean'] = np.mean(rec_list)
        model_result[dataset_name+'-RecK std'] = np.std(rec_list)
        model_result[dataset_name+'-Time'] = time_cost/args.trials
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    file_id = save_results(results, file_id)
    print(results)
