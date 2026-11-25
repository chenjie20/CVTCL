import warnings
import itertools

from ogb.linkproppred import Evaluator
from tqdm import tqdm

from layers import EdgeDecoder
from models import CVTCL, CVCTLv2
from loss_evl import *
from dataprocessing import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CVTCL')
parser.add_argument('--dataset_name', type=str, default='Cora',
                    choices=['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'CS', 'Physics', 'Actor', 'Chameleon'],
                    help='Dataset name')
parser.add_argument( "--node_coder_name", type=str, default='HO', choices=['HO', 'HE'], help='HO and HE denote homophilic and heterophilic graphs, respectively.' )
parser.add_argument('--gpu', default=0, type=int, help='GPU device idx.')

parser.add_argument('--lr', type=float, default=5e-3, help='Initializing learning rate chosen from [1e-3, 5e-3, 5e-2]')
parser.add_argument('--weight_decay', type=float, default=0, help='Initializing weight decay.')
parser.add_argument('--hidden_feature_dim', type=int, default=512, help='Dimensionality of hidden representation.')
parser.add_argument('--hidden_edge_dim', type=int, default=128, help='Dimensionality of hidden representation in the edge decoder.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--eval_steps', type=int, default=5, help='Evaluation steps for validation and testing.')
parser.add_argument('--num_neighbors_layers', type=int, nargs='+', default=[15, 10], help='Number of neighbors sampled at each GNN layer.')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument("--hits", type=int, default=100, choices=[100, 50], help='Hits for evaluation.')

parser.add_argument('--ratio', type=float, default=0.2, help='Default testing ratio for several datasets.')
parser.add_argument('--runs', type=int, default=5, help='Number of runs.')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42, help='Initializing random seed.')
parser.add_argument('--edge_dropout', type=float, default=0.2)
parser.add_argument('--model_dropout', type=float, default=0.1)

# parser.add_argument( "--edge_coder_name", type=str, default='EdgeDecoder', choices=['EdgeDecoder', 'ExtendedEdgeDecoder'], help='Edge coder name.' )
# parser.add_argument( "--augmented_view_type", type=str, default='redundant_ratio', choices=['redundant_ratio', 'redundant', 'complementary'], help='Augmented view type.')

args = parser.parse_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def test_full(model, data, batch_size=65536):
    model.eval()
    test_results = {}

    z = model(data.x, data.edge_index)

    edge_label_index = data.edge_label_index
    num_edges = edge_label_index.size(1)
    preds_set = []

    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        batch_edges = edge_label_index[:, start:end]
        preds = model.edge_decoder(z, batch_edges).view(-1).cpu()
        preds_set.append(preds)

    pred_results = torch.cat(preds_set, dim=0)
    target_results = data.edge_label.cpu()

    pos_test_pred = pred_results[target_results == 1]
    neg_test_pred = pred_results[target_results == 0]

    evaluator = Evaluator(name='ogbl-ppa')
    evaluator.K = args.hits
    hits_result = evaluator.eval({
        "y_pred_pos": pos_test_pred,
        "y_pred_neg": neg_test_pred
    })[f"hits@{args.hits}"]
    test_results[f'hits@{args.hits}'] = hits_result

    return test_results


@torch.no_grad()
def test(model, data_loader):
    model.eval()

    test_results = {}
    preds_set, targets_set = [], []

    for batch_data in data_loader:
        z = model(batch_data.x, batch_data.edge_index)
        preds = model.edge_decoder(z, batch_data.edge_label_index).view(-1).detach().cpu()
        preds_set.append(preds)
        targets_set.append(batch_data.edge_label.detach().cpu())

    pred_results = torch.cat(preds_set, dim=0)
    target_results = torch.cat(targets_set, dim=0)

    pos_test_pred = pred_results[target_results == 1]
    neg_test_pred = pred_results[target_results == 0]

    evaluator = Evaluator(name='ogbl-ppa')
    evaluator.K = args.hits
    hits_result = evaluator.eval({
        "y_pred_pos": pos_test_pred,
        "y_pred_neg": neg_test_pred
    })[f"hits@{args.hits}"]
    test_results[f'hits@{args.hits}'] = hits_result

    # target_results = target_results.detach().numpy()
    # pred_results = pred_results.detach().numpy()
    # test_auc = roc_auc_score(target_results, pred_results)
    # test_ap = average_precision_score(target_results, pred_results)
    # test_results['auc'] = test_auc
    # test_results['ap'] = test_ap

    return test_results


def train(model, optimizer, train_loader, generate_two_views):

    model.train()

    total_loss = 0.0
    for batch_train_data in train_loader:

        x, edge_index = batch_train_data.x, batch_train_data.edge_index

        edge_views = generate_two_views(edge_index)

        num_views = len(edge_views)
        assert num_views == 2

        loss = 0.0
        optimizer.zero_grad()
        for idx in range(num_views):
            v_idx = 1 - idx
            neg_edges = negative_sampling(
                edge_index,
                num_nodes=x.size(0),
                num_neg_samples=edge_views[v_idx].size(1),
            ).view_as(edge_views[v_idx])
            adj = to_sparse_tensor(edge_views[idx], x.size(0))
            z = model(x, adj)
            pos_out = model.edge_decoder(z, edge_views[v_idx], sigmoid=False)
            neg_out = model.edge_decoder(z, neg_edges, sigmoid=False)
            loss += binary_cross_entropy_loss(pos_out, neg_out)

            del neg_edges, adj, z

            # training_edges = torch.cat((edge_views[v_idx], neg_edges), dim=-1)
            # predicted_labels = model.edge_decoder(z, training_edges, sigmoid=False).squeeze()
            # training_labels = torch.cat((torch.ones(edge_views[v_idx].size()[1]), torch.zeros(neg_edges.size()[1])),
            #                             dim=0).to(device)
            # loss += criterion(predicted_labels, training_labels)

        loss = loss / num_views
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# def save_model(model, model_parent_dir, dataset_name, run, epoch):
#     model_file_name = f'{dataset_name}_run{run}_{epoch}_model.pth'
#     model_file_path = os.path.join(model_parent_dir, model_file_name)
#     os.makedirs(model_parent_dir, exist_ok=True)
#     torch.save(model.state_dict(), model_file_path)
#
#     return model_file_path


def set_default_config():
    if args.dataset_name in {'Cora', 'Citeseer', 'Pubmed'}:
        args.node_coder_name = 'HO'
        args.hits = 100
        args.epochs = 100
        args.eval_steps = 2
        args.edge_dropout = 0.2
        args.model_dropout = 0.8

    elif args.dataset_name in {'Photo', 'Computers'}:
        args.node_coder_name = 'HO'
        args.hits = 50
        args.eval_metrics = ['hits']
        args.epochs = 150
        args.eval_steps = 5
        args.edge_dropout = 0.2
        args.model_dropout = 0.1

    elif args.dataset_name in {'CS', 'Physics'}:
        args.node_coder_name = 'HO'
        args.hits = 50
        args.epochs = 150
        args.eval_steps = 5
        args.edge_dropout = 0.2
        args.model_dropout = 0.1

    elif args.dataset_name in {'Actor', 'Chameleon'}:
        args.node_coder_name = 'HE'
        args.hits = 100
        args.epochs = 150
        args.eval_steps = 5
        args.edge_dropout = 0.2
        args.model_dropout = 0.2

    elif args.dataset_name in {'Squirrel', 'Cornell', 'Texas', 'Wisconsin'}:
        args.hits = 100
        args.epochs = 100
        args.eval_steps = 5
        args.edge_dropout = 0.2
        args.model_dropout = 0.8

    else:
        raise NotImplementedError('Dataset {} is not supported.'.format(args.dataset_name))


def get_param_grids():
    if args.dataset_name in {'Cora', 'Citeseer', 'Pubmed'}:
        lr_params = np.array([5e-3, 1e-3, 5e-4, 1e-2], dtype=np.float32)
        dims_layers = np.array([512, 1024, 2048], dtype=np.int32)
        batch_sizes = np.array([512, 1024, 2048], dtype=np.int32)
        num_neighbors_layers_set = np.array([[50, 25], [30, 15], [15, 10]], dtype=np.int32)
        model_dropouts = np.array([0.6, 0.8], dtype=np.float32)

        # lr_params = np.array([5e-3], dtype=np.float32) #Pubmed
        # dims_layers = np.array([2048, 1024], dtype=np.int32)
        # batch_sizes = np.array([512], dtype=np.int32)
        # # num_neighbors_layers_set = np.array([[50, 25], [30, 15], [15, 10], [10, 5]], dtype=np.int32)
        # num_neighbors_layers_set = np.array([[50, 25]], dtype=np.int32)
        # # model_dropouts = np.array([0.6, 0.7, 0.8], dtype=np.float32)
        # model_dropouts = np.array([0.6], dtype=np.float32)

    elif args.dataset_name in {'Photo', 'Computers'}:
        lr_params = np.array([5e-4], dtype=np.float32)
        if args.dataset_name == 'Computers':
            dims_layers = np.array([512], dtype=np.int32)
            batch_sizes = np.array([2048], dtype=np.int32)
        else:
            dims_layers = np.array([512], dtype=np.int32)
            batch_sizes = np.array([8192, 4096], dtype=np.int32)
        num_neighbors_layers_set = np.array([[400, 200]], dtype=np.int32)
        model_dropouts = np.array([0.05, 0.1], dtype=np.float32)

    elif args.dataset_name in {'CS', 'Physics'}:
        lr_params = np.array([5e-3, 1e-3], dtype=np.float32)
        if args.dataset_name == 'CS':
            dims_layers = np.array([1536, 1024], dtype=np.int32)
            batch_sizes = np.array([4096], dtype=np.int32)
        else:
            dims_layers = np.array([512, 256], dtype=np.int32)
            batch_sizes = np.array([1024], dtype=np.int32)
        num_neighbors_layers_set = np.array([[400, 200]], dtype=np.int32)
        model_dropouts = np.array([0.1, 0.2], dtype=np.float32)

    elif args.dataset_name in {'Actor', 'Chameleon'}:
        lr_params = np.array([5e-3, 1e-3, 1e-2], dtype=np.float32)
        dims_layers = np.array([512, 256], dtype=np.int32)
        batch_sizes = np.array([8192, 4096, 2048], dtype=np.int32)
        num_neighbors_layers_set = np.array([[400, 200]], dtype=np.int32)
        model_dropouts = np.array([0.05, 0.1], dtype=np.float32)

    else:
        raise NotImplementedError('Dataset {} is not supported.'.format(args.dataset_name))

    hidden_edge_dims = np.array([256], dtype=np.int32)
    edge_dropouts = np.array([0.1, 0.2], dtype=np.float32)

    return  lr_params, dims_layers, hidden_edge_dims, batch_sizes, num_neighbors_layers_set, model_dropouts, edge_dropouts


if __name__ == '__main__':

    print("Starting...")

    set_seed(args.seed)
    set_default_config()
    model_parent_dir = './models/models_%s_%d/' % (args.dataset_name, args.gpu)

    dataset, data_set = load_data(args.dataset_name, args.ratio, device)

    lr_params, dims_layers, hidden_edge_dims, batch_sizes, num_neighbors_layers_set, model_dropouts, edge_dropouts = get_param_grids()
    param_grid = list(itertools.product(lr_params, dims_layers, hidden_edge_dims, batch_sizes, num_neighbors_layers_set, model_dropouts, edge_dropouts))
    eval_size = int(args.epochs / args.eval_steps)

    for lr, dim, hidden_edge_dim, batch_size, num_neighbors_layers, model_dropout, edge_dropout in param_grid:
        args.lr = float(lr)
        args.hidden_feature_dim = int(dim)
        args.hidden_edge_dim = int(hidden_edge_dim)
        args.batch_size = int(batch_size)
        args.num_neighbors_layers = num_neighbors_layers
        args.model_dropout = float(model_dropout)
        args.edge_dropout = float(edge_dropout)
        print("==========\nArgs:{}\n==========".format(args))
        train_loader, val_loader, test_loader = get_loaders(data_set, args.batch_size, args.num_neighbors_layers,
                                                            args.num_workers)

        test_hits_set, val_hits_set = [], []
        for run in range(args.runs):
            edge_decoder = EdgeDecoder(args.hidden_feature_dim, args.hidden_edge_dim, dropout=args.edge_dropout)
            if args.node_coder_name == 'HO':
                model = CVTCL(edge_decoder, dataset.num_features, args.hidden_feature_dim, args.model_dropout)
            elif args.node_coder_name == 'HE':
                model = CVCTLv2(edge_decoder, dataset.num_features, args.hidden_feature_dim, args.hidden_feature_dim,
                                args.num_layers, args.model_dropout)
            else:
                raise NotImplementedError('Node coder {} is not supported.'.format(args.node_coder_name))

            model = model.to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

            # if args.augmented_view_type == 'redundant_ratio':
            #     generate_two_views = generate_two_redundant_views_with_ratio
            # elif args.augmented_view_type == 'redundant':
            #     generate_two_views = generate_two_redundant_views
            # elif args.augmented_view_type == 'complementary':
            #     generate_two_views = generate_two_complementary_views
            # else:
            #     raise NotImplementedError('Augmented view type {} is not supported.'.format(args.augmented_view_type))
            generate_two_views = generate_two_redundant_views_with_ratio

            pbar = tqdm(range(1, args.epochs + 1), desc=f'Training',
                        bar_format='{l_bar}{bar:20}{r_bar}', ncols=180)
            for epoch in pbar:
                total_loss = train(model, optimizer, train_loader, generate_two_views)
                if epoch % args.eval_steps == 0:
                    # val_results = test(model, val_loader)
                    # test_results = test(model, test_loader)
                    val_results = test_full(model, data_set['valid'])
                    test_results = test_full(model, data_set['test'])

                    # test_auc, test_ap = test_results['auc'], test_results['ap']
                    # val_auc, val_ap = val_results['auc'], val_results['ap']
                    # pbar.set_postfix(epoch=f'{epoch}', loss=f'{total_loss:.3f}', val_auc=f'{val_auc:.4f}',
                    #                  val_ap=f'{val_ap:.4f}',
                    #                  test_auc=f'{test_auc:.4f}', test_ap=f'{test_ap:.4f}')


                    test_hits_result = test_results[f'hits@{args.hits}']
                    val_hits_result = val_results[f'hits@{args.hits}']
                    test_hits_set.append(test_hits_result)
                    val_hits_set.append(val_hits_result)
                    pbar.set_postfix(epoch=f'{epoch}', loss=f'{total_loss:.3f}',
                                     test_hits=f'{test_hits_result:.4f}', val_hits=f'{val_hits_result:.4f}')
                    # print('epoch {} loss:{:.4f} test_hits:{:.2f} val_hits:{:.4f}'.format(epoch, total_loss, test_hits_result * 100, val_hits_result))
                    with open('CVTCL_%s_result_details_%s.txt' % (args.dataset_name, args.gpu), 'a+') as f:
                        f.write('{:.4f} \t {} \t {} \t {} \t {:.2f} \t {:.2f} \t {} \t {} \t {:.4f} \t {:.2f} \n'.format(args.lr,
                                                                               args.hidden_feature_dim,
                                                                               args.hidden_edge_dim,
                                                                               args.batch_size,
                                                                               args.model_dropout,
                                                                               args.edge_dropout,
                                                                               run,
                                                                               epoch,
                                                                               val_hits_result,
                                                                               test_hits_result * 100))
                        f.flush()
            del model
            torch.cuda.empty_cache()
            print(f'{run} finished.')

        test_hits_reuslts = torch.tensor(test_hits_set)
        val_hits_reuslts = torch.tensor(val_hits_set)
        test_hits_reuslts = test_hits_reuslts.view(args.runs, eval_size)
        val_hits_reuslts = val_hits_reuslts.view(args.runs, eval_size)
        _, selected_epoches = val_hits_reuslts.max(dim=1)

        test_hits_list = test_hits_reuslts[torch.arange(args.runs, dtype=torch.long), selected_epoches]
        test_hits_mean = test_hits_list.mean().item()
        test_hits_std = test_hits_list.std().item()
        print('test_hits_mean:{:.2f} test_hits_std:{:.4f}'.format(
            test_hits_mean * 100, test_hits_std))
        with open('CVTCL_%s_result_%s.txt' % (args.dataset_name, args.gpu),
                  'a+') as f:
            f.write(
                '{:.4f} \t {} \t {} \t {:.2f} \t {:.2f} \t {} \t {} \t {:.4f} \t {:.2f} \t \n'.format(args.lr,
                                                                                             args.hidden_feature_dim,
                                                                                             args.hidden_edge_dim,
                                                                                             args.model_dropout,
                                                                                             args.edge_dropout,
                                                                                             args.batch_size,
                                                                                             str(args.num_neighbors_layers),
                                                                                             test_hits_mean * 100,
                                                                                             test_hits_std * 100))
            f.flush()
