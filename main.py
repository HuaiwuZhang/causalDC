from dataset import *
from utils import *
from src.base.trainer import *
import argparse
import wandb

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes","True", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no","False", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dataset', type=str, default='drugcomb', help='pretrain dataset')
    parser.add_argument('--finetune_dataset', type=str, default='drugcomb', help='finetune dataset')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for data splitting (default: 42)')
    parser.add_argument('--torch_seed', type=int, default=42, help='Random seed for model parameter (default: 42)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
    parser.add_argument('--mode', type=str, default='pretrain', help='pretrain/finetuning/eval')
    parser.add_argument('--pretrain', type=str2bool, default=True, help='Training stage')
    parser.add_argument('--testing', type=str2bool, default=True, help='whether to use test')
    parser.add_argument('--model', type=str, default='synergyx_causal', help='used model')
    parser.add_argument('--task', type=str, default='regression', help='regression or classification')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=25, help='Early stop epoch')
    parser.add_argument('--save_iter', type=int, default=0)
    parser.add_argument('--n_exp', type=int, default=1, help='Experiment Index')
    parser.add_argument('--wandb', type=str2bool, default=True, help='Whether to use wandb')
    parser.add_argument('--direct_finetune', type=str2bool, default=True, help='whether to direct finetune')
    parser.add_argument('--labels', type=str, default='sscore', help='Metrics of Synergy: sscore/loewe/bliss/zip/hsa')
    parser.add_argument('--infer', type=str2bool, default='False', help='inference phase')
    parser.add_argument('--causal_ablation_ratio', type=float, default=0, help='causal_ablation_ratio: 0 - 1')
    parser.add_argument('--IG_drugA', type=str, default='', help='IG_drugA')
    parser.add_argument('--IG_drugB', type=str, default='', help='IG_drugB')
    parser.add_argument('--IG_ablation', type=str2bool, default=False, help='IG_ablation')
    parser.add_argument('--nc_nsclc', type=str2bool, default=False, help='nc_nsclc')
    parser.add_argument('--few_shot_ratio', type=float, default=1, help='Training data ratio when doing few-shot learning')
    parser.add_argument('--novel', type=str2bool, default=False, help='test novel combos')
    parser.add_argument('--get_cell_embedding', type=str2bool, default=False, help='Get Cell Embedding')
    parser.add_argument('--test_valid', type=str2bool, default=False, help='Calculate the performance on Valid set')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    args.steps = [100, 200, 300, 400]

    print(args)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    param_str_list = [args.model, args.split_seed, args.torch_seed, args.batch_size, args.n_exp, args.labels]
    param_str = '-'.join(str(value) for value in param_str_list)
    print(param_str)
    if args.few_shot_ratio < 1 and args.few_shot_ratio > 0:
        args.finetune_dataset = 'few_shot'
        param_str = param_str + '_few_shot_' + str(args.few_shot_ratio)
    args.log_dir = "{}/experiment/{}/{}".format(folder_path, args.finetune_dataset, param_str)
    if args.wandb:
        wandb.login(key="03455359de6e84d9bb06ffcf977c65a92e8fb8e8")
        wandb.login(key="<your-wandb-key>")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)
    if args.torch_seed != 0:
        torch.manual_seed(args.torch_seed)

    # Loading data & Preprocessing
    synergy, drug2smile, drug2espf, cline2expression, cline2mutation, cline2dependency, cline2copynumber, gene_dim, mutation_dim, label_index, tissue_name_list, cell_line_list, gene_list, drug2graph, atom_type, cline2id, novel_combos = get_data(args, folder_path)
    args.atom_type = atom_type
    columns = ['drugA', 'drugB', 'cell_line', 's_mean']
    synergy_df = pd.DataFrame(synergy, columns=columns)
    synergy_df_infer = synergy_df.drop_duplicates(subset=['cell_line'])
    synergy_df_infer = np.array(synergy_df_infer)
    infer_batch_size = len(synergy_df_infer)
    synergy_df_ig = find_drug_combinations(synergy_df, args)
    synergy_df_ig_celline = synergy_df_ig['cell_line']
    synergy_df_ig = np.array(synergy_df_ig)

    novel_combos_df = pd.DataFrame(novel_combos, columns=columns)
    novel_combos = np.array(novel_combos)

    source_train, source_val, source_test = synergy_data_split(synergy_df, args)
    if args.few_shot_ratio < 1 and args.few_shot_ratio > 0:
        source_train = source_train[:int(args.few_shot_ratio * len(source_train))]
        
    source_train_process = process_data(source_train, drug2smile, cline2expression, cline2mutation, cline2dependency, cline2copynumber)
    source_val_process = process_data(source_val, drug2smile, cline2expression, cline2mutation, cline2dependency, cline2copynumber)
    source_test_process = process_data(source_test, drug2smile, cline2expression, cline2mutation, cline2dependency, cline2copynumber)
    source_infer_process = process_data(synergy_df_infer, drug2smile, cline2expression, cline2mutation,
                                        cline2dependency, cline2copynumber)
    source_ig_process = process_data(synergy_df_ig, drug2smile, cline2expression, cline2mutation,
                                        cline2dependency, cline2copynumber)
    source_novel_process = process_data(novel_combos, drug2smile, cline2expression, cline2mutation,
                                     cline2dependency, cline2copynumber)

    # dataset & dataloader
    # ESPF
    if args.model in ['synergyx_causal', 'deepsynergy_causal', 'synergyx', 'deepsynergy']:
        train_loader = create_loader(source_train_process, args, drug2espf, shuffle=True, drop_last=True)
        val_loader = create_loader(source_val_process, args, drug2espf)
        test_loader = create_loader(source_test_process, args, drug2espf)
        infer_loader = create_infer_loader(source_infer_process, args, drug2espf, infer_batch_size)
        if args.IG_drugA and args.IG_drugB:
            ig_loader = create_infer_loader(source_ig_process, args, drug2espf, infer_batch_size=1)
        if args.IG_ablation:
            ig_test_loader = create_infer_loader(source_test_process, args, drug2espf, infer_batch_size=1)

        novel_loader = create_loader(source_novel_process, args, drug2espf)
    # 2D-Graph
    elif args.model in ['deepdds','deepdds_causal']:
        train_loader = create_graph_loader(source_train_process, args, drug2espf, drug2graph, shuffle=True, drop_last=True)
        val_loader = create_graph_loader(source_val_process, args, drug2espf, drug2graph)
        test_loader = create_graph_loader(source_test_process, args, drug2espf, drug2graph)
        infer_loader = create_graph_infer_loader(source_infer_process, args, drug2espf, drug2graph, infer_batch_size)

        novel_loader = create_graph_loader(source_novel_process, args, drug2espf, drug2graph)


    # wandb Setting
    if args.wandb:
        # Ablation
        if args.IG_ablation:
            wandb.init(
                project='CausalDC_CasaulGenes_Ablation',
                name=str(args.n_exp) + ' ' + args.model + ' ' + args.labels + ' ' + str(args.causal_ablation_ratio),
                config=vars(args)
            )

        # Few-shot
        elif args.few_shot_ratio < 1 and args.few_shot_ratio > 0:
            wandb.init(
                project='causal_few_shot',
                name=str(args.n_exp) + ' ' + args.model + ' ' + args.labels + ' ' + str(args.few_shot_ratio),
                config=vars(args)
            )

        # Valid performance
        elif args.test_valid:
            wandb.init(
                project='test_valid',
                name=str(args.n_exp) + ' ' + args.model+ ' ' + args.labels,
                config=vars(args)
            )

        # Trianing and Testing
        else:
            wandb.init(
                project='CausalDC',
                name=str(args.n_exp) + ' ' + args.model+ ' ' + args.labels,
                config=vars(args)
            )

    model = get_model(gene_dim, args).to(device)


    train_data = {}
    train_data['train_loader'] = train_loader
    train_data['val_loader'] = val_loader
    train_data['test_loader'] = test_loader
    train_data['infer_loader'] = infer_loader
    if args.IG_drugA and args.IG_drugB:
        train_data['ig_loader'] = ig_loader
    if args.IG_ablation:
        train_data['ig_test_loader'] = ig_test_loader
    train_data['novel_loader'] = novel_loader

    pretrainer = BaseTrainer(model=model, data = train_data, args=args)
    if args.model in ['synergyx', 'deepsynergy','deepdds']:
        if args.pretrain:
            pretrainer.pretrain('pretrain')
        if args.testing:
            pretrainer.test('pretrain')
        if args.nc_nsclc:
            pretrainer.test_nc('pretrain',mode='nc_nsclc')
        if args.test_valid:
            pretrainer.test('pretrain', mode='val')
    if args.model in ['synergyx_causal', 'deepsynergy_causal', 'deepdds_causal']:
        if args.infer:
            pretrainer.infer_causal(stage='pretrain', cell_line_list=cell_line_list, gene_list=gene_list)
        if args.pretrain:
            pretrainer.pretrain_causal('pretrain')
        if args.IG_drugA and args.IG_drugB:
            pretrainer.ig_analysis(stage='pretrain', cline2id=cline2id, gene_list=gene_list, synergy_df_ig_celline=synergy_df_ig_celline)
        if args.IG_ablation:
            pretrainer.ig_ablation(stage='pretrain', cline2id=cline2id, gene_list=gene_list,
                                   synergy_df_ig_celline=synergy_df_ig_celline)
        if args.testing:
            pretrainer.test_causal('pretrain')
        if args.nc_nsclc:
            pretrainer.test_nc_causal('pretrain',mode='nc_nsclc')
        if args.test_valid:
            pretrainer.test_causal('pretrain', mode='val')
        if args.novel:
            pretrainer.novel_causal('pretrain', mode='novel', synergy=novel_combos_df, cline2id=cline2id)
    
    
    if args.get_cell_embedding:
        pretrainer.get_cell_embedding('pretrain', mode='infer', cell_line_list=cell_line_list, gene_list=gene_list)
    if args.wandb:
        wandb.finish()
