DATASET_NAMES = ['linreg', 'mlpreg']

def make_dataset_name(args):
    #if hasattr(args, 'data_dim'):
    #    data_dim = args.data_dim
    if args.dataset == "linreg":
        name = f'linreg-dim{args.data_dim}-key{args.data_key}'
    elif args.dataset == "mlpreg":
        name = f'mlpreg-dim{args.data_dim}-mlp{args.data_neurons}-key{args.data_key}'
    else:
        raise Exception(f'Unknown dataset {args.dataset}')
    return name
