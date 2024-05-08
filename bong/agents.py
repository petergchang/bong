from bong.src import bbb, blr, bog, bong
import pandas as pd

AGENT_TYPES = ["fg-bong", "fg-l-bong", "fg-rep-bong", "fg-rep-l-bong",
               "fg-blr", "fg-bog", "fg-bbb", "fg-rep-bbb"]


LR_AGENT_TYPES = ["fg-blr", "fg-bog", "fg-rep-bog", "fg-bbb", "fg-rep-bbb"]


BONG_DICT = {
    "fg-bong": bong.fg_bong,
    "fg-l-bong": bong.fg_bong,
    "fg-rep-bong": bong.fg_reparam_bong,
    "fg-rep-l-bong": bong.fg_reparam_bong,
    "fg-blr": blr.fg_blr,
    "fg-bog": bog.fg_bog,
    "fg-bbb": bbb.fg_bbb,
    "fg-rep-bbb": bbb.fg_reparam_bbb,
}

AGENT_DICT_BASE = {
    'bong-fc': {'constructor': bong.fg_bong, 
                'needs_lr': False,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bong-fc-mom': {'constructor': bong.fg_reparam_bong, 
                'needs_lr': False,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bong-diag': {'constructor': bong.dg_bong, 
                'needs_lr': False,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bong-diag-mom': {'constructor': bong.dg_reparam_bong, 
                'needs_lr': False,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bong-dlr': {'constructor': bong.dlrg_bong, 
                'needs_lr': False,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': True,
                },
#################
#################
     'blr-fc': {'constructor': blr.fg_blr, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': False,
                'needs_rank': False,
                },
    'blr-fc-mom': {'constructor': blr.fg_reparam_blr, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': False,
                'needs_rank': False,
                },
    'blr-diag': {'constructor': blr.dg_blr, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'blr-diag-mom': {'constructor': blr.dg_reparam_blr, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'blr-dlr': {'constructor': blr.dlrg_blr, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': True,
                },
    ###########
#############
     'bog-fc': {'constructor': bog.fg_bog, 
                'needs_lr': True,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': False,
                'needs_rank': False,
                },
    'bog-fc-mom': {'constructor': bog.fg_reparam_bog, 
                'needs_lr': True,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': False,
                'needs_rank': False,
                },
    'bog-diag': {'constructor': bog.dg_bog, 
                'needs_lr': True,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bog-diag-mom': {'constructor': bog.dg_reparam_bog, 
                'needs_lr': True,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bog-dlr': {'constructor': bog.dlrg_bog, 
                'needs_lr': True,
                'needs_niter': False,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': True,
                },
###########
#############
     'bbb-fc': {'constructor': bbb.fg_bbb, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': False,
                'needs_rank': False,
                },
    'bbb-fc-mom': {'constructor': bbb.fg_reparam_bbb, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': False,
                'needs_rank': False,
                },
    'bbb-diag': {'constructor': bbb.dg_bbb, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bbb-diag-mom': {'constructor': bbb.dg_reparam_bbb, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': False,
                },
    'bbb-dlr': {'constructor': None, 
                'needs_lr': True,
                'needs_niter': True,
                'needs_nsamples': True,
                'linplugin': False,
                'empirical_fisher': True,
                'needs_rank': True,
                },
}

def convert_to_linplugin(args):
    args = args.copy()
    args['linplugin'] = True
    args['needs_nsamples'] = False
    args['empirical_fisher'] = False
    return args


LIN_DICT = { f'{agent}-lin': convert_to_linplugin(args) for (agent, args) in AGENT_DICT_BASE.items() }
AGENT_DICT = AGENT_DICT_BASE | LIN_DICT

AGENT_NAMES = AGENT_DICT.keys()

def make_agent_df(AGENT_DICT):
    lst = []
    for agent, props in AGENT_DICT.items():
        props['agent']=agent
        lst.append(props)

    df = pd.DataFrame(lst)
    return df