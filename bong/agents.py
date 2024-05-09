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


AGENT_DICT_MC = {
    'bong-fc': {'constructor': bong.fg_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bong-fc-mom': {'constructor': bong.fg_reparam_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bong-diag': {'constructor': bong.dg_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bong-diag-mom': {'constructor': bong.dg_reparam_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bong-dlr': {'constructor': bong.dlrg_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': None,
                },
#################
#################
     'blr-fc': {'constructor': blr.fg_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': False,
                'rank': 0,
                },
    'blr-fc-mom': {'constructor': blr.fg_reparam_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': False,
                'rank': 0,
                },
    'blr-diag': {'constructor': blr.dg_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'blr-diag-mom': {'constructor': blr.dg_reparam_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'blr-dlr': {'constructor': blr.dlrg_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': None,
                },
    ###########
#############
     'bog-fc': {'constructor': bog.fg_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': False,
                'rank': 0,
                },
    'bog-fc-mom': {'constructor': bog.fg_reparam_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': False,
                'rank': 0,
                },
    'bog-diag': {'constructor': bog.dg_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bog-diag-mom': {'constructor': bog.dg_reparam_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bog-dlr': {'constructor': bog.dlrg_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': None,
                },
###########
#############
     'bbb-fc': {'constructor': bbb.fg_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': False,
                'rank': 0,
                },
    'bbb-fc-mom': {'constructor': bbb.fg_reparam_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': False,
                'rank': 0,
                },
    'bbb-diag': {'constructor': bbb.dg_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bbb-diag-mom': {'constructor': bbb.dg_reparam_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': 0,
                },
    'bbb-dlr': {'constructor': bbb.dlrg_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': False,
                'ef': True,
                'rank': None,
                },
}

def convert_to_linplugin(args):
    args = args.copy()
    args['linplugin'] = True
    #args['needs_nsample'] = False
    args['nsample'] = 0
    args['ef'] = False
    return args


AGENT_DICT_LIN = { f'{agent}-lin': convert_to_linplugin(args) for (agent, args) in AGENT_DICT_MC.items() }
AGENT_DICT = AGENT_DICT_MC | AGENT_DICT_LIN # union of dicts (python 3.9+)

AGENT_NAMES = AGENT_DICT.keys()

def make_agent_df(AGENT_DICT):
    lst = []
    for agent, props in AGENT_DICT.items():
        props['agent']=agent
        lst.append(props)

    df = pd.DataFrame(lst)
    return df