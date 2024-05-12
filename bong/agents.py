from bong.src import bbb, blr, bog, bong
import pandas as pd
from bong.util import safestr

AGENT_TYPES = ["fg_bong", "fg_l_bong", "fg_rep_bong", "fg_rep_l_bong",
               "fg_blr", "fg_bog", "fg_bbb", "fg_rep_bbb"]


LR_AGENT_TYPES = ["fg_blr", "fg_bog", "fg_rep_bog", "fg_bbb", "fg_rep_bbb"]


BONG_DICT = {
    "fg_bong": bong.fg_bong,
    "fg_l_bong": bong.fg_bong,
    "fg_rep_bong": bong.fg_reparam_bong,
    "fg_rep_l_bong": bong.fg_reparam_bong,
    "fg_blr": blr.fg_blr,
    "fg_bog": bog.fg_bog,
    "fg_bbb": bbb.fg_bbb,
    "fg_rep_bbb": bbb.fg_reparam_bbb,
}

# A None field means there is no fixed value for this agent
# so the value must be specified at runtime
AGENT_DICT_MC = {
    'bong_fc': {'constructor': bong.fg_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bong_fc_mom': {'constructor': bong.fg_reparam_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bong_diag': {'constructor': bong.dg_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bong_diag_mom': {'constructor': bong.dg_reparam_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bong_dlr': {'constructor': bong.dlrg_bong, 
                'lr': 0,
                'niter': 0,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': None,
                },
#################
#################
     'blr_fc': {'constructor': blr.fg_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'blr_fc_mom': {'constructor': blr.fg_reparam_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'blr_diag': {'constructor': blr.dg_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'blr_diag_mom': {'constructor': blr.dg_reparam_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'blr_dlr': {'constructor': blr.dlrg_blr, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': None,
                },
    ###########
#############
     'bog_fc': {'constructor': bog.fg_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bog_fc_mom': {'constructor': bog.fg_reparam_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bog_diag': {'constructor': bog.dg_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bog_diag_mom': {'constructor': bog.dg_reparam_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bog_dlr': {'constructor': bog.dlrg_bog, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': None,
                },
###########
#############
     'bbb_fc': {'constructor': bbb.fg_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bbb_fc_mom': {'constructor': bbb.fg_reparam_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bbb_diag': {'constructor': bbb.dg_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': 1,
                'dlr_rank': 0,
                },
    'bbb_diag_mom': {'constructor': bbb.dg_reparam_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': 0,
                },
    'bbb_dlr': {'constructor': bbb.dlrg_bbb, 
                'lr': None,
                'niter': None,
                'nsample': None,
                'linplugin': 0,
                'ef': None,
                'dlr_rank': None,
                },
}

def convert_to_linplugin(args):
    args = args.copy()
    args['linplugin'] = 1
    args['nsample'] = 0
    args['ef'] = 0
    return args


AGENT_DICT_LIN = { f'{agent}_lin': convert_to_linplugin(args) for (agent, args) in AGENT_DICT_MC.items() }
AGENT_DICT = AGENT_DICT_MC | AGENT_DICT_LIN # union of dicts (python 3.9+)

AGENT_NAMES = AGENT_DICT.keys()

def make_agent_df(AGENT_DICT):
    lst = []
    for agent, props in AGENT_DICT.items():
        props['agent']=agent
        lst.append(props)

    df = pd.DataFrame(lst)
    return df

