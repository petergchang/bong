from functools import partial

from bong.src import bbb, blr, bog, bong
import pandas as pd
from bong.util import safestr, unsafestr

AGENT_TYPES = ["fg_bong", "fg_l_bong", "fg_rep_bong", "fg_rep_l_bong",
               "fg_blr", "fg_bog", "fg_bbb", "fg_rep_bbb"]


LR_AGENT_TYPES = ["fg_blr", "fg_bog", "fg_rep_bog", "fg_bbb", "fg_rep_bbb"]


DIAG_BONG_DICT = {
    # BONG
    "dg-bong": bong.dg_bong,
    "dg-l-bong": partial(bong.dg_bong, linplugin=True),
    "dg-rep-bong": bong.dg_reparam_bong,
    "dg-rep-l-bong": partial(bong.dg_reparam_bong, linplugin=True),
    # BOG
    "dg-bog": bog.dg_bog,
    "dg-l-bog": partial(bog.dg_bog, linplugin=True),
    "dg-rep-bog": bog.dg_reparam_bog,
    "dg-rep-l-bog": partial(bog.dg_reparam_bog, linplugin=True),
    # BBB
    "dg-bbb": bbb.dg_bbb,
    "dg-l-bbb": partial(bbb.dg_bbb, linplugin=True),
    "dg-rep-bbb": bbb.dg_reparam_bbb,
    "dg-rep-l-bbb": partial(bbb.dg_reparam_bbb, linplugin=True),
    # BLR
    "dg-blr": blr.dg_blr,
    "dg-l-blr": partial(blr.dg_blr, linplugin=True),
    "dg-rep-blr": blr.dg_reparam_blr,
    "dg-rep-l-blr": partial(blr.dg_reparam_blr, linplugin=True),
}


DLR_BONG_DICT = {
    # BONG
    "dlrg-bong": bong.dlrg_bong,
    "dlrg-l-bong": partial(bong.dlrg_bong, linplugin=True),
    # BOG
    "dlrg-bog": bog.dlrg_bog,
    "dlrg-l-bog": partial(bog.dlrg_bog, linplugin=True),
    # BBB
    "dlrg-bbb": bbb.dlrg_bbb,
    "dlrg-l-bbb": partial(bbb.dlrg_bbb, linplugin=True),
    # BLR
    "dlrg-blr": blr.dlrg_blr,
    "dlrg-l-blr": partial(blr.dlrg_blr, linplugin=True),
}


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

def make_agent_constructor(algo, param):
    if algo == "bong":
        if param == "fc": return  bong.fg_bong
        if param == "fc_mom": return  bong.fg_reparam_bong
        if param == "diag": return  bong.dg_bong
        if param == "diag_mom": return  bong.dg_reparam_bong
        if param == "dlr": return  bong.dlrg_bong

    if algo == "blr":
        if param == "fc": return  blr.fg_blr
        if param == "fc_mom": return  blr.fg_reparam_blr
        if param == "diag": return  blr.dg_blr
        if param == "diag_mom": return  blr.dg_reparam_blr
        if param == "dlr": return  blr.dlrg_blr

    if algo == "bog":
        if param == "fc": return  bog.fg_bog
        if param == "fc_mom": return  bog.fg_reparam_bog
        if param == "diag": return  bog.dg_bog
        if param == "diag_mom": return  bog.dg_reparam_bog
        if param == "dlr": return  bog.dlrg_bog
        
    if algo == "bbb":
        if param == "fc": return  bbb.fg_bbb
        if param == "fc_mom": return  bbb.fg_reparam_bbb
        if param == "diag": return  bbb.dg_bbb
        if param == "diag_mom": return  bbb.dg_reparam_bbb
        if param == "dlr": return  bbb.dlrg_bbb



def needs_rank(algo, param, lin):
    return (param == "dlr")

def needs_ef(algo, param, lin):
    return not(lin)

def needs_nsample(algo, param, lin):
    return not(lin)

def needs_niter(algo, param, lin):
    if (algo == "bong") or (algo == "bog"):
        return False
    else:
        return True

def needs_lr(algo, param, lin):
    if (algo == "bong"):
        return False
    else:
        return True


def make_agent_args(algo, param, lin, rank, ef, nsample, niter, lr):
    invalid = 99
    args = {'algo': algo, 'param': param, 'lin': lin}
    args['dlr_rank'] = (rank if needs_rank(algo, param, lin) else invalid)
    args['ef'] = (ef if needs_ef(algo, param, lin) else invalid)
    args['nsample'] = (nsample if needs_nsample(algo, param, lin) else invalid)
    args['niter'] = (niter if needs_niter(algo, param, lin) else invalid)
    args['lr'] = (lr if needs_lr(algo, param, lin) else invalid)
    return args


## OLD stuff below

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




def extract_optional_agent_args(props, learning_rate, num_iter, num_sample, ef, rank):
    # Givem all the possible flag values, extract the ones needed for this agent.
    # This prevents us creating multiple agents with irrelevant arguments that differ,
    # which would cause us to create unnecessary jobs.
    args = props.copy()
    if props['lr'] is None: args['lr'] = learning_rate
    if props['niter'] is None: args['niter'] = int(num_iter)
    if props['nsample'] is None: args['nsample'] = int(num_sample)
    if props['dlr_rank'] is None: args['dlr_rank'] = int(rank)
    if props['ef'] is None: args['ef']= int(ef) 
    args['linplugin'] = props['linplugin'] # derived from agent name, not a flag
    del args['constructor']
    return args

# This must match keys of AGENT_DICT_LIN
def make_agent_name_from_parts(algo, param, lin):
    if lin:
        agent = f'{algo}_{param}_lin'
    else:
        agent = f'{algo}_{param}'
    return agent


## OLD stuff below

def make_agent_df(AGENT_DICT):
    lst = []
    for agent, props in AGENT_DICT.items():
        props['agent']=agent
        lst.append(props)

    df = pd.DataFrame(lst)
    return df

def parse_agent_full_name_old(s):
    # example input: 'bong_fc-MC10-I0-LR0_05-EF1-Lin1-R10'
    parts = s.split('-')
    if len(parts)==6:
        agent, mc, niter, lr, ef, lin = parts
        rank = 0
    else:
        agent, mc, niter, lr, ef, lin, rank = parts
        rank = rank[1:]
    mc, niter, lr, ef, lin = mc[2:], niter[1:], lr[2:], ef[2:], lin[3:]

    parts = agent.split('_') # blr_diag or blr_diag_mom into
    algo = parts[0]
    param = "_".join(parts[1:])
    return {
        'name': s, 
        'algo': algo,
        'param': param,
        'mc': int(mc),
        'niter': int(niter),
        'lr': unsafestr(lr),
        'ef': int(ef),
        'lin': int(lin),
        'rank': int(rank)
        }


