from functools import partial

from bong.src import bbb, blr, bog, bong

AGENT_TYPES = [
    "fg_bong",
    "fg_l_bong",
    "fg_rep_bong",
    "fg_rep_l_bong",
    "fg_blr",
    "fg_bog",
    "fg_bbb",
    "fg_rep_bbb",
]


LR_AGENT_TYPES = ["fg_blr", "fg_bog", "fg_rep_bog", "fg_bbb", "fg_rep_bbb"]


DIAG_BONG_DICT = {
    # BONG
    "dg-bong": partial(bong.dg_bong, empirical_fisher=True),
    "dg-l-bong": partial(bong.dg_bong, linplugin=True),
    "dg-rep-bong": partial(bong.dg_reparam_bong, empirical_fisher=True),
    "dg-rep-l-bong": partial(bong.dg_reparam_bong, linplugin=True),
    # BOG
    "dg-bog": partial(bog.dg_bog, empirical_fisher=True),
    "dg-l-bog": partial(bog.dg_bog, linplugin=True),
    "dg-rep-bog": partial(bog.dg_reparam_bog, empirical_fisher=True),
    "dg-rep-l-bog": partial(bog.dg_reparam_bog, linplugin=True),
    # BBB
    "dg-bbb": partial(bbb.dg_bbb, empirical_fisher=True),
    "dg-l-bbb": partial(bbb.dg_bbb, linplugin=True),
    "dg-rep-bbb": partial(bbb.dg_reparam_bbb, empirical_fisher=True),
    "dg-rep-l-bbb": partial(bbb.dg_reparam_bbb, linplugin=True),
    # BLR
    "dg-blr": partial(blr.dg_blr, empirical_fisher=True),
    "dg-l-blr": partial(blr.dg_blr, linplugin=True),
    "dg-rep-blr": partial(blr.dg_reparam_blr, empirical_fisher=True),
    "dg-rep-l-blr": partial(blr.dg_reparam_blr, linplugin=True),
}


DLR_BONG_DICT = {
    # BONG
    "dlrg-bong": partial(bong.dlrg_bong, empirical_fisher=True),
    "dlrg-l-bong": partial(bong.dlrg_bong, linplugin=True, empirical_fisher=True),
    # BOG
    "dlrg-bog": partial(bog.dlrg_bog, empirical_fisher=True),
    "dlrg-l-bog": partial(bog.dlrg_bog, linplugin=True, empirical_fisher=True),
    # BBB
    "dlrg-bbb": partial(bbb.dlrg_bbb, empirical_fisher=True),
    "dlrg-l-bbb": partial(bbb.dlrg_bbb, linplugin=True, empirical_fisher=True),
    # BLR
    "dlrg-blr": partial(blr.dlrg_blr, empirical_fisher=True),
    "dlrg-l-blr": partial(blr.dlrg_blr, linplugin=True, empirical_fisher=True),
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
        if param == "fc":
            return bong.fg_bong
        if param == "fc_mom":
            return bong.fg_reparam_bong
        if param == "diag":
            return bong.dg_bong
        if param == "diag_mom":
            return bong.dg_reparam_bong
        if param == "dlr":
            return bong.dlrg_bong

    if algo == "blr":
        if param == "fc":
            return blr.fg_blr
        if param == "fc_mom":
            return blr.fg_reparam_blr
        if param == "diag":
            return blr.dg_blr
        if param == "diag_mom":
            return blr.dg_reparam_blr
        if param == "dlr":
            return blr.dlrg_blr

    if algo == "bog":
        if param == "fc":
            return bog.fg_bog
        if param == "fc_mom":
            return bog.fg_reparam_bog
        if param == "diag":
            return bog.dg_bog
        if param == "diag_mom":
            return bog.dg_reparam_bog
        if param == "dlr":
            return bog.dlrg_bog

    if algo == "bbb":
        if param == "fc":
            return bbb.fg_bbb
        if param == "fc_mom":
            return bbb.fg_reparam_bbb
        if param == "diag":
            return bbb.dg_bbb
        if param == "diag_mom":
            return bbb.dg_reparam_bbb
        if param == "dlr":
            return bbb.dlrg_bbb


def needs_rank(algo, param, lin):
    return param == "dlr"


def needs_ef(algo, param, lin):
    return not (lin)


def needs_nsample(algo, param, lin):
    return not (lin)


def needs_niter(algo, param, lin):
    return algo not in ("bong", "bog")


def needs_lr(algo, param, lin):
    return algo != "bong"


def make_agent_args(algo, param, lin, rank, ef, nsample, niter, lr):
    invalid = 99
    args = {"algo": algo, "param": param, "lin": lin}
    args["dlr_rank"] = rank if needs_rank(algo, param, lin) else invalid
    args["ef"] = ef if needs_ef(algo, param, lin) else invalid
    args["nsample"] = nsample if needs_nsample(algo, param, lin) else invalid
    args["niter"] = niter if needs_niter(algo, param, lin) else invalid
    args["lr"] = lr if needs_lr(algo, param, lin) else invalid
    return args
