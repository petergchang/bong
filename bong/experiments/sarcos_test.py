import jax.numpy as jnp
import jax

from models import (
    fit_linreg_baseline,
    fit_gauss_baseline,
    nll_linreg,
    make_model,
    calc_mse,
)
from do_job import *
from bong.util import find_first_true
from bong.agents import make_agent_constructor
from datasets import make_dataset


def main(args):
    if isinstance(args.key, int):
        key = jr.PRNGKey(args.key)
    else:
        key = args.key
    key, subkey = jr.split(key)
    key, data = make_dataset(subkey, args)
    key, subkey = jr.split(key)
    key, model = make_model(subkey, args, data)

    constructor = make_agent_constructor(args.algo, args.param)
    key, subkey = jr.split(key)
    agent = constructor(
        **model["model_kwargs"],
        agent_key=subkey,
        learning_rate=args.lr,
        num_iter=args.niter,
        num_samples=args.nsample,
        linplugin=args.lin,
        empirical_fisher=args.ef,
        rank=args.rank,
    )

    key, subkey = jr.split(key)
    results, elapsed, summary = run_agent(subkey, agent, data, model)
    df = pd.DataFrame(results)
    print(df.columns)
    mse_agent = df["mse"].to_numpy()[-1]
    nll_agent = df["nlpd-pi"].to_numpy()[-1]

    nlpd_agent = df["nlpd-mc"].to_numpy()
    print(nlpd_agent)
    nans = jnp.isnan(nlpd_agent)
    T = find_first_true(nans)
    print("NLPD agent non-nan until T=", T, "ntrain = ", args.ntrain)

    mu_y, v_y = fit_gauss_baseline(data["X_tr"], data["Y_tr"])
    w, sigma2 = fit_linreg_baseline(data["X_tr"], data["Y_tr"], method="lstsq")
    prediction = data["X_te"] @ w
    mse_linreg_baseline = calc_mse(prediction, data["Y_te"])
    print(
        f"n={args.ntrain}, MSE linreg={mse_linreg_baseline:.3f}, MSE agent={mse_agent:.3f}"
    )

    obs_var = 0.1 * jnp.var(data["Y_tr"])
    nll_linreg_baseline = jnp.mean(
        jax.vmap(nll_linreg, (None, None, 0, 0))(w, obs_var, data["X_te"], data["Y_te"])
    )
    print(
        f"n={args.ntrain}, NLL linreg={nll_linreg_baseline:.3f}, NLL agent={nll_agent:.3f}"
    )
    # expected_mse = 31.08 # from sarcos_demo
    # assert jnp.allclose(mse[-1], expected_mse, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="sarcos")
    parser.add_argument("--data_dim", type=int, default=0)
    parser.add_argument("--dgp_type", type=str, default="na")
    parser.add_argument("--dgp_str", type=str, default="na")
    parser.add_argument("--ntrain", type=int, default=2000)  # use all the data!
    parser.add_argument("--nval", type=int, default=0)
    parser.add_argument("--ntest", type=int, default=0)
    parser.add_argument("--add_ones", type=int, default=1)
    parser.add_argument("--init_var", type=float, default=1)  # mildly regularized
    parser.add_argument("--emission_noise", type=float, default=-1)

    # Model parameters
    # parser.add_argument("--agent", type=str, default="bong_fc", choices=AGENT_NAMES)
    parser.add_argument("--algo", type=str, default="bong")
    parser.add_argument("--param", type=str, default="dlr")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--ef", type=int, default=1)
    parser.add_argument("--lin", type=int, default=1)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="mlp")
    parser.add_argument("--model_str", type=str, default="20_20_1")
    parser.add_argument("--use_bias", type=int, default=1)
    parser.add_argument("--use_bias_layer1", type=int, default=1)

    # results
    parser.add_argument(
        "--dir", type=str, default="", help="directory to store results"
    )
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--key", type=int, default=0)

    args = parser.parse_args()
    print(args)
    main(args)
