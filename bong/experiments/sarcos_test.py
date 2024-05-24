from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import jax.numpy as jnp
import jax

from datasets import get_sarcos_data
from models import fit_linreg_baseline, fit_gauss_baseline, nll_gauss, nll_linreg
from do_job import *
from bong.util import run_rebayes_algorithm, get_gpu_name
#from bong.agents import AGENT_DICT, AGENT_NAMES, parse_agent_full_name, make_agent_name_from_parts
from bong.agents import make_agent_constructor
from datasets import make_dataset
from models import make_model

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
                        **model['model_kwargs'],
                        agent_key = subkey,
                        learning_rate = args.lr,
                        num_iter = args.niter,
                        num_samples = args.nsample,
                        linplugin = args.lin,
                        empirical_fisher = args.ef,
                        rank = args.rank
                    )

    key, subkey = jr.split(key)
    results, elapsed, summary = run_agent(subkey, agent, data, model)
    df = pd.DataFrame(results)
    mse = df['mse_te'].to_numpy()
    expected_mse = 31.08 # from sarcos_demo
    assert jnp.allclose(mse[-1], expected_mse, atol=1e-1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--dataset", type=str, default="sarcos") 
    parser.add_argument("--data_dim", type=int, default=0)
    parser.add_argument("--dgp_type", type=str, default="na") 
    parser.add_argument("--dgp_str", type=str, default="na")  
    parser.add_argument("--ntrain", type=int, default=0) # use all the data!
    parser.add_argument("--nval", type=int, default=0)
    parser.add_argument("--ntest", type=int, default=0)
    parser.add_argument("--add_ones", type=int, default=1) 
    parser.add_argument("--init_var", type=float, default=1000) # unregularized!

    parser.add_argument("--emission_noise", type=float, default=-1)
    parser.add_argument("--linreg_baseline", type=int, default=1)


    # Model parameters
    #parser.add_argument("--agent", type=str, default="bong_fc", choices=AGENT_NAMES)
    parser.add_argument("--algo", type=str, default="bong")
    parser.add_argument("--param", type=str, default="fc")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=100) 
    parser.add_argument("--ef", type=int, default=1)
    parser.add_argument("--lin", type=int, default=1)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="lin") # or mlp
    parser.add_argument("--model_str", type=str, default="")
    parser.add_argument("--use_bias", type=int, default=1) 
    parser.add_argument("--use_bias_layer1", type=int, default=1) 

    # results
    parser.add_argument("--dir", type=str, default="", help="directory to store results") 
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--key", type=int, default=0)


    args = parser.parse_args([])
    main(args)
