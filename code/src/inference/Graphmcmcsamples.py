from collections import defaultdict
import pickle
import os

from src.utils.SaveData import create_folder_and_files
import src.inference.mGG_Graphmcmc as rapid_Graphmcmc
# from src.inference import GG_Graphmcmc


def graphmcmcsamples(objmcmc, G, optional_arg=None, true_value=None, initial_value=None, output_dir=None, verbose=True):
    """
    Perform a Markov Chain Monte Carlo (MCMC) algorithm on the data assuming the graph model and specifications contained in the graphmcmc object.

    Args:
    ------
    Required args:
        objmcmc (graphmcmc): An object of the class graphmcmc, containing the graph model specifications and the parameters of the MCMC algorithm.
        G (sparse binary adjacency matrix): The graph represented as a sparse binary adjacency matrix.
        verbose (bool): Whether to print verbose output. Default is True.

    Optional args:
        true_value (dict): The true value of the parameters. Default is None.
        initial_value (dict): The initial value of the parameters. Default is None.
        output_dir (str): The path to the directory where the output files will be saved. Default is None.
        optional_arg (dict): Additional optional arguments. Default is None.

    Returns:
    ------
        graphmcmc: Updated graphmcmc object with the set of samples.
    """

    # Validate and prepare output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Create unique folder and destination
    unique_name = create_folder_and_files(objmcmc, output_dir)
    dest_unique_name = os.path.join(output_dir, unique_name)
    os.makedirs(dest_unique_name, exist_ok=True)

    objmodel = objmcmc.prior

    '''if objmodel.type == 'GG':
        if objmodel.typegraph in ['undirected', 'simple']:
            for k in range(objmcmc.settings["nchains"]):
                # print("-----------------------------------")
                # print(f"           MCMC chain {k}/{objmcmc.settings.nchains}        ")
                samples, stats,last = GG_Graphmcmc.GGgraphmcmc(
                    G, objmodel.param, objmcmc.settings, objmodel.typegraph,true_value=true_value,initial_value=initial_value, verbose=verbose)
                objmcmc.samples[k] = samples
                objmcmc.stats[k] = stats
                
                with open(os.path.join(dest_unique_name, f"{unique_name}_samples_{k}.pkl"), 'wb') as f:
                    pickle.dump(samples, f) 
                f.close()
                with open(os.path.join(dest_unique_name, f'{unique_name}_stats_{k}.pkl'), 'wb') as f:
                    pickle.dump(stats, f) 
                f.close()
                with open(os.path.join(dest_unique_name, f'{unique_name}_last_{k}.pkl'), 'wb') as f:
                    pickle.dump(last, f) 
                f.close()
        else:
            raise ValueError(f"Unknown type of graph {objmodel.typegraph}")'''

    if objmodel.type == "Rapid":
        if objmodel.typegraph in ['undirected', 'simple']:
            if optional_arg is None:
                optional_arg = defaultdict(lambda: None)
                optional_arg['nmass'] = 10
            for k in range(objmcmc.settings["nchains"]):
                # print("-----------------------------------")
                # print(f"           MCMC chain {k}/{objmcmc.settings.nchains} decoupling version        ")
                nmass = optional_arg["nmass"]
                samples, stats, last = rapid_Graphmcmc.Rapidgraphmcmc(
                    G, objmodel.param, objmcmc.settings, objmodel.typegraph, nmass, true_value, initial_value, verbose=verbose)

                with open(os.path.join(dest_unique_name, f'{unique_name}_samples_{k}.pkl'), 'wb') as f:
                    pickle.dump(samples, f)
                f.close()
                with open(os.path.join(dest_unique_name, f'{unique_name}_stats_{k}.pkl'), 'wb') as f:
                    pickle.dump(stats, f)
                f.close()
                with open(os.path.join(dest_unique_name, f'{unique_name}_last_{k}.pkl'), 'wb') as f:
                    pickle.dump(last, f)
                f.close()

                objmcmc.samples[k] = samples
                objmcmc.stats[k] = stats
                objmcmc.last[k] = last
        else:
            raise ValueError(f"Unknown type of graph {objmodel.typegraph}")
    else:
        raise NotImplementedError(
            f"Inference not implemented for graph model of type {objmodel.type}")

    with open(os.path.join(dest_unique_name, f'{unique_name}_objmcmc.pkl'), 'wb') as f:
        pickle.dump(objmcmc, f)
    f.close()
    return objmcmc
