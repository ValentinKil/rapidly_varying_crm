
"""
This module provides functionality to convert MCMC sampling results into an ArviZ InferenceData object, 
which is a standardized format for Bayesian inference data.
""" 
    
import arviz as az
import numpy as np

def to_arviz(self,nburn,thin):
    """
    Args:
    -----------
    self : object
        The object containing MCMC sampling results, metadata, and diagnostic statistics.
    nburn : int
        The number of burn-in samples to discard from the beginning of each chain.
    thin : int
        The thinning interval to apply to the samples.
        
    Returns:
    --------
    idata : arviz.InferenceData
        An ArviZ InferenceData object containing posterior samples, metadata, and diagnostic statistics.
    Notes:
    ------
    - The function assumes that the object (`self`) contains the following attributes:
        - `samples`: A list of dictionaries, where each dictionary represents a chain and contains parameter samples.
        - `settings`: A dictionary containing metadata about the MCMC settings.
        - `prior`: An object containing information about the prior distribution, with attributes `name`, `type`, `param`, and `typegraph`.
        - `stats`: A list of dictionaries, where each dictionary contains diagnostic statistics for a chain.
    - Metadata and diagnostic statistics are added to the InferenceData object as attributes.
    """
    samples = self.samples
    nchains = len(samples)
    
    # Get parameter names from the first chain
    param_names = list(samples[0].keys())
    
    # Collect all samples into an (nchains, ndraws) array for each parameter
    data = {
        param: np.array([chain[param][nburn//thin:] for chain in samples])  # Shape: (nchains, ndraws)
        for param in param_names
    }    
    # Create InferenceData from the dictionary
    idata = az.from_dict(posterior=data)
    
    # Extract metadata from objmcmc (settings, etc.)
    metadata = self.settings  # Assuming the metadata is in the settings attribute
    
    # Extract information from objmodel (self.prior)
    prior_info = {
        "prior_name": self.prior.name,
        "prior_type": self.prior.type,
        "prior_params": self.prior.param,
        "prior_typegraph": self.prior.typegraph
    }
    
    # Combine both objmcmc metadata and prior_info
    full_metadata = {**metadata, **prior_info}
    
    # Add metadata to InferenceData as attrs
    if full_metadata:
        idata.attrs['metadata'] = full_metadata
    
    # Extract statistics from objmcmc.stats
    stats = self.stats
    
    # Collect the same statistics for all chains (since all chains have the same statistics)
    diagnostic_stats = {}
    for key in stats[0].keys():  # Assumes all chains have the same keys
        diagnostic_stats[key] = np.array([chain[key] for chain in stats])
    
    # Add diagnostic statistics to InferenceData as attrs
    if diagnostic_stats:
        idata.attrs['diagnostics'] = diagnostic_stats
    
    return idata
