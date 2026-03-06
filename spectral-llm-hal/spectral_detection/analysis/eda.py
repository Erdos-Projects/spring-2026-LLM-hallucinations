import pandas as pd

# Valid categories 
VALID_DOMAINS = {
    "STEM",
    "Humanities",
    "Social Sciences",
    "Medicine & Health",
    "Law, Business, and Miscellaneous"
}

def compute_majority_valid_domain(domain_series):
    valid_subset = domain_series[domain_series.isin(VALID_DOMAINS)]
    
    if not valid_subset.empty:
        return valid_subset.mode().iloc[0]
    else:
        return pd.NA