import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_rare_metabolite_statistics(metabolite_features):
    print(f"Are there nan values: {metabolite_features.isna().any().any()}. Filling any nans with zeros.")
    plt.rc('xtick', labelsize=6)
    (((metabolite_features.shape[0] - (metabolite_features.round(decimals=8) == 0).sum(axis=0))
      / metabolite_features.shape[0]) * 100).round(0).sort_values(
        ascending=True).plot.bar(figsize=(16, 6), title='% samples non-zero value observed, per metabolite.')
    plt.show()
    (metabolite_features.shape[0] - (metabolite_features.round(decimals=8) == 0).sum(axis=0)
     ).sort_values(ascending=True).plot.bar(
        figsize=(16, 6), title='#number of samples non-zero value observed, per metabolite.')
    plt.show()


def preprocess_filter_rare_metabolite(metabolite_features: pd.DataFrame, verbose=True, percentage=85) -> pd.DataFrame:
    """
    Preprocess the metabolite: remove rare taxa.
    Metabolites that was found in <percentage of the samples is being removed.
    If verbose, plot and print statistics.

    :param metabolite_features:         (pd.DataFrame) Observed .
    :param verbose:                     (bool) Whether to plot and print statistics on the data.
    :param percentage:                  (int) The minimal percentage of samples to require.
    :return:                            (pd.DataFrame) Observed, with rare metabolite removed.
    """
    min_number_of_samples = int((metabolite_features.shape[0] / 100) * percentage)
    if verbose:
        plot_rare_metabolite_statistics(metabolite_features)
    non_rare_columns = metabolite_features.columns[((metabolite_features.shape[0] - (
                metabolite_features.round(decimals=8) == 0).sum(axis=0)) >= min_number_of_samples)]
    if verbose:
        print(f"There are {len(non_rare_columns)} metabolite with sufficient number of samples "
              f"(>{percentage}%) out of {metabolite_features.shape[1]} metabolites.")
    return metabolite_features[non_rare_columns]


class MetabolitePreprocessor(object):
    _HMDB_IDS_TO_NAME_PATH = 'h2m_translation/map_hmdb_id_to_name.pkl'
    _HMDB_NAMES_DESCRIPTION_PATH = 'h2m_translation/hmdb_name_and_description.pkl'

    """
    Metabolite preprocessor.
    Assuming the data is already processed in some basic way (Not raw metabolite features), including
    pqn normalization (probabilistic quotient normalization to an internal standard (m/z 278.189; real time, 3.81 min)
    for subsequent analysis: https://pubmed.ncbi.nlm.nih.gov/16808434/) and alike.

    Preprocesses metabolite data, to enable valid analysis:
        dropping rare metabolites (metabolites with not enough samples for analysis)
        preform log transform
        handle missing values/ zero values (assuming there is a detection threshold, and therefore, there are not
        really zero values in the samples, but very low values. Usually common setting for nans or zeros in this case
         to half of the min value per metabolite.)
         
         If HMDB, map from HMDB ids to names. 
    """
    def __init__(self, verbose=True, percentage=85):
        self.verbose = verbose
        self.percentage = percentage

    def preprocess(self, metabolite_features, is_hmdb=True, project_dir='/home/noa/lab_code/H2Mtranslation'):
        metabolite_features = preprocess_filter_rare_metabolite(metabolite_features, self.verbose, self.percentage)
        min_value_per_metabolite = metabolite_features.replace(to_replace=0, value=np.nan).min(axis=0) / 2
        metabolite_features.replace(to_replace=np.nan, value=0, inplace=True)
        metabolite_features.replace(to_replace=0, value=min_value_per_metabolite, inplace=True)
        metabolite_features = metabolite_features.apply(lambda x: np.log(x + 1))
        if is_hmdb:
            map_hmdb_id_to_name = pd.read_pickle(f'{project_dir}/{MetabolitePreprocessor._HMDB_IDS_TO_NAME_PATH}')
            metabolite_features = metabolite_features.rename(columns=map_hmdb_id_to_name)
        return metabolite_features
