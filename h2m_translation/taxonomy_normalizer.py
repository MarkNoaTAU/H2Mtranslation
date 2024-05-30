import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


def plot_rare_taxa_statistics(taxa):
    plt.rc('xtick', labelsize=6)
    (((taxa.shape[0] - (taxa.round(decimals=8) == 0).sum(axis=0)) / taxa.shape[0]) * 100).round(0).sort_values(
        ascending=True).plot.bar(figsize=(16, 6), title='% samples non-zero value observed, per taxa.')
    plt.show()
    (taxa.shape[0] - (taxa.round(decimals=8) == 0).sum(axis=0)).sort_values(ascending=True).plot.bar(
        figsize=(16, 6), title='#number of samples non-zero value observed, per taxa.')
    plt.show()
    (taxa.shape[0] - (taxa.round(decimals=8) == 0).sum(axis=0)).plot.hist(
        title='number of non-zero samples per genus')
    plt.show()


def preprocess_filter_rare_taxa(taxa: pd.DataFrame, verbose=True, percentage=10) -> pd.DataFrame:
    """
    Preprocess the taxonomy: remove rare taxa.
    Taxa that was found in <percentage of the samples is being removed.
    Here, we filter according to non-zero values, on observed abundance, but it's more common to filter to
    some very small threshold on relative abundance to be less sensitive to sampling noise.
    If verbose, plot and print statistics.

    :param taxa:        (pd.DataFrame) Observed abundance.
    :param verbose:     (bool) Whether to plot and print statistics on the data.
    :param percentage:  (int) The minimal percentage of samples to require.
    :return:            (pd.DataFrame) Observed abundance, with rare taxa removed.
    """
    min_number_of_samples = int((taxa.shape[0] / 100) * percentage)
    if verbose:
        plot_rare_taxa_statistics(taxa)
        print(
            f"The number of genus/features that have non-zero values on > {percentage} % samples are: "
            f"{((taxa.shape[0] - (taxa.round(decimals=8) == 0).sum(axis=0)) >= min_number_of_samples).sum()}"
            f" out of {taxa.shape[1]} genus/features before-filtering.")

    non_rare_columns = taxa.columns[
        (taxa.shape[0] - (taxa.round(decimals=8) == 0).sum(axis=0)) >= min_number_of_samples]
    return taxa[non_rare_columns]


def plot_rare_taxa_statistics_relative_abundance(taxa, abundance_threshold):
    plt.rc('xtick', labelsize=6)
    ((((taxa > abundance_threshold).sum(axis=0)) / taxa.shape[0]) * 100).round(0).sort_values(
        ascending=True).plot.bar(figsize=(16, 6), title='% samples above abundance_threshold value observed, per taxa.')
    plt.show()
    ((taxa > abundance_threshold).sum(axis=0)).sort_values(ascending=True).plot.bar(
        figsize=(16, 6), title='#number of sample above abundance_threshold value observed, per taxa.')
    plt.show()
    ((taxa > abundance_threshold).sum(axis=0)).plot.hist(
        title='number of samples above threshold per genus')
    plt.show()


def preprocess_filter_rare_taxa_relative_abundance(taxa: pd.DataFrame, verbose=True, percentage=10,
                                                   abundance_threshold=0.001) -> pd.DataFrame:
    """
    Preprocess the taxonomy: remove rare taxa.
    Taxa that her quantile (set by quantile parameter, default 10) relative abundance is smaller than
    abundance threshold is filtered out. AKA we filter taxa that was not sample in at least quantile of the samples
    or that the abundance of the taxa in those samples are small.

    :param abundance_threshold: (float) The min relative abundance threshold for a taxa to be included in a sample.
    :param taxa:                (pd.DataFrame) Observed abundance.
    :param verbose:             (bool) Whether to plot and print statistics on the data.
    :param quantile:            (int) The minimal percentage of samples to require.
    :return:                    (pd.DataFrame) Observed abundance, with rare taxa removed.
    """
    min_number_of_samples = int((taxa.shape[0] / 100) * percentage)
    if verbose:
        plot_rare_taxa_statistics_relative_abundance(taxa, abundance_threshold)
        print(
            f"The number of genus/features that have non-zero values on > {percentage} % samples are: "
            f"{(((taxa > abundance_threshold).sum(axis=0)) >= min_number_of_samples).sum()}"
            f" out of {taxa.shape[1]} genus/features before-filtering.")

    non_rare_columns = taxa.columns[
        ((taxa > abundance_threshold).sum(axis=0)) >= min_number_of_samples]
    return taxa[non_rare_columns]


class TaxonomyNormalizer(ABC):
    def __init__(self, drop_unknown_taxa=False, unknown_taxa_sample_threshold=1.0):
        self.drop_unknown_taxa = drop_unknown_taxa
        self.unknown_taxa_sample_threshold = unknown_taxa_sample_threshold

    @abstractmethod
    def calculate_relative_abundance(self, observed_abundance):
        """

        :param observed_abundance: (pd.DataFrame) Observed abundance. Samples x Taxonomy columns.
        Taxonomy level may very, but usually expected genus/species level of annotation. Taxa that was not
        identified in this level will be aggregated under the 'unknown' column.

        :return: (pd.DataFrame) relative abundance, samples x Taxonomy columns. Where each sample taxonomy sum to 1,
        as it represent the relative abundance in the sample.
        """
        raise NotImplementedError

    def normalize(self, observed_abundance):
        """

        :param observed_abundance: (pd.DataFrame) Observed abundance. Samples x Taxonomy columns.
        Taxonomy level may very, but usually expected genus/species level of annotation. Taxa that was not
        identified in this level will be aggregated under the 'unknown' column.

        :return: (pd.DataFrame) relative abundance, samples x Taxonomy columns. Where each sample taxonomy sum to 1,
        as it represent the relative abundance in the sample.
        If, instance drop_unknown_taxa set to true will drop unknown column.
        Samples that their unknown taxa is larger than unknown_taxa_sample_threshold will be dropped as well.
        """

        relative_abundance = self.calculate_relative_abundance(observed_abundance)
        unknown_taxa_sample_threshold = self.unknown_taxa_sample_threshold
        # clean samples:
        relative_abundance = relative_abundance.query('Unknown <= @unknown_taxa_sample_threshold')
        # drop unknown, and recalculate the relative abundance (if we dropped, aka unknown_taxa_sample_threshold < 1)
        if self.drop_unknown_taxa:
            if unknown_taxa_sample_threshold:
                observed_abundance = observed_abundance.loc[relative_abundance.index, :].drop('Unknown', axis=1)
                relative_abundance = self.calculate_relative_abundance(observed_abundance)
                relative_abundance = relative_abundance
            else:
                relative_abundance = relative_abundance.drop('Unknown', axis=1)
        # sanity check:
        if not (relative_abundance.sum(axis=1).round(decimals=8) == 1).all():
            raise ValueError("Taxonomy normalization failed. Not in all samples relative abundance summed to 1.")
        return relative_abundance


class NaiveTaxonomyNormalizer(TaxonomyNormalizer):
    """
    Naive taxonomy normalizer, sometimes called in the literature TSS - total sum scaling.
    Each taxon is normalized by the sum of abundance taxa in the sample.
    """
    def __init__(self, drop_unknown_taxa=False, unknown_taxa_sample_threshold=1.0):
        super(NaiveTaxonomyNormalizer, self).__init__(drop_unknown_taxa=drop_unknown_taxa,
                                                      unknown_taxa_sample_threshold=unknown_taxa_sample_threshold)

    def calculate_relative_abundance(self, observed_abundance):
        return observed_abundance.divide(observed_abundance.sum(axis=1), axis=0)
