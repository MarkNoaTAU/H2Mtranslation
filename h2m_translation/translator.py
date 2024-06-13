import pandas as pd
import numpy as np
from enum import Enum
from functools import partial
from typing import Optional


def majority_vote(reference_genus_table):
    """
    Takes a reference genus table and returns the query genus based on majority vote.
    :param reference_genus_table: (pd.DataFrame) reference genus table (MGBC table for specific reference genus)
    :return: Taxonomic label for the query genus.
    """
    if reference_genus_table['query_genus'].isna().all():
        return np.nan
    return reference_genus_table.dropna().groupby('query_genus').apply(lambda x: len(x)).idxmax()


class Direction(Enum):
    H2M = 1
    M2H = 2


class DatabaseClosestTaxa(Enum):
    taxonomy = 'taxonomy'
    functional = 'all_annotations'
    kegg = 'KEGG.eggnog'


class AggregationFunction(Enum):
    majority = partial(majority_vote)

    def __call__(self, *args):
        self.value(*args)


class MGBC_Translator(object):
    _CLOSEST_TAXA_FILE = 'data/closest_tax.tsv'
    """
    Translate Mice to Human or Human to Mice genus by finding the closest bacteria.
    This translator using the MGBC toolkit:
    https://www.cell.com/cell-host-microbe/fulltext/S1931-3128(21)00568-0?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1931312821005680%3Fshowall%3Dtrue
    But instead of mapping in Species-level (as done in the article) it aggregate the mapping to genus level.
    """
    def __init__(self, direction: Direction, use_database: DatabaseClosestTaxa,
                 aggregation_function: AggregationFunction,
                 mgbc_toolkit_path: Optional[str] = '/home/noa/lab_code/MGBC-Toolkit',
                 short_taxonomic_naming: Optional[bool] = True):
        """

        :param direction:               (Direction) direction of translation (from which host to which host)
        :param use_database:            (DatabaseClosestTaxa) To define the distance metric a database have been
                                        queried. MGBC support different functional or taxonomic databases.
        :param aggregation_function:    (AggregationFunction) Function/heuristic to aggregate the species level
                                       mapping  to genus level.
        """
        self.direction = direction
        self.use_database = use_database
        self.aggregation_function = aggregation_function
        self.file_path = f'{mgbc_toolkit_path}/{MGBC_Translator._CLOSEST_TAXA_FILE}'
        self.short_taxonomic_naming = short_taxonomic_naming

    def __repr__(self):
        return (f"MGBC Translator using direction: {self.direction}, method/database: {self.use_database}, and "
                f"aggregation heuristic: {self.aggregation_function}.")

    def __str__(self):
        return (f"MGBC Translator using direction: {self.direction}, method/database: {self.use_database}, and "
                f"aggregation heuristic: {self.aggregation_function}.")

    def load_table(self):
        mgbc_closest_taxa_table = pd.read_csv(self.file_path, sep='\t', header=None,
                                               names=['method', 'reference_genome', 'query_genome', 'distance',
                                                      'reference_taxonomy', 'query_taxonomy'])

        mgbc_closest_taxa_table['reference_genus'] = mgbc_closest_taxa_table.reference_taxonomy.str.extract(
            '(.*;g__[^;]+).*')
        mgbc_closest_taxa_table['query_genus'] = mgbc_closest_taxa_table.query_taxonomy.str.extract('(.*;g__[^;]+).*')
        return mgbc_closest_taxa_table

    def filter_table(self, closest_taxa_table):
        """"
        Define the lookup table based on the translation direction and database (that defines the distance):
        """
        # Define the lookup table based on the translation direction and database (that defines the distance):
        if self.direction == Direction.M2H:
            closest_taxa_table = closest_taxa_table.query('reference_genome.str.contains("MGBC")')
        elif self.direction == Direction.H2M:
            closest_taxa_table = closest_taxa_table.query('query_genome.str.contains("MGBC")')
        else:
            raise ValueError(f'Direction {self.direction} is not supported.')

        if self.use_database.value not in closest_taxa_table['method'].unique():
            raise ValueError(
                f'Must provide a database supported by MGBC, to define the closest taxa distance. '
                f'{self.use_database.value} is not supported.')
        method_database = self.use_database.value
        closest_taxa_table = closest_taxa_table.query('method == @method_database')
        return closest_taxa_table

    def translation_map(self):
        """

        :return: (pd.Series) Mapping from one host bacteria to the other (all available genus-to-genus mapping).
                            Based on the direction, database (method) and aggregation function defined.
        """
        closest_taxa_table = self.load_table()
        closest_taxa_table = self.filter_table(closest_taxa_table)
        # Future work: Support statistic extraction on the mapping before aggregating? How much they agree and so on?
        translation_map = closest_taxa_table.groupby('reference_genus').apply(lambda x:
                                                                              self.aggregation_function.value(x))
        if self.short_taxonomic_naming:
            translation_map = translation_map.rename(
                index=pd.Series(translation_map.index.unique(), index=translation_map.index.unique()).str.extract(
                    '.*g__(.*)').squeeze()).str.extract('.*g__(.*)').squeeze()
        return translation_map
