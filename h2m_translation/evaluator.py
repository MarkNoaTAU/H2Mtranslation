""" Evaluator for metabolite level predicton """
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from utils import metabolite_to_str
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors


def evaluation_report_metabolite_level_v0(metabolite_features, metabolite_name, host, pred_test, dir_name=None):
    dir_name = host if dir_name is None else dir_name

    y = metabolite_features[metabolite_name]

    # First align the indexes:
    pred_test, y_test = pred_test.align(y)
    # Spearman correlation:
    spearman_corr, p_value = stats.spearmanr(pred_test, y_test)
    # R2
    r2 = r2_score(pred_test, y_test)
    # RMSE
    rmse = mean_squared_error(pred_test, y_test)
    # MAPE
    MAPE = mean_absolute_percentage_error(pred_test, y_test)

    # Log:
    score_df = pd.Series(
        {'spearman_corr': spearman_corr, 'p_value': p_value, 'r2': r2, 'rmse': rmse, 'MAPE': MAPE}).to_frame(
        metabolite_name)
    metabolite_name_str = metabolite_to_str(metabolite_name)
    score_df.to_pickle(f'metabolite_level_regressors/{dir_name}/score_df_{metabolite_name_str}.pkl')

    # Lets generate an evaluation report!
    metadata_o = pd.read_csv('mice/haddad_osa/original/haddad_6weeks_metadata_matched.txt', sep='\t').set_index(
        '#SampleID')
    metadata_o = pd.concat([metadata_o['host_subject_id'].to_frame()
                               , metadata_o['Description'].str.extract(r'.*collection (\d+) of .*').squeeze().to_frame(
            'seq_sample_number'),
                            metadata_o['exposure_type'].map({'IHH': False, 'Air': True}).to_frame('control'),
                            metadata_o['cage_number'].to_frame()], axis=1)
    metadata_o = metadata_o[metadata_o.control]

    subject_to_color = pd.Series(list(mcolors.TABLEAU_COLORS.values())[:len(metadata_o['host_subject_id'].unique())],
                                 index=metadata_o['host_subject_id'].unique())
    metadata_o['subject_color'] = metadata_o['host_subject_id'].apply(lambda x: subject_to_color[x])

    cage_to_color = pd.Series(list(mcolors.TABLEAU_COLORS.values())[:len(metadata_o['cage_number'].unique())],
                              index=metadata_o['cage_number'].unique())
    metadata_o['cage_color'] = metadata_o['cage_number'].apply(lambda x: cage_to_color[x])

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    pd.DataFrame({'y_test': y_test, 'pred_test': pred_test}).plot.kde(ax=axes[0][0])

    pd.DataFrame({'y_test': y_test, 'pred_test': pred_test}).plot.scatter(x='pred_test', y='y_test', ax=axes[1][0],
                                                                          title='Metabolite levels')

    pd.DataFrame({'y_test': y_test, 'pred_test': pred_test}).plot.scatter(x='pred_test', y='y_test', ax=axes[1][1],
                                                                          c=metadata_o['cage_color'].reindex(
                                                                              y_test.index),
                                                                          title='Metabolite levels by cage')

    pd.DataFrame({'y_test': y_test, 'pred_test': pred_test}).plot.scatter(x='pred_test', y='y_test', ax=axes[1][2],
                                                                          c=metadata_o['subject_color'].reindex(
                                                                              y_test.index),
                                                                          title='Metabolite levels by subject')


    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='jpg', pad_inches=0.1, edgecolor='gray', bbox_inches='tight')
    # Avoid displaying the figure when calling this function:
    plt.close(fig)

    encoded_fig = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = ('<html> \n <title> Evaluation Report </title> \n ' +
            '<body> \n '
            f' <p><b><u> Metabolite name (Target):</u> {metabolite_name} </b></p>'
            ' <p> Test scores: </p>'
            f'<p> Spearmans correlation: {spearman_corr}, p-value: {p_value} </p>'
            f'<p> R-squared: {r2} </p>'
            f'<p> RMSE: {rmse} </p>'
            f'<p> MAPE: {MAPE} \n </p> '
            ' </body> ' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded_fig) + '</html>')

    metabolite_name_str = metabolite_to_str(metabolite_name)
    with open(f'metabolite_level_regressors/{dir_name}/eval_report_{metabolite_name_str}_RF_default.html', 'w') as f:
        f.write(html)
    return html


def evaluation_report_metabolite_level_all_v0(score_dataframes, metabolite_features, host='mice', dataset_name='OSA',
                                              dir_name=None,
                                              robust_well_predicted_path='/home/noa/lab_code/H2Mtranslation/h2m_translation/map_hmdb_id_to_name.pkl'):
    dir_name = host if dir_name is None else dir_name

    # Basic statistics:
    figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    score_dataframes.loc['spearman_corr', :].plot.hist(title=f'spearman corr of predicted {host} metabolite level', ax=ax1)
    score_dataframes.loc['MAPE', :].plot.hist(title=f'MAPE of predicted {host} metabolite level', ax=ax2)
    score_dataframes.loc['rmse', :].plot.hist(title=f'rmse of predicted {host} metabolite level', ax=ax3)

    tmpfile = BytesIO()
    # Avoid displaying the figure when calling this function:
    plt.savefig(tmpfile, format='jpg', pad_inches=0.1, edgecolor='gray', bbox_inches='tight')
    plt.close(figure)

    score_df_figs = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    # Define well-predicted metabolites:
    # extract the p-value and spearman correlation and preform FDR correction:
    spearman_corr = score_dataframes.loc['spearman_corr', :]
    p_values = score_dataframes.loc['p_value', :]

    # Define the well-predicted metabolties (Metabolites with a predictability of ρ > 0.3 and an FDR < 0.1 were
    # referred to as ‘well-predicted’ metabolites.)
    rejected, p_values_corrected = fdrcorrection(p_values, alpha=0.1)
    score_dataframes.loc['p_values_corrected', :] = p_values_corrected
    well_predicted_metabolites = score_dataframes.columns[
        (score_dataframes.loc['p_values_corrected', :] < 0.1) & (score_dataframes.loc['spearman_corr', :] > 0.3)]

    # Compare to Efrat's paper "Robust Well predicted metabolites):
    map_hmdb_id_to_name = pd.read_pickle(robust_well_predicted_path)
    universal_robust_well_predicted = pd.read_excel(
        "supplementary_robustness and universality of gut microbiome-metabolome associations.xlsx",
        sheet_name='Table S6', skiprows=9)


    map_name_to_hmdb = pd.Series(map_hmdb_id_to_name.index, index=map_hmdb_id_to_name)
    metabolite_features_hmdb_labeled = metabolite_features.rename(columns=map_name_to_hmdb)

    metabolites_shared = metabolite_features_hmdb_labeled.columns.intersection(
        pd.Index(universal_robust_well_predicted['HMDB ID']))
    score_dataframes_shared = score_dataframes.rename(columns=map_name_to_hmdb).loc[:, metabolites_shared]
    universal_robust_well_predicted_shared = universal_robust_well_predicted[
        universal_robust_well_predicted['HMDB ID'].isin(metabolites_shared)]

    # For the discrete comparison I would like to look on "Robust (1)" true/ false confusion matrix
    # confusion matrix:

    universal_robust_metabolite = universal_robust_well_predicted_shared.set_index('HMDB ID')['Robust (1)']
    well_predicted_mice = (score_dataframes_shared.loc['p_values_corrected', :] < 0.1) & (
                score_dataframes_shared.loc['spearman_corr', :] > 0.3)

    # Align them!
    well_predicted_mice, universal_robust_metabolite = well_predicted_mice.align(universal_robust_metabolite, axis=0)

    cm = confusion_matrix(y_true=well_predicted_mice, y_pred=universal_robust_metabolite)
    cm_fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yess'],
                ax=ax)
    plt.xlabel('Universally Robust', fontsize=13)
    plt.ylabel(f'Well predicted in {host}', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)

    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='jpg', pad_inches=0.1, edgecolor='gray', bbox_inches='tight')
    # Avoid displaying the figure when calling this function:
    cm_fig_encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close(cm_fig)

    # For the continuous comparison I would like to look on "REM Mean Spearman Rho [95% confidence interval]" and plot the Universall (including the interval) vs OSA mice dataset coefficient + plot the (-log p_value_after_fdr correction) of universall vs. OSA dataset

    universal_metabolite_coef = universal_robust_well_predicted_shared.set_index('HMDB ID')[
        "REM Mean Spearman Rho [95% confidence interval]"].str.extract('(.*) \[.*').squeeze().astype(float)
    universal_metabolite_coef_confidence_interval_min = universal_robust_well_predicted_shared.set_index('HMDB ID')[
        "REM Mean Spearman Rho [95% confidence interval]"].str.extract('.* \[(.*),.*]').squeeze().astype(float)
    universal_metabolite_coef_confidence_interval_max = universal_robust_well_predicted_shared.set_index('HMDB ID')[
        "REM Mean Spearman Rho [95% confidence interval]"].str.extract('.* \[.*,(.*)]').squeeze().astype(float)
    well_predicted_mice_coef = score_dataframes_shared.loc['spearman_corr', :]

    # Need to align them! to have the same order of Metabolite!
    universal_metabolite_coef = universal_metabolite_coef.sort_index()
    universal_metabolite_coef_confidence_interval_min = universal_metabolite_coef_confidence_interval_min.sort_index()
    universal_metabolite_coef_confidence_interval_max = universal_metabolite_coef_confidence_interval_max.sort_index()
    well_predicted_mice_coef = well_predicted_mice_coef.sort_index()

    yerr = pd.DataFrame({'min': universal_metabolite_coef_confidence_interval_min,
                         'max': universal_metabolite_coef_confidence_interval_max}).T
    yerr[yerr < 0] = 0

    universal_metabolite_pvalue_corrected = universal_robust_well_predicted_shared.set_index('HMDB ID')['REM Rho - FDR']
    well_predicted_mice_p_values_corrected = score_dataframes_shared.loc['p_values_corrected', :]

    universal_metabolite_pvalue_corrected, well_predicted_mice_p_values_corrected = universal_metabolite_pvalue_corrected.align(
        well_predicted_mice_p_values_corrected)
    universal_metabolite_pvalue_corrected = universal_metabolite_pvalue_corrected.apply(lambda x: - np.log(x))
    well_predicted_mice_p_values_corrected = well_predicted_mice_p_values_corrected.apply(lambda x: - np.log(x))

    plt.style.use('_mpl-gallery')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    axes[0].errorbar(x=well_predicted_mice_coef, y=universal_metabolite_coef, yerr=yerr, fmt='o', linewidth=2,
                     capsize=6, ecolor='blue')
    axes[1].scatter(x=well_predicted_mice_coef, y=universal_metabolite_coef)
    axes[2].scatter(x=well_predicted_mice_p_values_corrected, y=universal_metabolite_pvalue_corrected)

    axes[0].set_xlabel(f'Well predicted in {host}', fontsize=9)
    axes[0].set_ylabel('Universally Robust in Human', fontsize=9)
    axes[0].set_title('Prediction Score (Spearman coefficient)', fontsize=12)

    axes[1].set_xlabel(f'Well predicted in {host}', fontsize=9)
    axes[1].set_ylabel('Universally Robust in Human', fontsize=9)
    axes[1].set_title('Prediction Score (Spearman coefficient)', fontsize=12)

    axes[2].set_xlabel(f'Well predicted in {host}', fontsize=9)
    axes[2].set_ylabel('Universally Robust in Human', fontsize=9)
    axes[2].set_title('-log(P value FDR corrected)', fontsize=12)

    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='jpg', pad_inches=0.1, edgecolor='gray', bbox_inches='tight')

    # Avoid displaying the figure when calling this function:
    plt.close(fig)

    cont_analysis_fig = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cont_analysis_fig = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    report = ('<html> \n <title> Evaluation Report </title> \n ' +
              '<body> \n '
              f' <p><b><u> Evaluation Report of Metabolites Level prediction on {host}  </b></u></p>'
              f'<p> Prediction scores across all metabolites: </p>'
              + '<img src=\'data:image/png;base64,{}\'>'.format(score_df_figs) +
              "<p> Define the <b> well-predicted metabolite </b>  as metabolites with a predictability of ρ > 0.3 (aka spearman-coefficient larger them 0.3) and an FDR < 0.1. </p>"
              f" <p> Number of well predicted metabolites: {well_predicted_metabolites.shape[0]}, out of {metabolite_features.shape[1]} metabolites \n </p>"
              f" <p> Well predicted metabolites: {', '.join(well_predicted_metabolites.tolist())} </p>"
              f"<p> <u> Comparing the predictability of metabolites in {host} - {dataset_name} and  Human metabolites in Efrat's dataset: </u>"
              f"<p> There are {metabolite_features_hmdb_labeled.columns.intersection(pd.Index(universal_robust_well_predicted['HMDB ID'])).shape[0]} HMDB in intersection between my {dataset_name} dataset metabolites and Human metabolites in Efrat's dataset. </p> "
              F"<p> Our of {metabolite_features_hmdb_labeled.shape[1]} metabolites in OSA and {len(universal_robust_well_predicted['HMDB ID'].unique())} in the Human meta-analysis </p>"
              + '<img src=\'data:image/png;base64,{}\'>'.format(cm_fig_encoded) +
              "<p>\n \n </p>" +
              '<img src=\'data:image/png;base64,{}\'>'.format(cont_analysis_fig) +
              '</body>'
              '</html>'
              )

    with open(f'metabolite_level_regressors/{dir_name}/eval_report_all_metabolites_RF_default.html', 'w') as f:
        f.write(report)

    return report

