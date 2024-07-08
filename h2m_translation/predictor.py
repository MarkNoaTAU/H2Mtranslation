""" Metabolite level ML predictor """
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from utils import metabolite_to_str

def predict_metabolite_level_v0(relative_abundance, metabolite_features, metabolite_name, metadata_host_subject_id,
                                host='mice', random_state=10, verbose=False, dir_name=None):
    dir_name = host if dir_name is None else dir_name
    # we set the number of splits as the number of group we have to do leave-one-subject out cross validation!
    # In OSA we have 8 mice (control subjects) so n_splits will be 16, in iHMP we have 90 subjects so the number will be 90.
    n_splits = len(metadata_host_subject_id.unique())

    reg = RandomForestRegressor(random_state=random_state)
    scaler = StandardScaler()

    gkf = GroupKFold(n_splits=n_splits)
    X = relative_abundance
    y = metabolite_features[metabolite_name]
    groups = metadata_host_subject_id

    reg_list = []
    scaler_list = []
    pred_test_before_inverse_transform = pd.Series()
    pred_test = pd.Series()
    # r2 is the coefficient of determination and define as (1-u/v) where u is ((y_true - y_pred)** 2).sum() and v is ((y_true - y_true.mean()) ** 2).sum()
    for train_index, test_index in gkf.split(X=X, y=y, groups=groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # We want to apply scaler transformation on the target!
        y_train_scaled = scaler.fit_transform(y_train.to_frame())
        reg.fit(X_train, y_train_scaled.squeeze())
        y_pred_before_inverse_transform = reg.predict(X_test)
        # inverse transform the y prediction:
        y_pred = scaler.inverse_transform(y_pred_before_inverse_transform.reshape(-1, 1)).squeeze()

        pred_test_before_inverse_transform = pd.concat(
            [pred_test_before_inverse_transform, pd.Series(y_pred_before_inverse_transform, index=X_test.index)])
        pred_test = pd.concat([pred_test, pd.Series(y_pred, index=X_test.index)])

        reg_list.append(reg)
        scaler_list.append(scaler)
        if verbose:
            print(f"Number of train samples: {len(train_index)}, number of test samples: {len(test_index)}")
            print(f"Training set score: {reg.score(X_train, y_train)}")
            print(f"Test set score: {reg.score(X_test, y_test)}")

        # Log:
        metabolite_name_str = metabolite_to_str(metabolite_name)
        pred_test.to_frame(metabolite_name).to_pickle(
            f'metabolite_level_regressors/{dir_name}/pred_test_{metabolite_name_str}.pkl')

    return pred_test
