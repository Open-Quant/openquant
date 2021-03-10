# File written by Sean and Soren on July 7 2020
# In reference to: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Labelling/Trend-Follow-Question.ipynb
# Data from here: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Labelling/sample_dollar_bars.csv

# Imports
import pandas as pd
import mlfinlab as ml
import sys
import mlfinlab.data_structures as ds
import numpy as np
import os
import datetime
import math
import sklearn as sk
from mlfinlab.datasets import (load_tick_sample, load_stock_prices, load_dollar_bar_sample)
import matplotlib.pyplot as plt
import pyfolio as pf


if __name__ == '__main__':
    # Initialize Data
    # Set index
    csv_name = sys.argv[1]
    data = pd.read_csv(csv_name)
    data['date_time'] = pd.to_datetime(data.date_time)
    data.set_index('date_time', drop=True, inplace=True)
    print(data)
    input('raw data')

    ### Fit the primary model: Trend Following
    # Set moving average windows
    fast_window = 12
    slow_window = 26

    data['fast_mavg'] = data['close'].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
    data['slow_mavg'] = data['close'].rolling(window=slow_window, min_periods=slow_window, center=False).mean()
    print(data['fast_mavg'])
    input('fast mavg')
    print(data['slow_mavg'])
    input('slow mavg')

    # Compute sides
    # 1 for long signals and -1 for short (fast<slow)
    data['side'] = np.nan
    long_signals = data['fast_mavg'] >= data['slow_mavg']
    short_signals = data['fast_mavg'] < data['slow_mavg']
    data.loc[long_signals, 'side'] = 1
    data.loc[short_signals, 'side'] = -1
    data = data.dropna()
    print(long_signals)
    input('long signals')
    print(short_signals)
    input('short signals')
    print(data)
    input('data')

    # 1) Remove look biase by lagging the signal
    # 2) Save the raw data
    data['side'] = data['side'].shift(1)
    raw_data = data.copy()

    # Set daily volatility
    # TODO make util function
    daily_vol = ml.util.get_daily_vol(close=data['close'], lookback=50)
    print(daily_vol)
    input('daily vol')

    # Convert from daily vol to hourly vol (since our data in hourly)
    trading_hours_in_day = 8
    trading_days_in_year = 252
    hourly_vol = daily_vol / math.sqrt(trading_hours_in_day * trading_days_in_year)
    hourly_vol_mean = hourly_vol.mean()
    print(hourly_vol)
    input('hourly vol')

    # Apply symetric CUSUM filter and get timestamps for events
    cusum_events = ml.filters.cusum_filter(data['close'], threshold=hourly_vol_mean * 0.5)
    print(cusum_events)
    input('cusum events')

    # Compute vertical barrier
    vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events, close=data['close'], num_days=1)
    print(vertical_barriers)
    input('vertical barriers')

    # Setting profit and stop loss
    pt_sl = [1,2]
    min_ret = 0.005

    # Get the triple barrier events from mlfinlab function
    triple_barrier_events = ml.labeling.get_events(close=data['close'],
                                                t_events=cusum_events,
                                                pt_sl=pt_sl,
                                                #check between hourly and daily vol
                                                target=daily_vol,
                                                min_ret=min_ret,
                                                num_threads=3,
                                                vertical_barrier_times=vertical_barriers,
                                                side_prediction=data['side'],
                                                verbose=True
                                                )
    print(triple_barrier_events)
    input('triple barrier events')

    # Write the newly created events to a csv                                               
    triple_barrier_events.to_csv('barrier_events.csv', index=False)

    # Compute labels and output labels
    # check to make sure we are pulling the right column
    labels = ml.labeling.get_bins(triple_barrier_events, data['close'])
    print(labels)
    input('labels')
    
    ### Results of Primary Model (testing accuracy of predictions)
    # Set variables for analysis
    primary_forecast = pd.DataFrame(labels['bin'])
    primary_forecast['pred'] = 1
    primary_forecast.columns = ['actual', 'pred']

    # Performance Metrics
    actual = primary_forecast['actual']
    pred = primary_forecast['pred']

    # Output Statements for analysis
    print('Classification Report')
    print(sk.metrics.classification_report(y_true=actual, y_pred=pred, zero_division=False))
    print("Confusion Matrix")
    print(sk.metrics.confusion_matrix(actual, pred))
    print("Accuracy")
    print(sk.metrics.accuracy_score(actual, pred))

    ### Fit the Meta model (Train forest model)
    # Get the log returns
    raw_data['log_ret'] = np.log(raw_data['close']).diff()

    #  Create new momentum columns
    raw_data['mom1'] = raw_data['close'].pct_change(periods=1)
    raw_data['mom2'] = raw_data['close'].pct_change(periods=2)
    raw_data['mom3'] = raw_data['close'].pct_change(periods=3)
    raw_data['mom4'] = raw_data['close'].pct_change(periods=4)
    raw_data['mom5'] = raw_data['close'].pct_change(periods=5)

    # Create new volatility columns
    raw_data['volatility_50'] = raw_data['log_ret'].rolling(window=50, min_periods=50, center=False).std()
    raw_data['volatility_31'] = raw_data['log_ret'].rolling(window=31, min_periods=31, center=False).std()
    raw_data['volatility_15'] = raw_data['log_ret'].rolling(window=15, min_periods=15, center=False).std()

    # Serial Correlation (Takes about 4 minutes)
    window_autocorr = 50

    raw_data['autocorr_1'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
    raw_data['autocorr_2'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
    raw_data['autocorr_3'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
    raw_data['autocorr_4'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
    raw_data['autocorr_5'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

    # Get the various log -t returns
    raw_data['log_t1'] = raw_data['log_ret'].shift(1)
    raw_data['log_t2'] = raw_data['log_ret'].shift(2)
    raw_data['log_t3'] = raw_data['log_ret'].shift(3)
    raw_data['log_t4'] = raw_data['log_ret'].shift(4)
    raw_data['log_t5'] = raw_data['log_ret'].shift(5)

    # Recompute sides
    raw_data['side'] = np.nan
    long_signals = raw_data['fast_mavg'] >= raw_data['slow_mavg']
    short_signals = raw_data['fast_mavg'] < raw_data['slow_mavg']
    raw_data.loc[long_signals, 'side'] = 1
    raw_data.loc[short_signals, 'side'] = -1

    # Remove look ahead bias
    raw_data = raw_data.shift(1)

    # Get features at event dates
    x = raw_data.loc[labels.index, :]

    # Drop unwanted columns
    x.drop(['open', 'high', 'low', 'close', 'cum_vol', 'cum_dollar', 'cum_ticks','fast_mavg', 'slow_mavg'], axis=1, inplace=True)

    y = labels['bin']

    # Print value counts
    print(y.value_counts())

    ### Balance Classes
    # Split data into training, validation and test sets
    raw_data = raw_data.dropna()
    x_training_validation = x
    y_training_validation = y

    # Function to create train, test, and split data from sci-kit learn
    x_train, x_validate, y_train, y_validate = sk.model_selection.train_test_split(
                                            x_training_validation, 
                                            y_training_validation,
                                            test_size=0.70,
                                            train_size=0.30,  
                                            shuffle=False)


    # Create train dataframe
    train_df = pd.concat([y_train,x_train], axis=1, join='inner')
    train_df['bin'].value_counts()

    # Upsample training data for 50/50 split
    majority = train_df[train_df['bin'] == 0]
    minority = train_df[train_df['bin'] == 1]

    new_minority = sk.utils.resample(minority,
                            replace=True,
                            n_samples=majority.shape[0], # to match majority class
                            random_state=42) # figure random state out

    train_df = pd.concat([majority, new_minority])
    train_df = sk.utils.shuffle(train_df, random_state=42)

    # Print value counts
    print(train_df)
    train_df['bin'].value_counts()

    # Create training data
    y_train = train_df['bin']
    x_train = train_df.loc[:, train_df.columns != 'bin']

    # Fit a model
    parameters = {'max_depth': [2, 3, 4, 5, 7],
                'n_estimators': [1, 10, 25, 50, 100],
                'random_state': [42] }

    def perform_grid_search(x_data, y_data):
        rf = sk.ensemble.RandomForestClassifier()

        clf = sk.model_selection.GridSearchCV(rf, parameters, cv=4, scoring='roc_auc', n_jobs=3)

        clf.fit(x_data, y_data)

        print(clf.cv_results_['mean_test_score'])
        print(clf.best_params_['n_estimators'])
        print(clf.best_params_['max_depth'])

        return clf.best_params_['n_estimators'], clf.best_params_['max_depth']

    # extract parameters
    n_estimator, depth = perform_grid_search(x_train, y_train)
    c_random_state = 42
    print(n_estimator, depth, c_random_state)

    # Refit a new model with best params, so we can see feature importance
    rf = sk.ensemble.RandomForestClassifier(max_depth=depth, n_estimators=n_estimator, random_state=c_random_state)
    rf.fit(x_train, y_train.values.ravel())

    ### Training Metrics
    y_pred_rf = rf.predict_proba(x_train)[:, 1]
    y_pred = rf.predict(x_train)
    fpr_rf, tpr_rf, _ = sk.metrics.roc_curve(y_train, y_pred_rf)
    print(sk.metrics.classification_report(y_train, y_pred))

    print("Confusion Matrix")
    print(sk.metrics.confusion_matrix(y_train, y_pred))

    print('')
    print("Accuracy")
    print(sk.metrics.accuracy_score(y_train, y_pred))

    # THIS ONLY WORKS IN JUPYTER NOTEBOOK. THIS CAN SHOW PRETTY GRAPHS.
    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()


    # Meta-label
    # Performance Metrics
    y_pred_rf = rf.predict_proba(x_validate)[:, 1]
    y_pred = rf.predict(x_validate)
    fpr_rf, tpr_rf, _ = sk.metrics.roc_curve(y_validate, y_pred_rf)
    print(sk.metrics.classification_report(y_validate, y_pred))

    print("Confusion Matrix")
    print(sk.metrics.confusion_matrix(y_validate, y_pred))

    print('')
    print("Accuracy")
    print(sk.metrics.accuracy_score(y_validate, y_pred))

    # THIS ONLY WORKS IN JUPYTER NOTEBOOK. THIS CAN SHOW PRETTY GRAPHS.
    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()

    print(x_validate.index.min())
    print(x_validate.index.max())

    # Primary model
    primary_forecast = pd.DataFrame(labels['bin'])
    primary_forecast['pred'] = 1
    primary_forecast.columns = ['actual', 'pred']
    print(labels)
    start = primary_forecast.index.get_loc('2011-08-02 19:31:14.387')
    end = primary_forecast.index.get_loc('2012-07-27 20:14:35.480') + 1

    subset_prim = primary_forecast[start:end]

    # Performance Metrics
    actual = subset_prim['actual']
    pred = subset_prim['pred']
    print(sk.metrics.classification_report(y_true=actual, y_pred=pred))

    print("Confusion Matrix")
    print(sk.metrics.confusion_matrix(actual, pred))

    print('')
    print("Accuracy")
    print(sk.metrics.accuracy_score(actual, pred))

    ### Feature Importance
    title = 'Feature Importance'
    figsize = (15,5)

    feat_imp = pd.DataFrame({'Importance':rf.feature_importances_})
    feat_imp['feature'] = x.columns
    feat_imp.sort_values(by="Importance", ascending=False, inplace=True)
    feat_imp = feat_imp

    feat_imp.sort_values(by='Importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)

    # Plotting works only in jupyter notebook
    # plt.xlabel('Feature Importance Score')
    # plt.show()

    ### PERFORMANCE TEAR SHEET

    # Without Meta Labeling

    def get_daily_returns(intraday_returns):
        """
        Daily returns for pyfolio
        """
        cum_rets = ((intraday_returns + 1).cumprod())
        # Downsample to daily
        daily_rets = cum_rets.resample('B').last()
        # Forward fill, Percent change, Drop NaN
        daily_rets = daily_rets.ffill().pct_change().dropna()
        return daily_rets

    valid_dates = x_validate.index
    base_rets = labels.loc[valid_dates, 'ret']
    primary_model_rets = get_daily_returns(base_rets)

    # Set up the function to extract the KPI's from pyfolio
    perf_func = pf.timeseries.perf_stats

    # Save the statistics in a dataframe
    perf_stats_all = perf_func(returns=primary_model_rets, 
                            factor_returns=None, 
                            positions=None,
                            transactions=None,
                            turnover_denom="AGB")

    perf_stats_df = pd.DataFrame(data=perf_stats_all, columns=['Primary Model'])
    pf.show_perf_stats(primary_model_rets)

    # With Meta Labeling

    meta_returns = labels.loc[valid_dates, 'ret'] * y_pred
    daily_meta_rets = get_daily_returns(meta_returns)

    # Save KPIs in a dataframe
    erf_stats_all = perf_func(returns=daily_meta_rets, 
                            factor_returns=None, 
                            positions=None,
                            transactions=None,
                            turnover_denom="AGB")

    perf_stats_df['Meta Model'] = perf_stats_all

    pf.show_perf_stats(daily_meta_rets)

    ### PERFORM OUT OF SAMPLE TEST

    # Meta Model Metrics
    # Extract data for out-of-sample (00S)
    x_oos = x
    y_oos = y

    # Performance Metrics
    y_pred_rf = rf.predict_proba(x_oos)[:, 1]
    y_pred = rf.predict(x_oos)
    fpr_rf, tpr_rf, _ = sk.metrics.roc_curve(y_oos, y_pred_rf)
    print(sk.metrics.classification_report(y_oos, y_pred))

    print("Confusion Matrix")
    print(sk.metrics.confusion_matrix(y_oos, y_pred))

    print('')
    print("Accuracy")
    print(sk.metrics.accuracy_score(y_oos, y_pred))

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()



