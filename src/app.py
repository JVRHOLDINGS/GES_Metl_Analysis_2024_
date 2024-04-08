import numpy.linalg
import pandas as pd
import pathlib
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openpyxl
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import scipy
from scipy import stats, integrate
from scipy.stats import gaussian_kde
from dateutil.relativedelta import relativedelta

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
df = pd.read_excel(DATA_PATH.joinpath("metallics dataset.xlsx"))

app = Dash(__name__, title="Metallics Analysis", meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# VERSION 2 TO DO
# CLEAN CODE
# ALL RATIO/SIGNALS ANOTHER WAY OTHERWISE SIGNALS ARE VERY LOCAL
# MEAN REVERSION TIME GRAPHIC
# PRICE PREDICTION BASED REGRESSION ON RATIO/SPREAD LT AVE - DEVIATIONS * MEAN REVERSION
# GRAPH OF PRODUCT(S) + DISTRIBUTION OF RETURNS + SEASONALITY
# INPUT FOR FUTURES CURVES +> OUTPUT TO PRODUCT

# GLOBAL INPUTS
long_run_days = 3650
number_days = 30
corr_matrix_days = 45
weighting_type = 'square'

# PREPARE THE DATASET
data = df.copy()
raw_data = data.copy()
all_dates = data.iloc[:, 0]
dates = data.iloc[:, 0][:number_days]


def error_correct(data):
    window_size = 15
    threshold = 0.10
    flat_threshold = 0.20

    for col in data.columns.drop('Date'):
        # remove '' values
        data[col] = data[col].replace('', np.NaN)

        # generate rolling mean and std
        col_mean = data[col].rolling(window=window_size).mean()
        col_std = data[col].rolling(window=window_size).std()

        # check if difference between data point is beyond threshold, identify as error
        # data.loc[(data[col].diff().abs() > data[col] * flat_threshold), col] = np.NaN

        # check if value is 0 or negative, identify as error
        data.loc[data[col] <= 0, col] = np.NaN

        # after the window check if mean or std has changed beyond threshold, identify as error
        # data.loc[(col_mean.diff().abs() > col_mean * threshold) | (col_std.diff().abs() > col_std * threshold), col] = np.NaN

        # correct errors by filling NaN
        data[col] = data[col].ffill(limit=30)

    return data

data_model = error_correct(data)

# GENERATE DATA ANALYSIS

# GENERATE RATIOS AND SPREADS
data = data_model.drop('Date', axis=1)  # remove the first column 'Date'
col_names = list(data)  # get the column names from the dataset
cols = len(data.columns)  # number of columns in dataset

all_ratios = pd.DataFrame()
all_spreads = pd.DataFrame()

# hold fixed a column
for col_name in data.columns:
    reference_col = data_model[col_name]
    reference_spreads = pd.DataFrame()
    reference_ratios = pd.DataFrame()
    # cycle though all columns to give ratio and spread
    for col in data.columns:
        relative_col = data_model[col]

        spread = reference_col - relative_col
        ratio = reference_col / relative_col

        # if dividing by zero, NaN, handling when the data runs out per column
        ratio = ratio.replace([np.inf, -np.inf], np.NaN)

        reference_ratios[col] = ratio
        reference_spreads[col] = spread

    # combine all into a dataframe
    all_spreads = pd.concat([all_spreads, reference_spreads], axis=1)
    all_ratios = pd.concat([all_ratios, reference_ratios], axis=1)

    LT_ave_ratios = all_ratios.head(long_run_days).mean(skipna=True)  # long run data average
    LT_ave_spreads = all_spreads.head(long_run_days).mean(skipna=True)  # long run data average

ST_ave_ratios = all_ratios.head(number_days).mean(skipna=True)  # average for number of days selected
ST_ave_spreads = all_spreads.head(number_days).mean(skipna=True)  # average for number of days selected

minimum_ratios = all_ratios.min()  # LT minimum ratio
maximum_ratios = all_ratios.max()  # LT maximum ratio

minimum_spreads = all_spreads.min()  # LT minimum spread
maximum_spreads = all_spreads.max()  # LT maximum spread

# quantiles
u_quantile_ratio = all_ratios.quantile(0.95)
l_quantile_ratio = all_ratios.quantile(0.05)

u_quantile_spread = all_spreads.quantile(0.95)
l_quantile_spread = all_spreads.quantile(0.05)


# CREATE HISTOGRAM FOR DISTRIBUTION ANALYSIS
def create_histogram(input, reference, target, method):
    n = (reference * cols) + target
    df = input.iloc[:, n]
    current_val = df.iloc[0]
    total_values = df.dropna()
    try:
        # Calculate the KDE
        density = gaussian_kde(df.dropna())
        # Use KDE for calculating probability estimate
        upper_bound = total_values.max()
        prob, _ = integrate.quad(density, current_val, upper_bound)
        prob = round(prob, 2)

        # Calculate the moments of the data
        data_mean = df.mean(skipna=True)
        data_std = df.std(skipna=True)
        std_from_mean = round((current_val - data_mean) / data_std, 2)
        # Calculate the proportion of values within 1, 2, and 3 standard deviations
        within_1_std = df[(df > data_mean - data_std) & (df < data_mean + data_std)].count() / len(total_values)
        within_2_std = df[(df > data_mean - 2 * data_std) & (df < data_mean + 2 * data_std)].count() / len(total_values)
        within_3_std = df[(df > data_mean - 3 * data_std) & (df < data_mean + 3 * data_std)].count() / len(total_values)
        within_1_std = round(within_1_std, 2)
        within_2_std = round(within_2_std, 2)
        within_3_std = round(within_3_std, 2)

        xs = np.linspace(min(df), max(df), 100)
        ys = density(xs)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df,
                                   nbinsx=75,
                                   name='Ratio Distribution',
                                   histnorm='probability density',
                                   marker=dict(color='green', opacity=0.5)))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='KDE'))
        fig.update_layout(barmode='overlay')
        fig['data'][0]['marker']['color'] = 'green'  # Histogram
        fig['data'][1]['line']['color'] = 'red'  # Density plot

        # Add a vertical line at the current value
        fig.add_shape(type="line",
                      x0=current_val,
                      x1=current_val, y0=0, y1=1, yref="paper",
                      line=dict(color="Blue", width=2))
        if method == "ratio":
            fig.update_layout(title="Ratio Distribution Analysis")
        elif method == "spread":
            fig.update_layout(title="Spread Distribution Analysis")

        probability_text = f"""
            ±1σ: {within_1_std * 100}% of values
            ±2σ: {within_2_std * 100}% of values
            ±3σ: {within_3_std * 100}% of values
            Current value: {round(current_val, 2)}
            The current value is {std_from_mean} σ from the mean
            {prob * 100}% values lie above the current value, {round((1 - prob) * 100, 2)}% lie below
            """
    except numpy.linalg.LinAlgError:
        probability_text = f"""
            Cannot analyze distribution
            """
        fig = go.Figure()
    return fig, probability_text

# PRODUCT DATA ANALYSIS
def item_analysis(ref_idx,tperiod, time, nyears):
    time_period = tperiod #INPUT
    n = nyears #INPUT NUMBER OF SEASONAL YEARS
    reference_item_idx = ref_idx #INPUT
    reference_item = raw_data.iloc[:,reference_item_idx+1].dropna() #flip data so reads left to right

    if re.search(r'\$/gross ton', reference_item.name):
        reference_item =round(reference_item / 0.984207,2)
    elif re.search(r'\$/short ton|\$/cwt', reference_item.name):
        reference_item = round(reference_item / 1.10231,2)

    reference_item_length = len(reference_item)
    start_date = pd.to_datetime(raw_data.iloc[:,0].iloc[0])
    reference_item_dates = raw_data.iloc[:,0].iloc[:reference_item_length] #flip data so reads left to right
    reference_item_dates = pd.to_datetime(reference_item_dates)
    results = []
    select_periods = [1,3,6,12,36,60] #month to 5 years
    def find_nearest_date(target_date, date_series):
        nearest_date = date_series.iloc[(date_series - target_date).abs().argsort()[:1]]
        return nearest_date.values[0]
    for period in select_periods:
        target_date = start_date - relativedelta(months=period)
        if target_date in reference_item_dates.values:
            results.append(target_date)
        else:
            nearest_date = find_nearest_date(target_date, reference_item_dates)
            results.append(nearest_date)
    num_elements_list = []
    for end_date in results:
        # Filter the DataFrame to include only the dates between the start and end dates
        filtered_df = reference_item_dates[(reference_item_dates <= start_date) & (reference_item_dates >= end_date)]
        num_elements_list.append(len(filtered_df))
    num_elements_list.append(len(reference_item)) #add the max length of data
    t_max = num_elements_list[int(time_period)]
    t = time #INPUT FOR t in LEN(num_element_list[time_period[)
    def subdivide_data_by_year(date_data, price_data):
        df = pd.DataFrame({
            'date': pd.to_datetime(date_data),
            'price': price_data
        })
        # Extract the year from each date and store it in a new column 'year'
        df['year'] = df['date'].dt.year
        # Group the DataFrame by 'year' and store each group in a dictionary
        groups = {year: group for year, group in df.groupby('year')}
        return groups

    groups = subdivide_data_by_year(reference_item_dates[::-1], reference_item[::-1])
    def rebase_price_data(groups):
        # Initialize an empty dictionary to store the rebased data
        rebased_data = {}
        # Iterate over each year group
        for year, group in groups.items():
            # Calculate the base price (the price of the earliest date)
            base_price = group['price'].iloc[0]
            # Rebase the price data by dividing each price by the base price and subtracting 1
            rebased_prices = group['price'] / base_price - 1
            rebased_df = pd.DataFrame(rebased_prices)
            rebased_df.index = group['date']
            rebased_data[year] = rebased_df
        return rebased_data

    rebased_data = rebase_price_data(groups)
    def average_rebased_data(rebased_data):
        monthly_data = pd.DataFrame()
        for year, df in rebased_data.items():
            # Group the data by month and calculate the mean
            monthly_df = df.groupby(df.index.month).mean()
            monthly_data = pd.concat([monthly_data, monthly_df], axis=0)
        # Calculate the average across each column
        average_data = monthly_data.groupby(monthly_data.index).mean()
        # Create a new DataFrame with months as columns and years as rows
        average_data_by_month = average_data.unstack()
        # Rename the columns to month names
        average_data_by_month.columns = ['Jan', 'Feb', 'Mar',
                                         'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep',
                                         'Oct', 'Nov', 'Dec']
        return average_data_by_month

    rebased_average = average_rebased_data(rebased_data)
    def calculate_rate_of_change(price, date, smoothing):
        # Create a DataFrame from the price and date series
        df = pd.DataFrame({
            'date': pd.to_datetime(date),
            'price': price
        })
        # Set the date as the index
        df.set_index('date', inplace=True)
        df['rate of change'] = df['price'].diff() # Calculate the rate of change of prices
        df['acceleration'] = df['rate of change'].diff() # Calculate the rate of change of the rate of change
        df['rate of change smooth'] = df['rate of change'].rolling(smoothing).mean()
        df['acceleration smooth'] = df['acceleration'].rolling(smoothing).mean()
        return df[['rate of change smooth', 'acceleration smooth']]

    rate_change = calculate_rate_of_change(reference_item[::-1],
                                           reference_item_dates[::-1],
                                           14)
    rate_change = rate_change[::-1]
    figure_rate_of_change = rate_change.iloc[:num_elements_list[time_period]]

    current_value_roc = rate_change.iloc[:,0].iloc[t]
    current_value_accel = rate_change.iloc[:,1].iloc[t]

    #FIGURES
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"rowspan": 2}, {}], [None, {}]],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1)
    if isinstance(nyears, int):
        Latest_n = list(rebased_data.keys())[-nyears:]
        for year in Latest_n:
            fig.add_trace(
                go.Scatter(x=rebased_data[year].index.month,
                       y=rebased_data[year]['price'],
                       mode='lines',
                       name=str(year)),
                       row=2, col=2)
            fig.update_xaxes(tickvals=list(range(1, 13)),
                         ticktext=['Jan', 'Feb', 'Mar',
                                   'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep',
                                   'Oct', 'Nov', 'Dec'],
                        row=2, col=2)
    elif nyears == 'average':
        fig.add_trace(
                go.Scatter(x=list(range(1, 13)),
                       y=rebased_average,
                       mode='lines',
                       name='LT Average'),
                       row=2, col=2)
        fig.update_xaxes(tickvals=list(range(1, 13)),
                         ticktext=['Jan', 'Feb', 'Mar',
                                   'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep',
                                   'Oct', 'Nov', 'Dec'],
                         row=2, col=2)

    figure_ref_data = reference_item.iloc[:num_elements_list[time_period]].iloc[::-1]
    figure_dates = reference_item_dates.iloc[:num_elements_list[time_period]].iloc[::-1]
    current_y = figure_ref_data[t]
    current_x = figure_dates[t]
    fig.add_trace(go.Scatter(x=figure_dates,
                             y=figure_ref_data,
                             mode='lines',
                             name=col_names[reference_item_idx]),
                             row=1, col=1
                            )
    fig.add_trace(
        go.Scatter(
            x=[current_x],
            y=[current_y],
            mode='markers',
            name = 'Highlighted Date',
            marker=dict(
                size=6,
                color='black',
                symbol='circle')),  # or 'circle' for a dot
            row=1, col=1
            )

    fig.add_trace(
        go.Violin(y=rate_change.iloc[:,0].dropna(),
                  line_color='black',
                  fillcolor='green',
                  opacity=0.5,
                  legendgroup='Rate of Change',
                  scalegroup='Rate of Change',
                  name='Rate of Change',
                  side='positive',
                  meanline_visible=True),
                  row=1, col=2
    )
    fig.add_trace(
        go.Violin(y=rate_change.iloc[:,1].dropna(),
                  line_color='black',
                  fillcolor='lightblue',
                  opacity=0.6,
                  legendgroup='Acceleration',
                  scalegroup='Acceleration',
                  name='Acceleration',
                  side='negative',
                  meanline_visible=True),
                  row=1,col=2
    )
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.add_shape(
        type="line",
        x0=0,
        y0=current_value_roc,
        x1=0.35,
        y1=current_value_roc,
        line=dict(color="Red", width=2),
        xref='x2', yref='y2'
    )
    fig.add_shape(
        type="line",
        x0=1,
        y0=current_value_accel,
        x1=0.65,
        y1=current_value_accel,
        line=dict(color="Blue", width=2),
        xref='x2', yref='y2'
    )

    fig.update_xaxes(title = col_names[reference_item_idx], row=1, col=1)
    fig.update_xaxes(tickvals=[], row=1, col=2)
    fig.update_xaxes(title = 'Momentum Analysis', row=1, col=2)
    fig.update_xaxes(title = col_names[reference_item_idx] + ' Seasonality', row=2, col=2)
    return fig, t_max

# truncate by number of days
all_ratios_pretrunc = all_ratios.copy()
all_spreads_pretrunc = all_spreads.copy()

all_ratios = all_ratios.iloc[:number_days]
all_spreads = all_spreads.iloc[:number_days]

col_data_ratio = []
col_data_spread = []

for col in all_ratios.columns:
    col_data_ratio = all_ratios.loc[:col]

for col in all_spreads.columns:
    col_data_spread = all_spreads.loc[:col]

mean_abs_ratio = col_data_ratio.mean()
mean_abs_ratio = pd.DataFrame(mean_abs_ratio).T
mean_abs_spread = col_data_spread.mean()
mean_abs_spread = pd.DataFrame(mean_abs_spread).T

col_data_ratio = pd.DataFrame(col_data_ratio)
col_data_spread = pd.DataFrame(col_data_spread)

# CREATE CORRELATION MATRIX
# pad the data by the number of rows in window size
data_corr = data_model.copy()
data_corr = data_corr.drop('Date', axis=1)
data_corr = data_corr.iloc[:(number_days + corr_matrix_days)]

# reverse the rows so padded rows are at the top
data_corr = data_corr[::-1]
correlation = data_corr.rolling(corr_matrix_days).corr()
# slice the first rows of the window size, will be NaN
correlation = correlation[(corr_matrix_days * len(correlation.columns)):]
# reverse the correlation back to original order
correlation = correlation[::-1]
# Convert the DataFrame to a list of lists
data_HM = correlation.values.tolist()

# CALCULATE MEAN REVERSION TIMES
# calculate the difference from the current value in a row and the long term average
diffs_ratios = all_ratios - LT_ave_ratios.values
diffs_spreads = all_spreads - LT_ave_spreads.values

# Create a dataframe to store the average revert time for each item
average_revert_time_ratios = pd.DataFrame(index=['Average Revert Time'], columns=all_ratios.columns)
average_revert_time_spreads = pd.DataFrame(index=['Average Revert Time'], columns=all_spreads.columns)

# Calculate the average revert time for each item
for i in range(len(diffs_ratios.columns)):
    # Find the indices where the difference changes sign
    sign_change_indices_ratios = ((diffs_ratios.iloc[:, i].shift() * diffs_ratios.iloc[:, i]) < 0).to_numpy().nonzero()[
        0]
    sign_change_indices_spreads = \
        ((diffs_spreads.iloc[:, i].shift() * diffs_spreads.iloc[:, i]) < 0).to_numpy().nonzero()[0]

    if len(sign_change_indices_ratios) > 1:
        # Calculate the differences between consecutive indices
        revert_times_ratios = np.diff(sign_change_indices_ratios)

        # The average revert time is the mean of these
        average_revert_time_ratios.iloc[0, i] = revert_times_ratios.mean()
    else:
        # If the sign changes less than twice, set the average revert time to NaN
        average_revert_time_ratios.iloc[0, i] = pd.NA

    if len(sign_change_indices_spreads) > 1:
        # Calculate the differences between consecutive indices
        revert_times_spreads = np.diff(sign_change_indices_spreads)
        average_revert_time_spreads.iloc[0, i] = revert_times_spreads.mean()
    else:
        average_revert_time_spreads.iloc[0, i] = pd.NA

# CALCULATE NORMALIZED RATIO AND SPREAD SIGNALS
# standardize the columns in ratios and spreads
std_ratio = []
std_spread = []
for col in all_ratios.columns:
    std_ratio = stats.zscore(all_ratios.loc[:col])
for col in all_spreads.columns:
    std_spread = stats.zscore(all_spreads.loc[:col])

# find minimum and maximum
minimum_ratio_signal = -std_ratio.min()
maximum_ratio_signal = -std_ratio.max()

minimum_spread_signal = -std_spread.min()
maximum_spread_signal = -std_spread.max()

std_ave_ratio = round(std_ratio.mean(), 2)
std_ave_spread = round(std_spread.mean(), 2)

std_ratio = pd.DataFrame(std_ratio)
std_ave_ratio = pd.DataFrame(std_ave_ratio).T

std_spread = pd.DataFrame(std_spread)
std_ave_spread = pd.DataFrame(std_ave_spread).T

# signals are differences from 0
signals_ratio = []
signals_spread = []
for i in range(len(std_ratio)):
    signal_ratio = std_ave_ratio - std_ratio.iloc[i]  # this inverts the signal < 0 overvalued; > 0 undervalued
    signal_spread = std_ave_spread - std_spread.iloc[i]  # this inverts the signal < 0 overvalued; > 0 undervalued

    signals_ratio.append(signal_ratio)
    signals_spread.append(signal_spread)

# generate ratio and spread absolute graphic data
abs_ratio_signals = []
abs_spread_signals = []
for i in range(len(col_data_ratio)):
    abs_ratio_signal = col_data_ratio.iloc[i] - mean_abs_ratio + mean_abs_ratio
    abs_ratio_signals.append(abs_ratio_signal)

    abs_spread_signal = col_data_spread.iloc[i] - mean_abs_spread + mean_abs_spread
    abs_spread_signals.append(abs_spread_signal)

# generate weighted averages
col = all_ratios.columns.tolist()
current = data_model.copy().drop('Date', axis=1)
current = current.head(number_days)

# create the predictions from the long run averages
predictions_ratios = []
predictions_spread = []
for i in range(len(dates)):
    current_price = current.iloc[:1]
    predictions_i_ratio = []
    predictions_i_spread = []
    predictions_ratios.append(predictions_i_ratio)
    predictions_spread.append(predictions_i_spread)
    for j in range(0, len(LT_ave_ratios), cols):
        ave_ratio = LT_ave_ratios.iloc[j:j + cols]
        ave_spread = LT_ave_spreads.iloc[j:j + cols]

        pred_ratio = current_price * ave_ratio
        pred_spread = current_price + ave_spread

        predictions_i_ratio.append(pred_ratio)
        predictions_i_spread.append(pred_spread)

weightings = []
# weighting method absolute, square or logarithmic derived from correlation matrix
for k in range(0, len(correlation), cols):
    wgt = correlation[k:k + cols]
    if weighting_type == 'absolute':
        wgt = wgt.abs()
    elif weighting_type == 'square':
        wgt = wgt.pow(2)
    elif weighting_type == 'log':
        wgt = np.log(wgt + 1.0000001)

    for n in range(cols):
        wgt.iloc[n] = wgt.iloc[n] / wgt.iloc[n].sum()
    weightings.append(wgt)

# create weighted averages
weighted_average_ratio = []
weighted_average_spread = []
for i in range(len(dates)):
    wgt_pred_ratio = []
    wgt_pred_spread = []
    for j in range(cols):
        wgt_item_ratio = (predictions_ratios[i][j] * weightings[i].iloc[cols - 1 - j]).sum(axis=1)
        wgt_item_spread = (predictions_spread[i][j] * weightings[i].iloc[cols - 1 - j]).sum(axis=1)

        wgt_pred_ratio.append(wgt_item_ratio)
        wgt_pred_spread.append(wgt_item_spread)

    weighted_average_ratio.append(wgt_pred_ratio)
    weighted_average_spread.append(wgt_pred_spread)

# clean the result and convert to a DataFrame
weighted_average_clean_ratio = []
weighted_average_clean_spread = []

for i in range(len(weighted_average_ratio)):
    clean_i_ratio = []
    clean_i_spread = []
    for j in range(len(weighted_average_ratio[i])):
        clean_i_ratio.append(weighted_average_ratio[i][j][0])
        clean_i_spread.append(weighted_average_spread[i][j][0])

    weighted_average_clean_ratio.append(clean_i_ratio)
    weighted_average_clean_spread.append(clean_i_spread)

weighted_average_ratio = pd.DataFrame(weighted_average_ratio, columns=col[0:cols])
weighted_average_clean_ratio = pd.DataFrame(weighted_average_clean_ratio, columns=col[0:cols])
weighted_average_spread = pd.DataFrame(weighted_average_spread, columns=col[0:cols])
weighted_average_clean_spread = pd.DataFrame(weighted_average_clean_spread, columns=col[0:cols])

# unit conversion
# Filter columns that contain '$/gross ton' in their name and divide by 0.984207
column_conversion = weighted_average_clean_ratio.filter(regex='\$/gross ton').columns
weighted_average_clean_ratio[column_conversion] = weighted_average_clean_ratio[column_conversion] / 0.984207
column_conversion = weighted_average_clean_spread.filter(regex='\$/gross ton').columns
weighted_average_clean_spread[column_conversion] = weighted_average_clean_spread[column_conversion] / 0.984207
column_conversion = current.filter(regex='\$/gross ton').columns
current[column_conversion] = current[column_conversion] / 0.984207

# Filter columns that contain '$/short ton' or '$/cwt' in their name and divide by 1.10231
column_conversion = weighted_average_clean_ratio.filter(regex='\$/short ton|\$/cwt').columns
weighted_average_clean_ratio[column_conversion] = weighted_average_clean_ratio[column_conversion] / 1.10231
column_conversion = weighted_average_clean_spread.filter(regex='\$/short ton|\$/cwt').columns
weighted_average_clean_spread[column_conversion] = weighted_average_clean_spread[column_conversion] / 1.10231
column_conversion = current.filter(regex='\$/short ton|\$/cwt').columns
current[column_conversion] = current[column_conversion] / 1.10231

average_pred_price = (
                             weighted_average_clean_ratio + weighted_average_clean_spread) / 2  # predicted price average from ratio and spread
minimum_pred_price = weighted_average_clean_ratio.where(weighted_average_clean_ratio < weighted_average_clean_spread,
                                                        weighted_average_clean_spread)
maximum_pred_price = weighted_average_clean_ratio.where(weighted_average_clean_ratio > weighted_average_clean_spread,
                                                        weighted_average_clean_spread)

# GENERATE FIGURES
# Create a correlation matrix
fig_correlation_matrix = go.Figure(data=go.Heatmap(
    z=data_HM,
    x=correlation.columns,
    y=correlation.columns[::-1],
    hoverongaps=False,
    colorbar=dict(title='Correlation', titleside='right'),
    hovertemplate='<i>Correlation</i>: %{z} <i>Reference</i>: %{x}<extra></extra>',
    # Display the correlation value on hover
    xgap=1,  # Gap between squares in x direction
    ygap=1,  # Gap between squares in y direction
))

# Add slider
steps = []
for i in range(0, len(correlation), len(correlation.columns)):
    step = dict(
        method="restyle",
        args=["z", [data_HM[i:i + len(correlation.columns)]]],
        label='<b>' + str(dates[int(i / len(correlation.columns))]) + '</b>'
    )
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Date: "},
    pad={"t": 50},
    steps=steps,
    x=0,  # This will move the slider to the left (-), right (+)
    y=1.15,  # This will move the slider to the bottom (-), top (+)
    len=1  # This will make the slider shorter
)]

scale = 0.95
fig_correlation_matrix.update_layout(
    title="Correlation Matrix",
    sliders=sliders,
    plot_bgcolor='black',  # Make the background color black
    autosize=True,
    width=2048 * scale,
    height=1536 * scale,
    scene=dict(aspectmode='cube'),
    xaxis_showgrid=False,  # This will remove the grid for the x-axis
    yaxis_showgrid=False,  # This will remove the grid for the y-axis
)

# plot for standardized ratio and spread signals
# standardized ratio
traces_ratio = []
for i in range(len(dates)):
    for j in range(0, len(signals_ratio[i].T), cols):
        # Define colors based on z values
        colors = []
        for value in signals_ratio[i].values.tolist()[0][j:j + cols]:
            if value < -1.25:
                colors.append('darkred')
            elif -1.25 < value < -1:
                colors.append('red')
            elif -1.00 < value < -0.50:
                colors.append('tomato')
            elif -0.50 < value < 0.50:
                colors.append('orange')
            elif 0.50 < value < 1.00:
                colors.append('chartreuse')
            elif 1 < value < 1.25:
                colors.append('green')
            elif value > 1.25:
                colors.append('darkgreen')
            elif value == 'nan' or np.nan or pd.NA:
                colors.append('black')

        # Create a trace with signal data points
        trace = go.Scatter(
            x=signals_ratio[i].columns.tolist()[j:j + cols],
            y=signals_ratio[i].values.tolist()[0][j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.6
            ),
            visible=(i == 0),
            name='signal'
        )
        traces_ratio.append(trace)
        # Create a trace with signal maximum points
        trace_max = go.Scatter(
            x=signals_ratio[i].columns.tolist()[j:j + cols],
            y=maximum_ratio_signal[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-up',
                color='black',
                opacity=0.6
            ),
            visible=(i == 0),
            name='maximum'
        )
        traces_ratio.append(trace_max)
        # Create a trace with signal minimum points
        trace_min = go.Scatter(
            x=signals_ratio[i].columns.tolist()[j:j + cols],
            y=minimum_ratio_signal[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-down',
                color='black',
                opacity=0.6
            ),
            visible=(i == 0),
            name='minimum'
        )
        traces_ratio.append(trace_min)
# create the standardized ratio signal figure
fig_std_ratio = go.Figure(data=traces_ratio)

# Add a horizontal line at y=0
fig_std_ratio.add_shape(
    type="line",
    x0=0,
    x1=cols,
    y0=0,
    y1=0,
    line=dict(
        color="blue",
        width=1,
    ),
)
fig_std_ratio.update_layout(title="Ratio Signal Analysis")

# standardized spread
traces_spread = []
for i in range(len(dates)):
    for j in range(0, len(signals_spread[i].T), cols):
        # Define colors based on z values
        colors = []
        for value in signals_spread[i].values.tolist()[0][j:j + cols]:
            if value < -1.25:
                colors.append('darkred')
            elif -1.25 < value < -1:
                colors.append('red')
            elif -1.00 < value < -0.50:
                colors.append('tomato')
            elif -0.50 < value < 0.50:
                colors.append('orange')
            elif 0.50 < value < 1.00:
                colors.append('chartreuse')
            elif 1 < value < 1.25:
                colors.append('green')
            elif value > 1.25:
                colors.append('darkgreen')
            elif value == 'nan' or np.nan or pd.NA:
                colors.append('black')

        # Create a trace with signal data points
        trace = go.Scatter(
            x=signals_spread[i].columns.tolist()[j:j + cols],
            y=signals_spread[i].values.tolist()[0][j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.6
            ),
            visible=(i == 0),
            name='signal'
        )
        traces_spread.append(trace)
        # Create a trace with signal maximum points
        trace_max = go.Scatter(
            x=signals_spread[i].columns.tolist()[j:j + cols],
            y=maximum_spread_signal[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-up',
                color='black',
                opacity=0.6
            ),
            visible=(i == 0),
            name='maximum'
        )
        traces_spread.append(trace_max)
        # Create a trace with signal minimum points
        trace_min = go.Scatter(
            x=signals_spread[i].columns.tolist()[j:j + cols],
            y=minimum_spread_signal[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-down',
                color='black',
                opacity=0.6
            ),
            visible=(i == 0),
            name='minimum'
        )
        traces_spread.append(trace_min)

# create the standardized spread signal figure
fig_std_spread = go.Figure(data=traces_spread)

# Add a horizontal line at y=0
fig_std_spread.add_shape(
    type="line",
    x0=0,
    x1=cols,
    y0=0,
    y1=0,
    line=dict(
        color="blue",
        width=1,
    ),
)
fig_std_spread.update_layout(title="Spread Signal Analysis")

# absolute ratio graphic
traces_abs_ratio = []
for i in range(len(dates)):
    j_length = (len(np.array(abs_ratio_signals[i]).transpose()))
    for j in range(0, j_length, cols):
        colors = []
        for value, max_value, min_value in zip(
                abs_ratio_signals[i].values.tolist()[0][j:j + cols],
                maximum_ratios.values[j:j + cols],
                minimum_ratios.values[j:j + cols]
        ):
            # Calculate the midpoint
            midpoint = (max_value + min_value) / 2
            # Calculate the range (denominator)
            range_value = max_value - min_value
            # Check if the range is not zero
            if range_value != 0:
                # Normalize the value to the interval [-1, 1]
                normalized_value = (value - midpoint) / range_value
                # Determine the color based on the normalized value
                if -1.00 <= normalized_value < -0.50:
                    colors.append('dark green')
                elif -0.50 < normalized_value < -0.25:
                    colors.append('green')
                elif 0. - 25 < normalized_value < 0.25:
                    colors.append('orange')
                elif 0.25 < normalized_value < 0.50:
                    colors.append('red')
                elif 0.50 < normalized_value <= 1:
                    colors.append('dark red')
                else:
                    colors.append('gray')

        # Create a trace with signal data points
        trace = go.Scatter(
            x=abs_ratio_signals[i].columns.tolist()[j:j + cols],
            y=abs_ratio_signals[i].values.tolist()[0][j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.75
            ),
            visible=(i == 0),
            name='current'
        )
        traces_abs_ratio.append(trace)

        trace_max = go.Scatter(
            x=abs_ratio_signals[i].columns.tolist()[j:j + cols],
            y=maximum_ratios[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-down',
                color='black',
                opacity=0.40
            ),
            visible=(i == 0),
            name='ST maximum'
        )
        traces_abs_ratio.append(trace_max)

        trace_min = go.Scatter(
            x=abs_ratio_signals[i].columns.tolist()[j:j + cols],
            y=minimum_ratios[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-up',
                color='black',
                opacity=0.40
            ),
            visible=(i == 0),
            name='ST minimum'
        )
        traces_abs_ratio.append(trace_min)

        trace_ave = go.Scatter(
            x=abs_ratio_signals[i].columns.tolist()[j:j + cols],
            y=LT_ave_ratios[j:j + cols],
            mode='markers',
            marker=dict(
                size=7,
                symbol='cross',
                color='blue',
                opacity=0.40
            ),
            visible=(i == 0),
            name='LR average'
        )
        traces_abs_ratio.append(trace_ave)

        # Add a bar chart trace for the lower quartile
        trace_l_quartile = go.Bar(
            x=abs_ratio_signals[i].columns.tolist()[j:j + cols],
            y=l_quantile_ratio[j:j + cols],
            marker=dict(
                color='rgba(255, 255, 255, 0)',  # Set color to transparent
            ),
            visible=(i == 0),
            showlegend=False,  # Do not show this trace in the legend
        )
        traces_abs_ratio.append(trace_l_quartile)

        # Add a bar chart trace for the difference between upper and lower quartiles
        trace_quantile_diff = go.Bar(
            x=abs_ratio_signals[i].columns.tolist()[j:j + cols],
            y=[u - l for u, l in zip(u_quantile_ratio[j:j + cols], l_quantile_ratio[j:j + cols])],
            marker=dict(
                color='purple',
                opacity=0.10
            ),
            visible=(i == 0),
            name='interquartile range'
        )
        traces_abs_ratio.append(trace_quantile_diff)

# Update layout to stack the bar
fig_abs_ratio = go.Figure(data=traces_abs_ratio)
fig_abs_ratio.update_layout(barmode='stack', title="Ratio Analysis IQR: 90%")

# absolute spread graphic
traces_abs_spread = []
for i in range(len(dates)):
    j_length = (len(np.array(abs_spread_signals[i]).transpose()))
    for j in range(0, j_length, cols):
        colors = []
        for value, max_value, min_value in zip(
                abs_spread_signals[i].values.tolist()[0][j:j + cols],
                maximum_spreads.values[j:j + cols],
                minimum_spreads.values[j:j + cols]
        ):
            # Calculate the midpoint
            midpoint = (max_value + min_value) / 2
            # Calculate the range (denominator)
            range_value = max_value - min_value
            # Check if the range is not zero
            if range_value != 0:
                # Normalize the value to the interval [-1, 1]
                normalized_value = (value - midpoint) / range_value
                # Determine the color based on the normalized value
                if -1.00 < normalized_value < -0.50:
                    colors.append('dark green')
                elif -0.50 < normalized_value < -0.25:
                    colors.append('green')
                elif 0. - 25 < normalized_value < 0.25:
                    colors.append('orange')
                elif 0.25 < normalized_value < 0.50:
                    colors.append('red')
                elif 0.50 < normalized_value < 1:
                    colors.append('dark red')
                else:
                    colors.append('gray')

        # Create a trace with signal data points
        trace = go.Scatter(
            x=abs_spread_signals[i].columns.tolist()[j:j + cols],
            y=abs_spread_signals[i].values.tolist()[0][j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.75
            ),
            visible=(i == 0),
            name='current'
        )
        traces_abs_spread.append(trace)

        trace_max = go.Scatter(
            x=abs_spread_signals[i].columns.tolist()[j:j + cols],
            y=maximum_spreads[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-down',
                color='black',
                opacity=0.40
            ),
            visible=(i == 0),
            name='ST maximum'
        )
        traces_abs_spread.append(trace_max)

        trace_min = go.Scatter(
            x=abs_spread_signals[i].columns.tolist()[j:j + cols],
            y=minimum_spreads[j:j + cols],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-up',
                color='black',
                opacity=0.40
            ),
            visible=(i == 0),
            name='ST minimum'
        )
        traces_abs_spread.append(trace_min)

        trace_ave = go.Scatter(
            x=abs_spread_signals[i].columns.tolist()[j:j + cols],
            y=LT_ave_spreads[j:j + cols],
            mode='markers',
            marker=dict(
                size=7,
                symbol='cross',
                color='blue',
                opacity=0.40
            ),
            visible=(i == 0),
            name='LR average'
        )
        traces_abs_spread.append(trace_ave)

        # Add a bar chart trace for the lower quartile
        trace_l_quartile = go.Bar(
            x=abs_spread_signals[i].columns.tolist()[j:j + cols],
            y=l_quantile_spread[j:j + cols],
            marker=dict(
                color='rgba(255, 255, 255, 0)',  # Set color to transparent
            ),
            visible=(i == 0),
            showlegend=False,  # Do not show this trace in the legend
        )
        traces_abs_spread.append(trace_l_quartile)

        # Add a bar chart trace for the difference between upper and lower quartiles
        trace_quantile_diff = go.Bar(
            x=abs_spread_signals[i].columns.tolist()[j:j + cols],
            y=[u - l for u, l in zip(u_quantile_spread[j:j + cols], l_quantile_spread[j:j + cols])],
            marker=dict(
                color='purple',
                opacity=0.10
            ),
            visible=(i == 0),
            name='interquartile range'
        )
        traces_abs_spread.append(trace_quantile_diff)

# Update layout to stack the bar
fig_abs_spread = go.Figure(data=traces_abs_spread)
fig_abs_spread.update_layout(barmode='stack', title="Spread Analysis IQR: 90%")

# predicted price graphic
traces_pred_price = []
for i in range(len(dates)):
    # Create a trace with signal data points
    trace = go.Scatter(
        x=average_pred_price.columns.tolist(),
        y=average_pred_price.iloc[i][0:cols],
        mode='markers',
        marker=dict(
            size=10,
            color='orange',
            opacity=0.75
        ),
        visible=(i == 0),  # make the first trace visible
        name='Average'  # name for the legend
    )
    traces_pred_price.append(trace)

    trace_max = go.Scatter(
        x=average_pred_price.columns.tolist(),
        y=maximum_pred_price.iloc[i][0:cols],
        mode='markers',
        marker=dict(
            size=10,
            symbol='triangle-down',
            color='black',
            opacity=0.40
        ),
        visible=(i == 0),
        name='Maximum'  # name for the legend
    )
    traces_pred_price.append(trace_max)

    trace_min = go.Scatter(
        x=average_pred_price.columns.tolist(),
        y=minimum_pred_price.iloc[i][0:cols],
        mode='markers',
        marker=dict(
            size=10,
            symbol='triangle-up',
            color='black',
            opacity=0.40
        ),
        visible=(i == 0),
        name='Minimum'  # name for the legend
    )
    traces_pred_price.append(trace_min)

    trace_current = go.Scatter(
        x=average_pred_price.columns.tolist(),
        y=current.iloc[i][0:cols],
        mode='markers',
        marker=dict(
            size=10,
            symbol='square',
            color='green',
            opacity=0.40
        ),
        visible=(i == 0),
        name='Current'  # name for the legend
    )
    traces_pred_price.append(trace_current)

fig_pred_price = go.Figure(data=traces_pred_price)
fig_pred_price.update_layout(title="Predicted Prices")

# GENERATE THE DASHBOARD
# create the Dash app layout

app.layout = html.Div([
    dcc.Graph(figure=fig_correlation_matrix),
    dcc.Graph(id='ratio_std_2d-scatter-plot', figure=fig_std_ratio, style={'height': '100vh'}),
    # Increase the size of the plot area
    html.Div([
        dcc.Slider(
            id='date_slider_ratio_std',
            min=0,
            max=len(dates) - 1,
            step=1,
            value=0,
            marks={i: dates[i] for i in range(0, len(dates), 10)},  # Display labels only for every 10th date
        ),
        html.Div(id='date_slider_ratio_std-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders
    html.Div([
        dcc.Slider(
            id='item_slider_ratio_std',
            min=0,
            max=cols - 1,
            step=1,
            value=0,
            marks={i: signals_ratio[i].columns.tolist()[i] for i in range(0, cols, 10)},
            # Display labels only for every 10th item
        ),
        html.Div(id='item_slider_ratio_std-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders
    dcc.Graph(id='spread_std_2d-scatter-plot', figure=fig_std_spread, style={'height': '100vh'}),
    # Increase the size of the plot area
    html.Div([
        dcc.Slider(
            id='date_slider_spread_std',
            min=0,
            max=len(dates) - 1,
            step=1,
            value=0,
            marks={i: dates[i] for i in range(0, len(dates), 10)},  # Display labels only for every 10th date
        ),
        html.Div(id='date_slider_spread_std-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders
    html.Div([
        dcc.Slider(
            id='item_slider_spread_std',
            min=0,
            max=cols - 1,
            step=1,
            value=0,
            marks={i: signals_spread[i].columns.tolist()[i] for i in range(0, cols, 10)},
            # Display labels only for every 10th item
        ),
        html.Div(id='item_slider_spread_std-output')  # Display the current step
    ], style={'margin': '30px 0px'}),
    dcc.Graph(id='abs_ratio_2d-scatter-plot', figure=fig_abs_ratio, style={'height': '100vh'}),
    # Increase the size of the plot area
    html.Div([
        dcc.Slider(
            id='abs_ratio_date_slider',
            min=0,
            max=len(dates) - 1,
            step=1,
            value=0,
            marks={i: dates[i] for i in range(0, len(dates), 10)},  # Display labels only for every 10th date
        ),
        html.Div(id='abs_ratio_date_slider-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders
    html.Div([
        dcc.Slider(
            id='abs_ratio_item_slider',
            min=0,
            max=cols - 1,
            step=1,
            value=0,
            marks={i: abs_ratio_signals[i].columns.tolist()[i] for i in range(0, cols, 10)},
            # Display labels only for every 10th item
        ),
        html.Div(id='abs_ratio_item_slider-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders,
    dcc.Graph(id='abs_spread_2d-scatter-plot', figure=fig_abs_ratio, style={'height': '100vh'}),
    # Increase the size of the plot area
    html.Div([
        dcc.Slider(
            id='abs_spread_date_slider',
            min=0,
            max=len(dates) - 1,
            step=1,
            value=0,
            marks={i: dates[i] for i in range(0, len(dates), 10)},  # Display labels only for every 10th date
        ),
        html.Div(id='abs_spread_date_slider-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders
    html.Div([
        dcc.Slider(
            id='abs_spread_item_slider',
            min=0,
            max=cols - 1,
            step=1,
            value=0,
            marks={i: abs_ratio_signals[i].columns.tolist()[i] for i in range(0, cols, 10)},
            # Display labels only for every 10th item
        ),
        html.Div(id='abs_spread_item_slider-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders
    dcc.Graph(id='pred_price_2d-scatter-plot', figure=fig_pred_price, style={'height': '100vh'}),
    # Increase the size of the plot area
    html.Div([
        dcc.Slider(
            id='pred_price_date_slider',
            min=0,
            max=len(dates) - 1,
            step=1,
            value=0,
            marks={i: dates[i] for i in range(0, len(dates), 10)},  # Display labels only for every 10th date
        ),
        html.Div(id='pred_price_date_slider-output')  # Display the current step
    ], style={'margin': '30px 0px'}),  # Increase the spacing between sliders
    html.Div([
        dcc.Graph(id='ratio_histogram-graph'),
        html.Div([
            html.Div(id='reference_slider_ratio-output'),  # Display the current step
            dcc.Slider(
                id='reference_slider_ratio',
                min=0,
                max=cols - 1,
                step=1,
                value=0,
                marks={i: col_names[i] for i in range(0, cols, 10)},
            ),
            html.Div(style={'height': '30px'}),  # Add a spacer Div
            html.Div(id='target_slider_ratio-output'),  # Display the current step
            dcc.Slider(
                id='target_slider_ratio',
                min=0,
                max=cols - 1,
                step=1,
                value=1,
                marks={i: col_names[i] for i in range(0, cols, 10)},
            ),
            html.Div(style={'height': '30px'}),  # Add a spacer Div
            html.Div(id='ratio_probability-text')
        ], style={'margin': '30px 0px'}),
        html.Div([
        dcc.Graph(id='spread_histogram-graph'),
        html.Div([
            html.Div(id='reference_slider_spread-output'),  # Display the current step
            dcc.Slider(
                id='reference_slider_spread',
                min=0,
                max=cols - 1,
                step=1,
                value=0,
                marks={i: col_names[i] for i in range(0, cols, 10)},
            ),
            html.Div(style={'height': '30px'}),  # Add a spacer Div
            html.Div(id='target_slider_spread-output'),  # Display the current step
            dcc.Slider(
                id='target_slider_spread',
                min=0,
                max=cols - 1,
                step=1,
                value=1,
                marks={i: col_names[i] for i in range(0, cols, 10)},
            ),
            html.Div(style={'height': '30px'}),  # Add a spacer Div
            html.Div(id='spread_probability-text')
        ], style={'margin': '30px 0px'})
    ])
    ]),
    dcc.Graph(id='item_graph', style={'height': '70vh'}),
    html.Div([
        html.Div(
        dcc.RadioItems(
                    id='tperiod',
                    options=[{'label': '1 Month', 'value': 0},
                                   {'label': '3 Month', 'value': 1},
                                   {'label': '6 Month', 'value': 2},
                                   {'label': '1 year', 'value': 3},
                                   {'label': '3 Year', 'value': 4},
                                   {'label': '5 Year', 'value': 5},
                                   {'label': 'Max', 'value': 6}],
                    value=2,
                    inline=True),
                    style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div(
        dcc.RadioItems(
                    id='n_years',
                    options=[{'label': '1 Year', 'value': 1},
                                   {'label': '3 Year', 'value': 3},
                                   {'label': '5 Year', 'value': 5},
                                   {'label': 'Average', 'value': 'average'}],
                    value='average',
                    inline=True),
                    style={'width': '50%', 'display': 'inline-block', 'textAlign': 'right'}
                  )
    ]),
        html.Div(style={'height': '30px'}),
        html.Div([
        html.Div(id='ref_idx-output'),
        html.Div(style={'height': '30px'}),
        dcc.Slider(id='ref_idx',
                    min=0,
                    max=cols - 1,
                    step=1,
                    value=0,
                    marks={i: col_names[i] for i in range(0, cols, 10)},
    ),
    html.Div(style={'height': '30px'}),
    html.Div(id='t_max_storage', style={'display': 'none'}),
    html.Div(id='time-output'),
    dcc.Slider(id='time',
               min=0,
               step=1,
               value=0)])
    ])

@app.callback(
    Output('ratio_std_2d-scatter-plot', 'figure'),
    Output('date_slider_ratio_std-output', 'children'),
    Output('item_slider_ratio_std-output', 'children'),
    Output('spread_std_2d-scatter-plot', 'figure'),
    Output('date_slider_spread_std-output', 'children'),
    Output('item_slider_spread_std-output', 'children'),
    Output('abs_ratio_2d-scatter-plot', 'figure'),
    Output('abs_ratio_date_slider-output', 'children'),
    Output('abs_ratio_item_slider-output', 'children'),
    Output('abs_spread_2d-scatter-plot', 'figure'),
    Output('abs_spread_date_slider-output', 'children'),
    Output('abs_spread_item_slider-output', 'children'),
    Output('pred_price_2d-scatter-plot', 'figure'),
    Output('pred_price_date_slider-output', 'children'),
    Output('ratio_histogram-graph', 'figure'),
    Output('ratio_probability-text', 'children'),
    Output('reference_slider_ratio-output', 'children'),
    Output('target_slider_ratio-output', 'children'),
    Output('spread_histogram-graph', 'figure'),
    Output('spread_probability-text', 'children'),
    Output('reference_slider_spread-output', 'children'),
    Output('target_slider_spread-output', 'children'),
    Output('item_graph', 'figure'),
    Output('t_max_storage', 'children'),
    Output('ref_idx-output', 'children'),
    Output('time-output', 'children'),
    Input('date_slider_ratio_std', 'value'),
    Input('item_slider_ratio_std', 'value'),
    Input('date_slider_spread_std', 'value'),
    Input('item_slider_spread_std', 'value'),
    Input('abs_ratio_date_slider', 'value'),
    Input('abs_ratio_item_slider', 'value'),
    Input('abs_spread_date_slider', 'value'),
    Input('abs_spread_item_slider', 'value'),
    Input('pred_price_date_slider', 'value'),
    Input('reference_slider_ratio', 'value'),
    Input('target_slider_ratio', 'value'),
    Input('reference_slider_spread', 'value'),
    Input('target_slider_spread', 'value'),
    Input('ref_idx', 'value'),
    Input('tperiod', 'value'),
    Input('time', 'value'),
    Input('n_years', 'value'),
)

def update_figure_ratio_std(date_slider_ratio_std, item_slider_ratio_std, date_slider_spread_std,
                            item_slider_spread_std, abs_ratio_date_slider, abs_ratio_item_slider,
                            abs_spread_date_slider, abs_spread_item_slider, pred_price_date_slider,
                            reference_slider_ratio, target_slider_ratio,
                            reference_slider_spread, target_slider_spread,
                            ref_idx, tperiod, time, n_years):
    for a in range(len(dates)):
        for g in range(cols):
            fig_std_ratio.data[(g * 3) + (cols * a * 3)].visible = (
                    a == date_slider_ratio_std and g == item_slider_ratio_std)  # signal trace
            fig_std_ratio.data[(g * 3) + 1 + (cols * a * 3)].visible = (
                    a == date_slider_ratio_std and g == item_slider_ratio_std)  # maximum trace
            fig_std_ratio.data[(g * 3) + 2 + (cols * a * 3)].visible = (
                    a == date_slider_ratio_std and g == item_slider_ratio_std)  # minimum trace

    for b in range(len(dates)):
        for h in range(cols):
            fig_std_spread.data[(h * 3) + (cols * b * 3)].visible = (
                    b == date_slider_spread_std and h == item_slider_spread_std)  # signal trace
            fig_std_spread.data[(h * 3) + 1 + (cols * b * 3)].visible = (
                    b == date_slider_spread_std and h == item_slider_spread_std)  # maximum trace
            fig_std_spread.data[(h * 3) + 2 + (cols * b * 3)].visible = (
                    b == date_slider_spread_std and h == item_slider_spread_std)  # minimum trace

    for c in range(len(dates)):
        for i in range(cols):
            fig_abs_ratio.data[(i * 6) + 0 + (cols * c * 6)].visible = (
                    c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # signal trace
            fig_abs_ratio.data[(i * 6) + 1 + (cols * c * 6)].visible = (
                    c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # maximum trace
            fig_abs_ratio.data[(i * 6) + 2 + (cols * c * 6)].visible = (
                    c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # minimum trace
            fig_abs_ratio.data[(i * 6) + 3 + (cols * c * 6)].visible = (
                    c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # average trace
            fig_abs_ratio.data[(i * 6) + 4 + (cols * c * 6)].visible = (
                    c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # quantile trace
            fig_abs_ratio.data[(i * 6) + 5 + (cols * c * 6)].visible = (
                    c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # quantile trace

    for d in range(len(dates)):
        for j in range(cols):
            fig_abs_spread.data[(j * 6) + 0 + (cols * d * 6)].visible = (
                    d == abs_spread_date_slider and j == abs_spread_item_slider)  # signal trace
            fig_abs_spread.data[(j * 6) + 1 + (cols * d * 6)].visible = (
                    d == abs_spread_date_slider and j == abs_spread_item_slider)  # maximum trace
            fig_abs_spread.data[(j * 6) + 2 + (cols * d * 6)].visible = (
                    d == abs_spread_date_slider and j == abs_spread_item_slider)  # minimum trace
            fig_abs_spread.data[(j * 6) + 3 + (cols * d * 6)].visible = (
                    d == abs_spread_date_slider and j == abs_spread_item_slider)  # average trace
            fig_abs_spread.data[(j * 6) + 4 + (cols * d * 6)].visible = (
                    d == abs_spread_date_slider and j == abs_spread_item_slider)  # quantile trace
            fig_abs_spread.data[(j * 6) + 5 + (cols * d * 6)].visible = (
                    d == abs_spread_date_slider and j == abs_spread_item_slider)  # quantile trace

    for e in range(0, len(fig_pred_price.data), 4):  # iterate over traces in steps of 4
        visibility = (e // 4 == pred_price_date_slider)  # check if current group of traces corresponds to selected date
        fig_pred_price.data[e].visible = visibility  # average trace
        fig_pred_price.data[e + 1].visible = visibility  # maximum trace
        fig_pred_price.data[e + 2].visible = visibility  # minimum trace
        fig_pred_price.data[e + 3].visible = visibility  # current trace

    fgr = create_histogram(all_ratios_pretrunc, reference_slider_ratio, target_slider_ratio, method='ratio')
    fig_ratio_distribution = fgr[0]
    ratio_prob_text = fgr[1]
    ratio_probability_lines = ratio_prob_text.split("\n")[1:-1]
    ratio_probability_components = []
    for line in ratio_probability_lines:
        ratio_probability_components.append(line)
        ratio_probability_components.append(html.Br())

    fgs = create_histogram(all_spreads_pretrunc, reference_slider_spread, target_slider_spread, method='spread')
    fig_spread_distribution = fgs[0]
    spread_prob_text = fgs[1]
    spread_probability_lines = spread_prob_text.split("\n")[1:-1]
    spread_probability_components = []
    for line in spread_probability_lines:
        spread_probability_components.append(line)
        spread_probability_components.append(html.Br())

    item_figure_generate = item_analysis(ref_idx,tperiod,time,n_years)
    item_fig = item_figure_generate[0]
    t_max = item_figure_generate[1]

    return fig_std_ratio, f'Date {dates[date_slider_ratio_std]}', f'Item {signals_ratio[a].columns.tolist()[item_slider_ratio_std]}', \
        fig_std_spread, f'Date {dates[date_slider_spread_std]}', f'Item {signals_spread[b].columns.tolist()[item_slider_spread_std]}', \
        fig_abs_ratio, f'Date {dates[abs_ratio_date_slider]}', f'Item {abs_ratio_signals[c].columns.tolist()[abs_ratio_item_slider]}', \
        fig_abs_spread, f'Date {dates[abs_spread_date_slider]}', f'Item {abs_spread_signals[d].columns.tolist()[abs_spread_item_slider]}', \
        fig_pred_price, f'Date {dates[pred_price_date_slider]}', \
        fig_ratio_distribution, ratio_probability_components, f'Reference: {col_names[reference_slider_ratio]}', f'Target: {col_names[target_slider_ratio]}',\
        fig_spread_distribution, spread_probability_components, f'Reference: {col_names[reference_slider_spread]}', f'Target: {col_names[target_slider_spread]}',\
        item_fig, t_max, f'Item: {col_names[ref_idx]}', f'Date: {all_dates[time]}'

@app.callback(
    Output('time', 'max'),
    Output('time', 'marks'),
    Input('t_max_storage', 'children')
)
def update_slider_max(t_max):
    scale = [(30, 1),
             (100, 10),
             (500, 20),
             (1000, 50),
             (5000, 100),
             (10000, 200),
             (20000, 500)]
    for max_val, step_val in scale:
        if t_max < max_val:
            step = step_val
            break
    marks = {i: str(i) for i in range(0, t_max + 1, step)}
    return t_max, marks

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
    #app.run_server(debug=True, port=8071)
