import pandas as pd
import pathlib
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
import numpy as np
import plotly.graph_objects as go
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import scipy
from scipy import stats

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
df = pd.read_excel(DATA_PATH.joinpath("metallics dataset.xlsx"))

app = Dash(__name__ , title="Metallics Analysis")
server = app.server

# global inputs
long_run_days = 3650
number_days = 30
corr_matrix_days = 45
weighting_type = 'square'

# access dataset
#df = pd.read_excel("metallics dataset.xlsx")
data = df.copy()
dates = data.iloc[:, 0][:number_days]

def error_correct(data):
    window_size = 15
    threshold = 0.10
    flat_threshold = 0.20

    for col in data.columns.drop('Date'):
        # remove '' values
        data[col] = data[col].replace('', np.NaN)

        # generate rolling mean and std
        col_mean = data[col].rolling(window = window_size).mean()
        col_std = data[col].rolling(window = window_size).std()

        # check if difference between data point is beyond threshold, identify as error
        #data.loc[(data[col].diff().abs() > data[col] * flat_threshold), col] = np.NaN

        # check if value is 0 or negative, identify as error
        data.loc[data[col] <= 0, col] = np.NaN

        # after the window check if mean or std has changed beyond threshold, identify as error
        #data.loc[(col_mean.diff().abs() > col_mean * threshold) | (col_std.diff().abs() > col_std * threshold), col] = np.NaN

        # correct errors by filling NaN
        data[col] = data[col].ffill(limit = 30)

    return data

data_model = error_correct(data)

# GENERATE DATA ANALYSIS

# GENERATE RATIOS AND SPREADS
data = data_model.iloc[:, 0]
data = data.iloc[:number_days]
col_names = list(data)[1:]
cols = len(data_model.drop('Date', axis=1).columns)  # number of columns in dataset
all_ratios = pd.DataFrame()
all_spreads = pd.DataFrame()

# hold fixed a column
for col_name in data_model.drop('Date', axis=1).columns:
    reference_col = data_model[col_name]
    reference_spreads = pd.DataFrame()
    reference_ratios = pd.DataFrame()

    # cycle though all columns to give ratio
    for col in data_model.drop('Date', axis=1).columns:
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

minimum_ratios = all_ratios.min()
maximum_ratios = all_ratios.max()

minimum_spreads = all_spreads.min()
maximum_spreads = all_spreads.max()

#quantiles
u_quantile_ratio = all_ratios.quantile(0.95)
l_quantile_ratio = all_ratios.quantile(0.05)

u_quantile_spread = all_spreads.quantile(0.95)
l_quantile_spread = all_spreads.quantile(0.05)

#truncate by number of days
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
dates = dates.values.tolist()

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
    sign_change_indices_spreads = ((diffs_spreads.iloc[:, i].shift() * diffs_spreads.iloc[:, i]) < 0).to_numpy().nonzero()[0]

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

#create the predictions from the long run averages
predictions_ratios = []
predictions_spread = []
for i in range(len(dates)):
    current_price = current.iloc[:1]
    predictions_i_ratio = []
    predictions_i_spread = []
    predictions_ratios.append(predictions_i_ratio)
    predictions_spread.append(predictions_i_spread)
    for j in range(0,len(LT_ave_ratios), cols):
        ave_ratio = LT_ave_ratios.iloc[j:j + cols]
        ave_spread = LT_ave_spreads.iloc[j:j + cols]

        pred_ratio = current_price * ave_ratio
        pred_spread = current_price + ave_spread

        predictions_i_ratio.append(pred_ratio)
        predictions_i_spread.append(pred_spread)

weightings = []
#weighting method absolute, square or logarithmic
for k in range(0,len(correlation), cols):
    wgt = correlation[k:k+cols]
    if weighting_type == 'absolute':
        wgt = wgt.abs()
    elif weighting_type == 'square':
        wgt = wgt.pow(2)
    elif weighting_type == 'log':
        wgt = np.log(wgt + 1.0000001)

    for n in range(cols):
        wgt.iloc[n] = wgt.iloc[n] / wgt.iloc[n].sum()
    weightings.append(wgt)

weighted_average_ratio = []
weighted_average_spread = []
for i in range(len(dates)):
    wgt_pred_ratio = []
    wgt_pred_spread = []
    for j in range(cols):
        wgt_item_ratio = (predictions_ratios[i][j] * weightings[i].iloc[cols-1-j]).sum(axis=1)
        wgt_item_spread = (predictions_spread[i][j] * weightings[i].iloc[cols-1-j]).sum(axis=1)
        wgt_pred_ratio.append(wgt_item_ratio)
        wgt_pred_spread.append(wgt_item_spread)

    weighted_average_ratio.append(wgt_pred_ratio)
    weighted_average_spread.append(wgt_pred_spread)

#clean the result and convert to a DataFrame
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

average_pred_price = (weighted_average_clean_ratio + weighted_average_clean_spread) / 2

minimum_pred_price = weighted_average_clean_ratio.where(weighted_average_clean_ratio < weighted_average_clean_spread, weighted_average_clean_spread)
maximum_pred_price = weighted_average_clean_ratio.where(weighted_average_clean_ratio > weighted_average_clean_spread, weighted_average_clean_spread)

column_conversion = maximum_pred_price.filter(regex='\$/short ton|\$/cwt').columns
current[column_conversion] = current[column_conversion] / 1.10231

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
                if -1.00 < normalized_value < -0.50:
                    colors.append('dark green')
                elif -0.50 < normalized_value < -0.25:
                    colors.append('green')
                elif 0.-25 < normalized_value < 0.25:
                    colors.append('orange')
                elif 0.25 < normalized_value < 0.50:
                    colors.append('red')
                elif 0.50 < normalized_value < 1:
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
            name = 'LR average'
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
                elif 0.-25 < normalized_value < 0.25:
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
            name = 'LR average'
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
            max= cols - 1,
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
            max= cols - 1,
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
    Input('date_slider_ratio_std', 'value'),
    Input('item_slider_ratio_std', 'value'),
    Input('date_slider_spread_std', 'value'),
    Input('item_slider_spread_std', 'value'),
    Input('abs_ratio_date_slider', 'value'),
    Input('abs_ratio_item_slider', 'value'),
    Input('abs_spread_date_slider', 'value'),
    Input('abs_spread_item_slider', 'value'),
    Input('pred_price_date_slider', 'value'),
)

def update_figure_ratio_std(date_slider_ratio_std, item_slider_ratio_std, date_slider_spread_std,item_slider_spread_std,abs_ratio_date_slider,abs_ratio_item_slider,abs_spread_date_slider, abs_spread_item_slider, pred_price_date_slider):
    for a in range(len(dates)):
        for g in range(cols):
            fig_std_ratio.data[(g * 3) + (cols * a * 3)].visible = (a == date_slider_ratio_std and g == item_slider_ratio_std)  # signal trace
            fig_std_ratio.data[(g * 3) + 1 + (cols * a * 3)].visible = (a == date_slider_ratio_std and g == item_slider_ratio_std)  # maximum trace
            fig_std_ratio.data[(g * 3) + 2 + (cols * a * 3)].visible = (a == date_slider_ratio_std and g == item_slider_ratio_std)  # minimum trace

    for b in range(len(dates)):
        for h in range(cols):
            fig_std_spread.data[(h * 3) + (cols * b * 3)].visible = (b == date_slider_spread_std and h == item_slider_spread_std)  # signal trace
            fig_std_spread.data[(h * 3) + 1 + (cols * b * 3)].visible = (b == date_slider_spread_std and h == item_slider_spread_std)  # maximum trace
            fig_std_spread.data[(h * 3) + 2 + (cols * b * 3)].visible = (b == date_slider_spread_std and h == item_slider_spread_std)  # minimum trace

    for c in range(len(dates)):
        for i in range(cols):
            fig_abs_ratio.data[(i * 6) + 0 + (cols * c * 6)].visible = (c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # signal trace
            fig_abs_ratio.data[(i * 6) + 1 + (cols * c * 6)].visible = (c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # maximum trace
            fig_abs_ratio.data[(i * 6) + 2 + (cols * c * 6)].visible = (c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # minimum trace
            fig_abs_ratio.data[(i * 6) + 3 + (cols * c * 6)].visible = (c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # average trace
            fig_abs_ratio.data[(i * 6) + 4 + (cols * c * 6)].visible = (c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # quantile trace
            fig_abs_ratio.data[(i * 6) + 5 + (cols * c * 6)].visible = (c == abs_ratio_date_slider and i == abs_ratio_item_slider)  # quantile trace

    for d in range(len(dates)):
        for j in range(cols):
            fig_abs_spread.data[(j * 6) + 0 + (cols * d * 6)].visible = (d == abs_spread_date_slider and j == abs_spread_item_slider)  # signal trace
            fig_abs_spread.data[(j * 6) + 1 + (cols * d * 6)].visible = (d == abs_spread_date_slider and j == abs_spread_item_slider)  # maximum trace
            fig_abs_spread.data[(j * 6) + 2 + (cols * d * 6)].visible = (d == abs_spread_date_slider and j == abs_spread_item_slider)  # minimum trace
            fig_abs_spread.data[(j * 6) + 3 + (cols * d * 6)].visible = (d == abs_spread_date_slider and j == abs_spread_item_slider)  # average trace
            fig_abs_spread.data[(j * 6) + 4 + (cols * d * 6)].visible = (d == abs_spread_date_slider and j == abs_spread_item_slider)  # quantile trace
            fig_abs_spread.data[(j * 6) + 5 + (cols *  d * 6)].visible = (d == abs_spread_date_slider and j == abs_spread_item_slider)  # quantile trace

    for e in range(0, len(fig_pred_price.data), 4):  # iterate over traces in steps of 4
        visibility = (e // 4 == pred_price_date_slider)  # check if current group of traces corresponds to selected date
        fig_pred_price.data[e].visible = visibility  # average trace
        fig_pred_price.data[e + 1].visible = visibility  # maximum trace
        fig_pred_price.data[e + 2].visible = visibility  # minimum trace
        fig_pred_price.data[e + 3].visible = visibility  # current trace

    return fig_std_ratio, f'Date {dates[date_slider_ratio_std]}', f'Item {signals_ratio[a].columns.tolist()[item_slider_ratio_std]}', \
        fig_std_spread, f'Date {dates[date_slider_spread_std]}', f'Item {signals_spread[b].columns.tolist()[item_slider_spread_std]}', \
        fig_abs_ratio, f'Date {dates[abs_ratio_date_slider]}', f'Item {abs_ratio_signals[c].columns.tolist()[abs_ratio_item_slider]}', \
        fig_abs_spread, f'Date {dates[abs_spread_date_slider]}', f'Item {abs_spread_signals[d].columns.tolist()[abs_spread_item_slider]}', \
        fig_pred_price, f'Date {dates[pred_price_date_slider]}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)