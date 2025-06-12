import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Keep only rows where TIME_PERIOD is a 4-digit year
df = df[df['TIME_PERIOD'].astype(str).str.fullmatch(r'\d{4}')]

# Now convert to int
df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(int)

# Ensure OBS_VALUE is numeric and drop rows with NaN
df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
df = df.dropna(subset=['OBS_VALUE'])

# Prepare forecasted data
forecast_frames = []
for REF_AREA in df['REF_AREA'].unique():
    REF_AREA_df = df[df['REF_AREA'] == REF_AREA].sort_values('TIME_PERIOD')
    X = REF_AREA_df['TIME_PERIOD'].values.reshape(-1, 1)
    y = REF_AREA_df['OBS_VALUE'].values
    model = LinearRegression()
    model.fit(X, y)
    # Forecast next two years
    last_year = REF_AREA_df['TIME_PERIOD'].max()
    future_years = np.array([last_year + 1, last_year + 2]).reshape(-1, 1)
    future_values = model.predict(future_years)
    forecast_df = pd.DataFrame({
        'TIME_PERIOD': future_years.flatten(),
        'REF_AREA': REF_AREA,
        'OBS_VALUE': future_values,
        'Type': 'Forecast'
    })
    forecast_frames.append(forecast_df)

# Mark original data as 'Actual'
df['Type'] = 'Actual'

# Combine actual and forecast data
df_all = pd.concat([df] + forecast_frames, ignore_index=True)

# Plot with Plotly Express, using REF_AREA as a filter (slicer)
fig = px.line(
    df_all,
    x='TIME_PERIOD',
    y='OBS_VALUE',
    color='Type',
    line_dash='Type',
    facet_col=None,
    title='Indexed Property Values with 2-Year Forecast',
    labels={'TIME_PERIOD': 'Year', 'OBS_VALUE': 'Property Value'},
    hover_data=['REF_AREA']
)

# Add slicer (dropdown) for REF_AREA
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=[
                dict(label=REF_AREA,
                     method='update',
                     args=[{'visible': (df_all['REF_AREA'] == REF_AREA).tolist()},
                           {'title': f'Indexed Property Values for {REF_AREA}'}])
                for REF_AREA in df_all['REF_AREA'].unique()
            ],
            direction="down",
            showactive=True,
            x=1.15,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

# By default, show all REF_AREAs
fig.for_each_trace(lambda t: t.update(visible=True))

fig.show()