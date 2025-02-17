import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt
import io


# Load all country data
import os

path = os.getcwd() + "/"  # Gets the current directory and ensures the path ends with '/'



import glob

# Define patterns for file deletion
patterns = ["*electricity_access.csv", "*_gdp_per_capita.csv", "*_urban_population_percentage.csv"]

# Loop through each pattern and delete matching files
for pattern in patterns:
    for file in glob.glob(os.path.join(path, pattern)):
        os.remove(file)
        print(f"Deleted: {file}")

countries = [f.split('.')[0] for f in os.listdir(path) if f.endswith('.csv')]

def load_country_data(country):
    df = pd.read_csv(f"{path}{country}.csv")
    df = df.rename(columns=lambda x: x.replace('_x', '').replace('_y', ''))
    return df

def polynomial_fit(df, column):
    coeffs = np.polyfit(df['year'], df[column], 4)
    poly = np.poly1d(coeffs)
    df['3d_poly_fit'] = poly(df['year'])
    return df, f"y={coeffs[0]:.2f} + {coeffs[1]:.2f}x + {coeffs[2]:.2f}x^2 + {coeffs[3]:.2f}x^3"

app = dash.Dash(__name__)
server = app.server
app.title = "Dashboard"

app.layout = html.Div([
    html.H1("Interactive Dashboard", style={'textAlign': 'center'}),
    
    # Country selection
    html.Label("Select Country:"),
    dcc.Dropdown(id='country-dropdown', options=[{'label': c, 'value': c} for c in countries], value=countries[0]),
    
    # Variable selection
    html.Label("Select Data Type:"),
    dcc.RadioItems(id='data-type', options=[{'label': 'Internet Usage', 'value': 'percentage_of_population_using_internet'}], value='percentage_of_population_using_internet', inline=True),
    
    dcc.Graph(id='line-chart'),
    
    # Map Visualization
    html.H3("Global Data Map"),
    dcc.Slider(id='year-slider', min=2000, max=2023, step=1, value=2023, marks={i: str(i) for i in range(2000, 2024, 1)}),
    dcc.Graph(id='map-chart'),
    
    # Pie Chart Analysis
    html.H3("Pie Chart Analysis (Comparison among countries)"),
    dcc.Checklist(id='pie-countries', options=[{'label': c, 'value': c} for c in countries], value=[countries[0]], inline=True),
    dcc.Graph(id='pie-chart'),
    
    # Correlation Analysis
    html.H3("Correlation Heatmap"),
    dcc.Graph(id='corr-heatmap'),
    
    # Data Report Download
    
])

@app.callback(
    Output('line-chart', 'figure'),
    Input('country-dropdown', 'value'),
    Input('data-type', 'value')
)
def update_line_chart(country, data_type):
    df = load_country_data(country)
    df, poly_eq = polynomial_fit(df, data_type)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['year'], y=df[data_type], mode='lines+markers', name='Real Data'))
    fig.add_trace(go.Scatter(x=df['year'], y=df['poly_fit'], mode='lines', line=dict(dash='dot'), name=f'Poly Fit: {poly_eq}'))
    fig.update_layout(title=f"{country}: {data_type} Trend", xaxis_title='Year', yaxis_title='Percentage')
    return fig

@app.callback(
    Output('map-chart', 'figure'),
    Input('year-slider', 'value'),
    Input('data-type', 'value')
)
def update_map(year, data_type):
    data = []
    for c in countries:
        df = load_country_data(c)
        row = df[df['year'] == year]
        if not row.empty:
            data.append({'country': c, 'value': row[data_type].values[0]})
    df_map = pd.DataFrame(data)
    fig = px.choropleth(df_map, locations='country', locationmode='country names', color='value', range_color=[0,100], title=f"{data_type} in {year}")
    return fig

@app.callback(
    Output('pie-chart', 'figure'),
    Input('pie-countries', 'value'),
    Input('year-slider', 'value'),
    Input('data-type', 'value')
)
def update_pie(countries, year, data_type):
    data = []
    for c in countries:
        df = load_country_data(c)
        row = df[df['year'] == year]
        if not row.empty:
            data.append({'country': c, 'value': row[data_type].values[0]})
    df_pie = pd.DataFrame(data)
    fig = px.pie(df_pie, names='country', values='value', title=f"{data_type} Distribution in {year}")
    return fig

@app.callback(
    Output('corr-heatmap', 'figure'),
    Input('country-dropdown', 'value')
)
def update_heatmap(country):
    df = load_country_data(country)
    corr_matrix = df.corr()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='hot', zmin=-1, zmax=1))
    fig.update_layout(title=f"Correlation Heatmap for {country}")
    return fig





if __name__ == '__main__':
    app.run_server(debug=True)
