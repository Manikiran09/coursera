import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and clean the data
def load_and_clean_data():
    file_path = os.path.join(os.path.expanduser("~/Desktop"), "final_with_coordinates.csv")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file 'final_with_coordinates.csv' was not found at {file_path}. Please ensure it is on your Desktop.")
        raise
    
    # Clean headers and data
    df.columns = [col.strip().replace('"', '') for col in df.columns]
    df = df.dropna(subset=['date_and_time_utc', 'payload_mass', 'launch_site', 'customer_type'])
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date_and_time_utc'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    df['year'] = df['date'].dt.year
    df['payload_mass'] = pd.to_numeric(df['payload_mass'], errors='coerce').fillna(0)
    df['launch_success_rate_rolling'] = pd.to_numeric(df['launch_success_rate_rolling'], errors='coerce').fillna(0)
    df['booster_landing_success_rate'] = pd.to_numeric(df['booster_landing_success_rate'], errors='coerce').fillna(0)
    df['is_night_launch'] = df['is_night_launch'].map({'True': True, 'False': False, True: True, False: False, None: False}).fillna(False)
    df['is_starlink'] = df['is_starlink'].map({'True': True, 'False': False, True: True, False: False, None: False}).fillna(False)
    return df

df = load_and_clean_data()

# Aggregate data for visualizations
launch_counts = df.groupby('year_month').size().reset_index(name='count')
payload_mass_hist = df['payload_mass']
launch_site_counts = df.groupby(['launch_site', 'latitude', 'longitude']).size().reset_index(name='launch_count')
success_rates = df[['date', 'launch_success_rate_rolling', 'booster_landing_success_rate']]
customer_types = df['customer_type'].value_counts().reset_index(name='count')
customer_types.columns = ['customer_type', 'count']

# Create visualizations
def create_launch_frequency_plot(df_filtered):
    counts = df_filtered.groupby('year_month').size().reset_index(name='count')
    fig = px.line(counts, x='year_month', y='count', title='Launch Frequency Over Time',
                  labels={'year_month': 'Year-Month', 'count': 'Number of Launches'})
    fig.update_layout(
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='#ffffff',
        title_font_size=20,
        xaxis_tickangle=45,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_payload_mass_histogram(df_filtered):
    fig = px.histogram(df_filtered, x='payload_mass', nbins=30, title='Payload Mass Distribution',
                       labels={'payload_mass': 'Payload Mass (kg)', 'count': 'Number of Launches'})
    fig.update_traces(marker=dict(color='#10b981'))
    fig.update_layout(
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='#ffffff',
        title_font_size=20,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_launch_site_map(df_filtered):
    site_counts = df_filtered.groupby(['launch_site', 'latitude', 'longitude']).size().reset_index(name='launch_count')
    fig = px.scatter_geo(site_counts, lat='latitude', lon='longitude', size='launch_count',
                         hover_name='launch_site', title='Launch Sites',
                         projection='natural earth')
    fig.update_traces(marker=dict(color='#f59e0b'))
    fig.update_layout(
        geo=dict(bgcolor='#1f2937', landcolor='#374151'),
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='#ffffff',
        title_font_size=20,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_success_rate_plot(df_filtered):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['launch_success_rate_rolling'],
                             mode='lines', name='Launch Success Rate', line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['booster_landing_success_rate'],
                             mode='lines', name='Booster Landing Success Rate', line=dict(color='#ef4444')))
    fig.update_layout(
        title='Success Rate Trends',
        xaxis_title='Date',
        yaxis_title='Success Rate',
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='#ffffff',
        title_font_size=20,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_customer_type_pie(df_filtered):
    counts = df_filtered['customer_type'].value_counts().reset_index(name='count')
    counts.columns = ['customer_type', 'count']
    fig = px.pie(counts, values='count', names='customer_type', title='Launches by Customer Type',
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='#ffffff',
        title_font_size=20,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# Define the app layout
app.layout = html.Div(className='bg-gray-900 text-white min-h-screen p-6', children=[
    html.Link(
        rel='stylesheet',
        href='https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'
    ),
    html.H1('SpaceX Launch Dashboard', className='text-4xl font-bold mb-6 text-center'),
    
    # Add filters
    html.Div(className='flex justify-center mb-6', children=[
        html.Div(className='mr-4', children=[
            html.Label('Select Year:', className='block text-lg mb-2'),
            dcc.Dropdown(
                id='year-filter',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': str(year), 'value': year} for year in sorted(df['year'].unique())],
                value='All',
                className='bg-gray-800 text-white p-2 rounded'
            )
        ]),
        html.Div(children=[
            html.Label('Select Rocket Type:', className='block text-lg mb-2'),
            dcc.RadioItems(
                id='rocket-filter',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Falcon 9', 'value': 'Falcon 9'},
                    {'label': 'Falcon Heavy', 'value': 'Falcon Heavy'}
                ],
                value='All',
                className='flex space-x-4'
            )
        ])
    ]),
    
    # Display interesting fact
    html.Div(className='text-center mb-6', children=[
        html.H2('Interesting Fact', className='text-2xl font-semibold mb-2'),
        html.P('Starlink missions dominate SpaceX launches, with over {}% of launches dedicated to deploying SpaceX\'s satellite constellation, showcasing their focus on global internet coverage.'.format(
            round(100 * len(df[df['is_starlink'] == True]) / len(df), 1)
        ), className='text-lg')
    ]),
    
    # Visualizations
    html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-6', children=[
        html.Div(className='bg-gray-800 p-4 rounded-lg', children=[
            dcc.Graph(id='launch-frequency-plot', style={'height': '400px'})
        ]),
        html.Div(className='bg-gray-800 p-4 rounded-lg', children=[
            dcc.Graph(id='payload-mass-histogram', style={'height': '400px'})
        ]),
        html.Div(className='bg-gray-800 p-4 rounded-lg', children=[
            dcc.Graph(id='launch-site-map', style={'height': '400px'})
        ]),
        html.Div(className='bg-gray-800 p-4 rounded-lg', children=[
            dcc.Graph(id='success-rate-plot', style={'height': '400px'})
        ]),
        html.Div(className='bg-gray-800 p-4 rounded-lg col-span-1 md:col-span-2', children=[
            dcc.Graph(id='customer-type-pie', style={'height': '400px'})
        ])
    ])
])

# Callback to update visualizations based on filters
@app.callback(
    [
        Output('launch-frequency-plot', 'figure'),
        Output('payload-mass-histogram', 'figure'),
        Output('launch-site-map', 'figure'),
        Output('success-rate-plot', 'figure'),
        Output('customer-type-pie', 'figure')
    ],
    [Input('year-filter', 'value'), Input('rocket-filter', 'value')]
)
def update_graphs(selected_year, selected_rocket):
    df_filtered = df.copy()
    
    if selected_year != 'All':
        df_filtered = df_filtered[df_filtered['year'] == int(selected_year)]
    
    if selected_rocket != 'All':
        df_filtered = df_filtered[df_filtered['rocket_type'] == selected_rocket]
    
    return (
        create_launch_frequency_plot(df_filtered),
        create_payload_mass_histogram(df_filtered),
        create_launch_site_map(df_filtered),
        create_success_rate_plot(df_filtered),
        create_customer_type_pie(df_filtered)
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)