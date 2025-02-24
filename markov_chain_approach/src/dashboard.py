import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

import argparse
import base64
import io


# Initialize the app
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Load data from the input file
def load_data(input_file):
    return pd.read_csv(input_file)

# Set up argument parser to get the input file path
parser = argparse.ArgumentParser(
    description='Script for generating the dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--input_file', help='CSV Input file for producing the dashboard', type=str,)
args = parser.parse_args()

# Load data using the input file argument
df = load_data(args.input_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# App layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'font-family': 'IBM Plex Sans'}),

    html.Div(id='summary-stats', style={'font-family': 'IBM Plex Sans'}),
    dcc.Dropdown(id='agent-dropdown', placeholder='Select Agent ID', style={'font-family': 'IBM Plex Sans'}),
    dcc.Tabs([
        dcc.Tab(label='activity', children=[
            dcc.Graph(id='activity-activity_type-graph'),
            html.Div(id='activity-behavior-graphs', style={'margin-top': '20px'})

        ], style={'font-family': 'IBM Plex Sans'}),
        dcc.Tab(label='Fraud by Device/Network', children=[
            dcc.Graph(id='fraud-device-graph'),
            dcc.Graph(id='fraud-network-graph')
        ], style={'font-family': 'IBM Plex Sans'}),
        dcc.Tab(label='Time Series Analysis', children=[
            dcc.Graph(id='time-series-graph')
        ], style={'font-family': 'IBM Plex Sans'})
    ])
], style={'font-family': 'IBM Plex Sans'})

@app.callback(
    Output('agent-dropdown', 'options'),
    Input('agent-dropdown', 'value')
)
def update_dropdown(selected_agent):
    # Get unique agents from the dataframe and add an 'All' option
    agent_options = [{'label': 'All', 'value': 'all'}]  # 'All' option
    agent_options += [{'label': str(agent), 'value': agent} for agent in df['real_id'].unique()]
    return agent_options

@app.callback(
    [Output('summary-stats', 'children'),
     Output('activity-activity_type-graph', 'figure'),
     Output('fraud-device-graph', 'figure'),
     Output('fraud-network-graph', 'figure'),

     Output('time-series-graph', 'figure'),
     Output('activity-behavior-graphs', 'children')],
    [Input('agent-dropdown', 'value')]
)
def update_dashboard(selected_agent):
    # Filter the dataframe based on the selected agent
    if selected_agent == 'all' or selected_agent is None:
        filtered_df = df  # Show all agents if 'all' is selected
    else:
        filtered_df = df[df['real_id'] == selected_agent]

    # Summary statistics
    total_activities = len(filtered_df)
    total_fraud = filtered_df['is_fraud'].sum()

    fraud_rate = (total_fraud / total_activities) * 100 if total_activities > 0 else 0
    summary = html.Div([
        html.H4(f"Total activities: {total_activities}"),
        html.H4(f"Fraudulent activities: {total_fraud}"),
        html.H4(f"Fraud Rate: {fraud_rate:.2f}%")
    ], style={'font-family': 'IBM Plex Sans'})
    
    # activity  Distribution
    activity_type_fig = px.histogram(filtered_df, x='activity_type', color='is_fraud', barmode='group', histnorm='probability', title='Activity Distribution by fraud (1)/not fraud(0)')
    
    # Fraud by Device
    device_fig = px.histogram(filtered_df, x='device', color='is_fraud', barmode='group', histnorm='probability', title='Fraud Occurrence by Device')
    
    # Fraud by Network
    network_fig = px.histogram(filtered_df, x='network', color='is_fraud', barmode='group', histnorm='probability', title='Fraud Occurrence by Network')
    
    # Time Series Analysis per Agent ID
    # Time Series Analysis per Agent ID, with separate lines for each behavior
    time_series_df = filtered_df.groupby(['timestamp', 'behavior']).size().reset_index(name='count')

    time_fig = px.line(
        time_series_df,
        x='timestamp',
        y='count',
        color='behavior',  # Different line for each behavior
        title=f'Activities Over Time for Agent {selected_agent if selected_agent != "all" else "All"}'
    )

    time_fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Activity Count',
        legend_title='Behavior'
    )

    time_fig = px.line(filtered_df.groupby(filtered_df['timestamp'].dt.floor('s')).size().reset_index(name='count'),
                       x='timestamp', y='count', title=f'Activities Over Time for Agent {selected_agent if selected_agent != "all" else "All"}')
    

    # Normalized Histograms for each behavior activity_type
    behavior_figs = []
    unique_behaviors = filtered_df['behavior'].unique()
    '''
    for behavior in unique_behaviors:
        behavior_df = filtered_df[filtered_df['behavior'] == behavior]
        behavior_fig = px.histogram(
            behavior_df,
            x='activity_type',
            color='fraud',
            barmode='group',
            histnorm='probability',  # Normalize to show frequencies
            title=f"Normalized Activity activity_type Distribution for Behavior: {behavior}"
        )
        behavior_fig.update_layout(
            yaxis_title='Frequency',  # Update the y-axis label for clarity
            xaxis_title='Activity activity_type'
        )
        behavior_figs.append(html.Div([
            dcc.Graph(figure=behavior_fig)
        ], style={'margin-bottom': '20px'}))
    '''
    # Overlaid activity  Distribution by Behavior
    overlay_fig = px.histogram(
        filtered_df,
        x='activity_type',
        color='behavior',  # Different colors for each behavior
        barmode='group',  # Overlay histograms
        histnorm='probability',  # Normalize to show frequencies
        title='Activity by Behavior'
    )
    overlay_fig.update_layout(
        xaxis_title='Activity activity_type',
        yaxis_title='Frequency',
        legend_title='Behavior'
    )
    
    return summary, activity_type_fig, device_fig, network_fig, time_fig, [dcc.Graph(figure=overlay_fig)] + behavior_figs



if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)  # Disable reloader to avoid multiple runs

