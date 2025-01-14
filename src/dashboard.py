import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import argparse

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
parser.add_argument('--input_file', help='CSV Input file for producing the dashboard', type=str, default='/home/molocco/fraud-detection-simulator/data/fraud_simulation_1K_activities.csv')
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
        dcc.Tab(label='Transaction Types', children=[
            dcc.Graph(id='transaction-type-graph'),
            html.Div(id='transaction-behavior-graphs', style={'margin-top': '20px'})
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
    [Output('agent-dropdown', 'options')],
    [Input('agent-dropdown', 'value')]  # You can now remove the upload data input
)
def update_dropdown(selected_agent):
    # Get unique agents from the dataframe
    agent_options = [{'label': str(agent), 'value': agent} for agent in df['agent_id'].unique()]
    return [agent_options]

@app.callback(
    [Output('summary-stats', 'children'),
     Output('transaction-type-graph', 'figure'),
     Output('fraud-device-graph', 'figure'),
     Output('fraud-network-graph', 'figure'),
     Output('time-series-graph', 'figure'),
     Output('transaction-behavior-graphs', 'children')],
    [Input('agent-dropdown', 'value')]
)
def update_dashboard(selected_agent):
    # Filter the dataframe if an agent is selected
    filtered_df = df if selected_agent is None else df[df['agent_id'] == selected_agent]

    # Summary statistics
    total_transactions = len(filtered_df)
    total_fraud = filtered_df['fraud'].sum()
    fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
    summary = html.Div([
        html.H4(f"Total Transactions: {total_transactions}"),
        html.H4(f"Fraudulent Transactions: {total_fraud}"),
        html.H4(f"Fraud Rate: {fraud_rate:.2f}%")
    ], style={'font-family': 'IBM Plex Sans'})
    
    # Transaction Types Distribution
    type_fig = px.histogram(filtered_df, x='action', color='fraud', barmode='group', histnorm='probability', title='Transaction Types Distribution')
    
    # Fraud by Device
    device_fig = px.histogram(filtered_df, x='device', color='fraud', barmode='group', histnorm='probability', title='Fraud Occurrence by Device')
    
    # Fraud by Network
    network_fig = px.histogram(filtered_df, x='network', color='fraud', barmode='group', histnorm='probability', title='Fraud Occurrence by Network')
    
    # Time Series Analysis per Agent ID
    time_fig = px.line(filtered_df.groupby(filtered_df['timestamp'].dt.floor('T')).size().reset_index(name='count'),
                       x='timestamp', y='count', title=f'Transactions Over Time for Agent {selected_agent if selected_agent else "All"}')
    
    # Normalized Histograms for each behavior type
    behavior_figs = []
    unique_behaviors = filtered_df['behavior'].unique()
    for behavior in unique_behaviors:
        behavior_df = filtered_df[filtered_df['behavior'] == behavior]
        behavior_fig = px.histogram(
            behavior_df,
            x='action',
            color='fraud',
            barmode='group',
            histnorm='probability',  # Normalize to show frequencies
            title=f"Normalized Action Distribution for Behavior: {behavior}"
        )
        behavior_fig.update_layout(
            yaxis_title='Frequency',  # Update the y-axis label for clarity
            xaxis_title='Action'
        )
        behavior_figs.append(html.Div([
            dcc.Graph(figure=behavior_fig)
        ], style={'margin-bottom': '20px'}))
    
    return summary, type_fig, device_fig, network_fig, time_fig, behavior_figs


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)  # Disable reloader to avoid multiple runs
