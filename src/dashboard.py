import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import base64
import io

# Initialize the app
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'font-family': 'IBM Plex Sans'}),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV File', style={'font-family': 'IBM Plex Sans'}),
        multiple=False
    ),
    html.Div(id='summary-stats', style={'font-family': 'IBM Plex Sans'}),
    dcc.Dropdown(id='agent-dropdown', placeholder='Select Agent ID', style={'font-family': 'IBM Plex Sans'}),
    dcc.Tabs([
        dcc.Tab(label='Transaction Types', children=[
            dcc.Graph(id='transaction-type-graph')
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

def parse_data(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

@app.callback(
    [Output('agent-dropdown', 'options')],
    [Input('upload-data', 'contents')]
)
def update_dropdown(contents):
    if contents is None:
        return [[]]
    df = parse_data(contents)
    agent_options = [{'label': str(agent), 'value': agent} for agent in df['agent_id'].unique()]
    return [agent_options]

@app.callback(
    [Output('summary-stats', 'children'),
     Output('transaction-type-graph', 'figure'),
     Output('fraud-device-graph', 'figure'),
     Output('fraud-network-graph', 'figure'),
     Output('time-series-graph', 'figure')],
    [Input('upload-data', 'contents'),
     Input('agent-dropdown', 'value')]
)
def update_dashboard(contents, selected_agent):
    if contents is None:
        return html.Div("Upload a CSV file to see the dashboard."), {}, {}, {}, {}

    df = parse_data(contents)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if selected_agent is not None:
        df = df[df['agent_id'] == selected_agent]

    # Summary statistics
    total_transactions = len(df)
    total_fraud = df['fraud'].sum()
    fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
    summary = html.Div([
        html.H4(f"Total Transactions: {total_transactions}"),
        html.H4(f"Fraudulent Transactions: {total_fraud}"),
        html.H4(f"Fraud Rate: {fraud_rate:.2f}%")
    ], style={'font-family': 'IBM Plex Sans'})
    
    # Transaction Types Distribution
    type_fig = px.histogram(df, x='type', color='fraud', barmode='group',
                            title='Transaction Types Distribution')
    
    # Fraud by Device
    device_fig = px.histogram(df, x='device', color='fraud', barmode='group',
                              title='Fraud Occurrence by Device')
    
    # Fraud by Network
    network_fig = px.histogram(df, x='network', color='fraud', barmode='group',
                               title='Fraud Occurrence by Network')
    
    # Time Series Analysis per Agent ID
    time_fig = px.line(df.groupby(df['timestamp'].dt.floor('T')).size().reset_index(name='count'),
                       x='timestamp', y='count', title=f'Transactions Over Time for Agent {selected_agent if selected_agent else "All"}')
    
    return summary, type_fig, device_fig, network_fig, time_fig

if __name__ == '__main__':
    app.run_server(debug=True)