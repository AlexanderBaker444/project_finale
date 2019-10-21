import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title='car value'

########### Set up the layout

app.layout = html.Div(children=[
    html.H1('Classification of Car Values'),
    html.Img(src=app.get_asset_url('car.png'), style={'width': 'auto', 'height': 'auto'}),
    html.Div([
        html.Div([
            html.Div([
                html.H6('Buying'),
                
                dcc.RadioItems(
                    id='slider-1',
                    options=[{'label':'low','value':0},{'label':'med','value':1},{'label':'high','value':2},{'label':'vhigh','value':3}],
                    value=1
                ),
            html.Br(),
            ], className='twelve columns'),
            
            html.Div([
                html.H6('Maint'),
                dcc.Slider(
                    id='slider-2',
                    min=0,
                    max=3,
                    step=1,
                    marks={0:'low',1:'med',2:'high',3:'vhigh'},
                    value=1
                ),
            html.Br(),
            ], className='twelve columns'),
            
            
            
            html.Div([
                html.H6('Doors'),
                dcc.RadioItems(
                    id='slider-3',
                    options=[{'label':'2','value':0},{'label':'3','value':1},{'label':'4','value':2},{'label':'5more','value':3}],
                    value=1
                ),
            html.Br(),
            ], className='twelve columns'),
            html.Div([
                html.H6('Persons'),
                dcc.RadioItems(
                    id='slider-4',
                    options=[{'label':'2','value':0},{'label':'4','value':1},{'label':'more','value':2}],
                    value=1
                ),
            html.Br(),
            ], className='twelve columns'),
            html.Div([
                html.H6('Lug Boot'),
                dcc.Slider(
                    id='slider-5',
                    min=0,
                    max=2,
                    step=1,
                    marks={0:'small',1:'med',2:'big'},
                    value=1
                ),
            html.Br(),
            ], className='twelve columns'),
            
               html.Div([
                html.H6('Safety'),
                dcc.Dropdown(
                    id='slider-6',
                    options=[{'label':'low','value':0},{'label':'med','value':1},{'label':'high','value':2}],
                    value=1
                ),
            html.Br(),
            ], className='twelve columns'),
            
            
            
            
            html.Div([
                html.H6('Model Type'),
                dcc.Dropdown(
                    id='k-drop',
                    options=[{'label':'logistic model', 'value': 1},
                            {'label':'nearest neighbors model', 'value': 2},
                            {'label': 'random forest model', 'value': 3},
                            {'label': 'support vector machine model', 'value': 4},],
                    value=1
                ),
            html.Br(),
            ], className='twelve columns'),
            
          
        ], className='twelve columns'),
        html.Div([
            html.H6('Graph'),
                dcc.Graph(
                    id='fig-1',
                ),
        ], className='twelve columns'),
        
    html.Br(),
    html.A('Code on Github', href='https://github.com/AlexBaker444/car_project'),
    
    ])
])

######### Define Callbacks

@app.callback(Output('fig-1', 'figure'),
              [Input('k-drop', 'value'),
               Input('slider-1', 'value'),
               Input('slider-2', 'value'),
               Input('slider-3', 'value'),
              Input('slider-4', 'value'),
              Input('slider-5', 'value'),
              Input('slider-6', 'value')])
def display_results(k, value0, value1,value2,value3,value4,value5):
    # read in the correct model
    
    file = open(f'resources/model{k}.pkl', 'rb')
    model=pickle.load(file)
    file.close()
    
    file=open('resources/data_car.pkl','rb')
    df=pickle.load(file)
    file.close()
    
    # define the new observation from the slide values
    new_observation=[[value0, value1,value2,value3,value4,value5]]
    prediction=model.predict(new_observation)
    specieslist=['unacc', 'acc', 'good','vgood']
    species_prediction=specieslist[prediction[0]]
    
    
    labels=df['car_value'].value_counts().index
    values=df['car_value'].value_counts().values
    
    colors = ['lightslategray',] * 5
    colors[prediction[0]] = 'yellow'

    fig1 = go.Figure(data=[go.Bar(
    x=labels,
    y=values,
    marker_color=colors # marker color can be a single color value or an iterable
    )])
    
    fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "xy"},{"type": "domain"}]],
    )

    fig.add_trace(go.Bar(x=labels,y=values,marker_color=colors),
                  row=1, col=1)


    fig.add_trace(go.Pie(labels=labels,values=values,marker_colors=colors),
                  row=1, col=2)


    fig.update_layout(height=700, showlegend=False)
    
    return fig



############ Execute the app
if __name__ == '__main__':
    app.run_server()