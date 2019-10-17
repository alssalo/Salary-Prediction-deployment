import numpy as np
import pandas as pd
from skimage import io, data, transform
from time import sleep

from sklearn.externals import joblib

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table

import plotly.express as px
import plotly.graph_objects as go

import dash_canvas
from dash_canvas.components import image_upload_zone
from dash_canvas.utils import (
    image_string_to_PILImage,
    array_to_data_url,
    parse_jsonstring_line,
    brightness_adjust,
    contrast_adjust,
)
from registration import register_tiles
from utils import StaticUrlPath
import pathlib

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server
app.config.suppress_callback_exceptions = True

# Load the model
model = joblib.load(open("model.pkl", 'rb'))

# declare and intialize  global variable
global salarydata_df
salarydata_df=pd.read_csv("train_data.csv")

global dictlist1
global dictlist2
global dictlist3
dictlist1 = []
dictlist2 = []
dictlist3 = []

for col in list(salarydata_df.jobType.unique()):
    dictlist1.append({'label':col,'value':col})

for col in list(salarydata_df.degree.unique()):
    dictlist2.append({'label':col,'value':col})

for col in list(salarydata_df.major.unique()):
    dictlist3.append({'label':col,'value':col})


# get relative data folder
PATH = pathlib.Path(__file__).parent

DATA_PATH = PATH.joinpath("data").resolve()

"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------App functions----------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# app functions

def demo_explanation():
    # Markdown files
    with open(PATH.joinpath("demo.md"), "r") as file:
        demo_md = file.read()

    return html.Div(
        html.Div([dcc.Markdown(demo_md, className="markdown")]),
        style={"margin": "10px"},
    )


def instructions():
    return html.P(
        children=[
            """
    For Prediction:
    -Choose values of the menu  
    -Press predict Button
    
    For Visualization:
    - Choose jobType field  
    - Press Viualization
    """
        ],
        className="instructions-sidebar",
    )


def feature_workshop(df,feat_list,col_name):
    group=df.groupby(feat_list)
    mean_encode=pd.DataFrame({col_name: group["salary"].mean()})
    df = pd.merge(df, mean_encode, on=feat_list, how='left')
    return df

"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------App layout---------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# app layout

app.layout = html.Div(
    children=[
        html.Div([
            html.H1(children="Salary Prediction App"),
                instructions(),
                html.Div([
                        html.Button(
                            "LEARN MORE",
                            className="button_instruction",
                            id="learn-more-button",),
                         ],
                        className="mobile_buttons",
                        ),
                html.Div(
                         # Empty child function for the callback
                         html.Div(id="demo-explanation", children=[])
                        ),
                html.Div([
                        html.Label("Job Type"),
                        dcc.Dropdown(
                                    id='my-dropdown',
                                    options=dictlist1,
                                    value='CEO',style={
                                    'margin':'0 auto',
                                    #'display': 'inherit',
                                    'text-align':' cente',
                                    'height': '0.5px', 
                                    'width': '325px',
                                    "margin-left": "29px",
                                    "margin-top": "2px",
                                    'color':'rgb(0,0,0)', 
                                    'font-size': "100%",
                                    'font-color': "rgba(255,255,255,255)",
                                    'background': 'rgba(0,0,0,0)',
                                    'position': 'relative',},
                                    ),
                                     ],
                    className="mobile_forms",
                ),
                         html.Br(),
                         html.Div([
                         html.Label("Degree"),
                         dcc.Dropdown(
                                    id='my-dropdown1',
                                    options=dictlist2,
                                    value='MASTERS',style={
                                    'margin':'0 auto',
                                    #'display': 'inherit',
                                    'text-align':' cente',
                                    'height': '0.5px', 
                                    'width': '325px',
                                    "margin-top": "2px",
                                    "margin-left": "29px",
                                    'color':'rgb(0,0,0)', 
                                    'font-size': "100%",
                                    'font-color': "rgba(255,255,255,255)",
                                    'background': 'rgba(0,0,0,0)',
                                    'position': 'relative',},
                                    ),
                                     ],
                    className="mobile_forms",
                ),
                         html.Br(),
                         html.Div([
                        html.Label("Major"),
                        dcc.Dropdown(
                            id='my-dropdown2',
                            options=dictlist3,
                            value='MATH',style={
                                'margin':'0 auto',
                                #'display': 'inherit',
                                'text-align':' cente',
                                'height': '0.5px', 
                                'width': '325px',
                                "margin-top": "2px",
                                "margin-left": "29px",
                                'color':'rgb(0,0,0)', 
                                'font-size': "100%",
                                'fontColor': "rgba(255,255,255,255)",
                                'backgroundColor': 'rgba(0,0,0,0)',
                                'position': 'relative',
                                },
                         ),
                    ],
                    className="mobile_forms",
                ),
                html.Br(),
                html.Div(
                    [
                        html.Label("Industry"),
                        dcc.RadioItems(
                            id="downsample",
                            options=[
                                {"label": "HEALTH", "value": "HEALTH"},
                                {"label": "AUTO", "value": "AUTO"},
                                {"label": "FINANCE", "value": "FINANCE"},
                                {"label": "EDUCATION", "value": "EDUCATION"},
                                {"label": "OIL", "value": "OIL"},
                                {"label": "SERVICE", "value": "SERVICE"},
                            ],
                            value="OIL",
                            labelStyle={"display": "inline-block",'text-align': 'justify'},
                            style={"margin-top": "-15px",'margin-right':"-15px",'fontColor':'#FFFFFF'},
                        ),
                        html.Label("Years of Experience (in [0-24] range)"),
                        dcc.Input(
                            id="exp", type="number", value=1, min=0, max=24
                        ),
                        html.Br(),
                        html.Label("Miles From Metropolis (in [1-99] range)"),
                        dcc.Input(
                            id="miles", type="number", value=1, min=1, max=99
                        ),
                        html.Br(),
                    ],
                    className="radio_items",
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Div([
                          html.Button(
                                    "Predict ", id="button-stitch", className="button_submit",style={'background-color':'turquoise'}
                                      ),
                          html.Button(
                                     "Visualize ", id="button-stitch2", className="button_submit",style={'background-color':'#4181FF'}
                                      ),
                          html.Div(id='container123')
                        ]),
                html.Br(),
                html.Div(id='my-div',style={ 'font-size': '3.2rem', 'line-height': '1.2' ,'letter-spacing': '-.1rem', 'margin-bottom': '2rem', 'color':'#407DFA','margin-left':'12px'})

            ],
            className="four columns instruction",
        ),
        html.Div(
            [
            html.Div(id="headingfortabs"),

             dcc.Tabs(
                    id="stitching-tabs",
                    value="canvas-tab",
                    children=[
                        dcc.Tab(label="Years of Exp Vs Salary", value="canvas-tab"),
                        dcc.Tab(label="Degree Vs Salary", value="result-tab"),
                        dcc.Tab(label="Industry Vs salary", value="help-tab"),
                    ],
                    className="tabs",
                ),
                html.Div(
                    id="tabs-content-example",
                    className="canvas",
                    style={"text-align": "left", "margin": "auto"},
                ),
                html.Div(className="upload_zone", id="upload-stitch", children=[]),
                html.Div(id="sh_x", hidden=True),
                html.Div(id="stitched-res", hidden=True),
                dcc.Store(id="memory-stitch"),
            ],
            className="eight columns result",
        ),
        
    ],
    className="row twelve columns",
)

"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------App callback definitions----------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# app callback definitions

@app.callback(
    Output("tabs-content-example", "children"), [Input("stitching-tabs", "value")]#,[State("my-dropdown", "value")]#,[State("button-stitch2", "n_clicks")]
)
def fill_tab(tab):#,state):
    #if state!=None:
        if (tab == "canvas-tab" ):
            #fig=make_figure(1,state)
            return [
                html.H2('Experience Vs Salary'),
                dcc.Graph(id='my-graph')
                #dcc.Graph(fig[0])
        ]
        elif tab == "result-tab":
            return [
                html.H2('Degree Vs Salary'),
                dcc.Graph(id='my-graph2')
        ]
        return [
            html.H2('Industry Vs Salary'),
            dcc.Graph(id='my-graph3')
    ]



@app.callback([Output('my-graph', 'figure'),Output('my-graph2', 'figure'),Output('my-graph3', 'figure')] ,[Input("button-stitch2", "n_clicks"),Input("stitching-tabs", "value")],
[State("my-dropdown", "value")])#,State("button-stitch2", "n_clicks")])
def make_figure(vals,tab,selected_dropdown_value):#,state):
    if(vals!=None): #and state!=None):
        a=feature_workshop(salarydata_df[:100000],["jobType","yearsExperience"],"job_exp_mean")
        data=a[a.jobType==selected_dropdown_value]
        if selected_dropdown_value ==None:
            selected_dropdown_value="CFO"
        fig1=px.line(a[a.jobType=="CEO"], x="yearsExperience", y="job_exp_mean",render_mode='webgl')#,title=a.jobType)

        a2=feature_workshop(salarydata_df[:1000],["jobType","degree"],"job_exp_mean")
        data2=a2[a2.jobType==selected_dropdown_value]

        fig2 = go.Figure(data=[go.Bar(x=data2.degree, y=data2.job_exp_mean)])
        fig2.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)

        a3=feature_workshop(salarydata_df[:1000],["jobType","industry"],"job_exp_mean")
        data3=a3[a3.jobType==selected_dropdown_value]

        fig3 = go.Figure(data=[go.Bar(x=data3.industry, y=data3.job_exp_mean)])
        fig3.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)

        return(fig1,fig2,fig3)#go.Figure(go.Scatter(x=data.yearsExperience, y=data.job_exp_mean)))

       
@app.callback(
    Output("my-div", "children"),
    [Input("button-stitch", "n_clicks")],
    [
    State("my-dropdown", "value"),
    State("my-dropdown1", "value"),
    State("my-dropdown2", "value"),
    State("downsample", "value"),
    State("exp", "value"),
    State("miles", "value")
    ]) 
def predict(vals,jobtype,Degree,Major,Industry,Exp,Miles):
    features=[jobtype,Degree,Major,Industry,Exp,Miles]
    features[-1]=int(features[-1])
    features[-2]=int(features[-2])
    train=pd.read_csv("train_data.csv")
    train.drop(["salary","companyId"],axis=1,inplace=True)
    final_features = pd.DataFrame([features],columns=["jobType","degree","major","industry","yearsExperience","milesFromMetropolis"])
    final_features=pd.concat([final_features,train]).reset_index(drop=True)
    prediction = model.predict(final_features[0:50])
    prediction=prediction[0]
    if(vals==None):
        return(None)
    if (prediction<0 or prediction>400):
        return("Unusual feature combination!! Please try again with more valid feature combination.......")
    else:
        return("Prediction: "+str(round(float(prediction),3))+str(vals))
        #return("Working")




@app.callback(
    [Output("demo-explanation", "children"), Output("learn-more-button", "children")],
    [Input("learn-more-button", "n_clicks")],
)
def learn_more(n_clicks):
    if n_clicks == None:
        n_clicks = 0
    if (n_clicks % 2) == 1:
        n_clicks += 1
        return (
            html.Div(
                className="demo_container",
                style={"margin-bottom": "30px"},
                children=[demo_explanation()],
            ),
            "Close",
        )

    n_clicks += 1
    return (html.Div(), "Learn More")


"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------Main function----------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


if __name__ == "__main__":
    app.run_server(debug=True)
