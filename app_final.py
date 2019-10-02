"""""


"""""
# import packages

from flask import request, jsonify, render_template,Flask,redirect


import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.externals import joblib

#import requests

# intialize global variables

global salarydata_df
salarydata_df=pd.read_csv("train_data.csv")

global dictlist
dictlist = []
for col in list(salarydata_df.jobType.unique()):
    dictlist.append({'label':col,'value':col})

# intialize dash and flask app variables

flask_app = Flask(__name__)

dash_app = dash.Dash(__name__, server=flask_app, url_base_pathname='/dash_app/')

# Load the model
model = joblib.load(open("model.pkl", 'rb'))
   

"""
            ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            
            ......................................................................Dash app defintion begins.........................................................................................
            
            ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
def feature_workshop(df,feat_list,col_name):
    group=df.groupby(feat_list)
    mean_encode=pd.DataFrame({col_name: group["salary"].mean()})
    df = pd.merge(df, mean_encode, on=feat_list, how='left')
    return df

# app layout

dash_app.layout = html.Div([
    html.Div([

        html.H1('Salary Analysis'),
        # First let users choose stocks
        html.H2('Choose the job Type'),
        dcc.Dropdown(
            id='my-dropdown',
            options=dictlist,
            value='CEO'
        ),
        html.H2('Experience Vs Salary'),
        dcc.Graph(id='my-graph'),
        html.P('')

    ],style={'width': '40%', 'display': 'inline-block'}),
    html.Div([
        html.H2('Salary Vs Degree'),
        dcc.Graph(id='my-graph2'),
        html.P(''),
        html.H2('Salary Vs Industry'),
        dcc.Graph(id='my-graph3'),
        html.P('')
    ], style={'width': '55%', 'float': 'right', 'display': 'inline-block'})
  
    
]
                       )#,style={'background-image': 'url("https://mondaymorning.nitrkl.ac.in/uploads/post/placement.jpg")'})



# app callbacks

@dash_app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def make_figure(selected_dropdown_value):
    a=feature_workshop(salarydata_df,["jobType","yearsExperience"],"job_exp_mean")
    data=a[a.jobType==selected_dropdown_value]
    #fig = go.Figure()
    #fig = go.Figure(go.Scatter(x=data.yearsExperience, y=data.job_exp_mean))
    #fig.add_trace(go.Scatter(x=data.yearsExperience, y=data.job_exp_mean,mode='lines',name='lines'))
    #fig.add_trace(go.Scatter(x=data.yearsExperience, y=data.salary,mode='markers', name='markers'))
    fig1=px.line(a[a.jobType=="CEO"], x="yearsExperience", y="job_exp_mean")
    return(fig1)#go.Figure(go.Scatter(x=data.yearsExperience, y=data.job_exp_mean)))


@dash_app.callback(Output('my-graph2', 'figure'), [Input('my-dropdown', 'value')])
def make_figure(selected_dropdown_value):
    a=feature_workshop(salarydata_df,["jobType","degree"],"job_exp_mean")
    data=a[a.jobType==selected_dropdown_value]
    fig = go.Figure(data=[go.Bar(x=data.degree, y=data.job_exp_mean)])
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)
    return(fig)#go.Figure(go.Scatter(x=data.yearsExperience, y=data.job_exp_mean)))

@dash_app.callback(Output('my-graph3', 'figure'), [Input('my-dropdown', 'value')])
def make_figure(selected_dropdown_value):
    a=feature_workshop(salarydata_df,["jobType","industry"],"job_exp_mean")
    data=a[a.jobType==selected_dropdown_value]
    fig = go.Figure(data=[go.Bar(x=data.industry, y=data.job_exp_mean)])
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)
    return(fig)#go.Figure(go.Scatter(x=data.yearsExperience, y=data.job_exp_mean)))

"""
         ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
         
         ...................................................................... Flask app begins.................................................................................................
         
         ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""



@flask_app.route('/')
def home():
    return render_template('index.html')


@flask_app.route('/predict',methods=['POST'])
def predict():
 
    return render_template('predict_form.html')

@flask_app.route('/analyse_button',methods=['POST'])
def analyse_button():
    return redirect('/dash_app/')

@flask_app.route('/predict/predict_button2',methods=['POST'])
def predict_button2():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    features[-1]=int(features[-1])
    features[-2]=int(features[-2])
    train=pd.read_csv("train_data.csv")
    train.drop(["salary","companyId"],axis=1,inplace=True)
    final_features = pd.DataFrame([features],columns=["jobType","degree","major","industry","yearsExperience","milesFromMetropolis"])
    final_features=pd.concat([final_features,train]).reset_index(drop=True)
    prediction = model.predict(final_features[0:50])
    prediction=prediction[0]

    output = prediction
    
    return render_template('predict_form.html', prediction_text='Employee Salary should be $ {}'.format(output))

@flask_app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

"""
===============================================================================================================================================================================================================================================

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

===============================================================================================================================================================================================================================================
"""
if __name__ == "__main__":
    flask_app.run(debug=True)


