#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:18:37 2020

@author: paulinafigol
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import base64
import dash_table
from itertools import repeat



#def cost_benefit_analysis(y_true, y_pred,
#        perc_contacted, 
#        total_applicable_customers,
#        cost_of_reviewing_query,
#        cost_of_false_negative):
    
##    if type(cost_of_reviewing_query) not in (int, float):
 #       raise TypeError(f'cost_of_reviewing_query {cost_of_reviewing_query} not valid. Please input a number.')  
#    elif cost_of_reviewing_query <0:
#        raise InputError(f'cost_of_reviewing_query{cost_of_reviewing_query} less than 0. Please input 0 or a positive number.') 
  
#    if type(cost_of_false_negative) not in (int, float):
#        raise TypeError(f'cost_of_false_negative {cost_of_false_negative} not valid. Please input a number.')
#    elif cost_of_false_negative <0:
#        raise InputError(f'cost_of_false_negative{cost_of_false_negative} less than 0. Please input 0 or a positive number.') 
 
#    if (perc_contacted <0 or perc_contacted>100):
#        raise PercentageError(f"Percentage value {perc_contacted} outside of the limits. Please input a number between 0 and 100.") 


params = ['x_axis', 'utility2']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '#F9F0F9',#'#AF1C63',
    'text': '#FFFFFF'
}



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(style={'backgroundColor': colors['background'], 'font-family': 'Corbel Regular'},children=[
        
    html.Img(
                id='las-logo',
                src='data:image/png;base64,{}'.format(
                    base64.b64encode(
                        open('assets/Logo Guidelines v2 Black and White-61.png', 'rb').read()
                    ).decode()
                ), style={'height':'25%', 'width':'25%'}
            ),
    # html.Link(href='/assets/stylesheet.css', rel='stylesheet'),
    html.Br(),
    html.H3(
        children='Using Machine Learning to solve Data Quality Issues',
        style={'text-align': 'center', 'color': '#332F33'}
    ), 
    html.Br(),
    
    dcc.Markdown('''   
                    
The below plot uses a text-based machine learning model predicting __escalation of data quality related issues__ gathered by a financial organisation. 

The utility cost-benefit plot reflects percentage saving/cost from automating a fraction of data issues predictions. As the model predicts escalation 
of the issues, there is an optimal point of queries for which tagging should be automated - these are the most likely escalated queries. Their prompt
escalation is associated with cost and time savings.''', style={'text-align': 'center', 'color': '#6D6A6D'}),
    html.Br(),
        dcc.Markdown('''Submit the following values to create utility plot:''', 
                     style={'text-align': 'center', 'color': '#332F33'}),
        html.Div([
    html.P([
        html.Label('Cost of reviewing a query by agent:'),
        # dcc.Input(id='mother_birth', value=1952, type='number'),
        dcc.Input(
                id='reviewing_query',
                placeholder='Enter valid area..',
                type='number',
                value=None)
    ],
    style={'width': '250px', 'margin-right': 'auto','font-weight': 'bold',
           'margin-left': 'auto', 'text-align': 'center', 'color': '#332F33'}),
    # 'margin-left': '40px', 'text-align': 'center'}),

    html.P([
        html.Label('Cost of false negative (query which was falsly predicted to be non-escalation):'),
        dcc.Input(
                id='false_negative',
                placeholder='Enter valid area..',
                type='number',
                value=None
                )
    ],
    style={'width': '250px', 'margin-right': 'auto','font-weight': 'bold',
           'margin-left': 'auto', 'text-align': 'center', 'color': '#332F33'}),

#    html.P([
#        html.Label('Percentage to automate:'),
#        dcc.Input(
#                id='perc',
#                placeholder='Enter a number between 0 and 100..',
#                type='number',
#                value=None
#                )
 #   ],
 #   style={'width': '250px', 'margin-right': 'auto',
 #          'margin-left': 'auto', 'text-align': 'center', 'color': colors['text']}),

    html.P([
        html.Label('Total number of data issues:'),
        dcc.Input(
                id='queries',
                placeholder='Enter valid number..',
                type='number',
                value=None,
                min=0
                )
    ],
    style={'width': '250px', 'margin-right': 'auto', 'font-weight': 'bold',
           'margin-left': 'auto', 'text-align': 'center', 'color': '#332F33'}),
    ],

               className='input-wrapper'),    
    html.Div(id='dd2-output-container'),
#   html.Div(
#        dash_table.DataTable(id='datatable-upload-container',                          
#                             columns=[{"name": i, "id": i} for i in params]),
#        style={'height': 'auto', 'overflowY': 'scroll', 'whiteSpace': 'normal'},
#        className='six columns'
#        ),
 
        #html.Button(id='submit-button', n_clicks=0, children='Submit'),
        
    html.P([
         html.Button(id='submit-button', n_clicks=0, children='Submit')
    ],
    style={'width': '250px', 'margin-right': 'auto',
           'margin-left': 'auto', 'text-align': 'center', 'color':'white', 'BackgroundColor':'white'}),

#        style={'width': '250px', 'margin-right': 'auto',
#           'margin-left': 'auto', 'text-align': 'center'},
         html.Div(
    [
        dcc.Graph(
            id='example-graph',
                style={
            'height': 700, 'color': '#656365'
        }
        )]),
    dcc.RangeSlider(
        id='non-linear-range-slider',
        #marks={i: '{}'.format(10 ** i) for i in range(4)},
        min = 0,
        max = 1,
        value=[0,1],
        dots=False,
        step=0.01,
        updatemode='drag'
    ),
    html.Br(),
    dash_table.DataTable(id='datatable-upload-container',                          
                                 columns=[{"name": i, "id": i} for i in params])    
    ])

@app.callback(
     [dash.dependencies.Output('dd2-output-container', 'children'),
      dash.dependencies.Output('datatable-upload-container', 'data')#,
      #dash.dependencies.Output('report-text', 'children')
      ],
      [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('reviewing_query', 'value'),
    dash.dependencies.State('false_negative', 'value'),
#    dash.dependencies.State('perc', 'value'),
    dash.dependencies.State('queries', 'value')
    ])
    
def utilityfunc(n_clicks, value1, value2, value4):
      
#    perc_contacted = 100
    total_no_queries = value4
    
    df_empty = pd.DataFrame(columns = params)
    
    if (value1 == None) or (value2 ==None) or (value4==None):
        #print("Gain by automating %d%% of the queries in Pounds: %s" % (perc_contacted, Current_spent))
        return [dcc.Markdown('''Gain is unknown''',style={'color': '#332F33'})], df_empty.to_dict('records')
        
    #import train and test data sets:
    y_test_T = pd.read_csv('y_test.csv', header=None).iloc[:,1]#[1]
    X_test_T = pd.read_csv('X_test.csv')#[1]
    X_test_T = X_test_T.drop(["Unnamed: 0"],axis = 1)
    
    Textmodel = pickle.load(open('textBased_model.sav', 'rb'))
    probas = Textmodel.predict_proba(X_test_T)[:,1]
    pred = probas
    
    VALUE_TRUE_POSITIVE = abs(value1)
    VALUE_FALSE_POSITIVE = (-abs(value1))
    VALUE_TRUE_NEGATIVE = (-abs(value1))
    VALUE_FALSE_NEGATIVE = (-abs(value2))
    
    Current_spent = VALUE_FALSE_POSITIVE*total_no_queries
    #Current_spent *= total_no_queries / len(y_test_T)
    
 #   k = int(len(y_test_T)*perc_contacted / 100)
#    a = list(zip(pred,y_test_T))
#    a.sort(key=lambda x: x[0], reverse=True)
    
#    true_positives = sum(map(lambda x: x[1], a[:k]))
    #false_positives = k - true_positives
#    false_negatives = sum(map(lambda x: x[1], a[k:]))
    #true_negatives = len(y_test_T) - k - false_negatives
        
 #   k = int(perc_contacted*len(pred) /100)
    a = list(zip(pred,y_test_T.values))
    a.sort(key=lambda x: x[0], reverse=True)
    data = pd.DataFrame(a, columns=['pred_proba', 'true'])
    TPFPTNFN = pd.DataFrame(columns=['TP','FP','TN','FN'])
    
 #   for i in range(len(a)):
 #       if (round(a[i][0])==1 and a[i][1]==1).all():
 #           TPFPTNFN = TPFPTNFN.append({'TP': 1}, ignore_index=True)
 #       elif (round(a[i][0])==0 and a[i][1]==0).all():
 #               TPFPTNFN = TPFPTNFN.append({'TN': 1}, ignore_index=True)
 #       elif (round(a[i][0])==1 and a[i][1]==0).all():
 #               TPFPTNFN = TPFPTNFN.append({'FP': 1}, ignore_index=True)
 #       elif (round(a[i][0])==0 and a[i][1]==1).all():
 #               TPFPTNFN = TPFPTNFN.append({'FN': 1}, ignore_index=True)
 
    TP = list(repeat(0, len(a)))
    FP = list(repeat(0, len(a)))
    TN = list(repeat(0, len(a)))
    FN = list(repeat(0, len(a)))
    
    for i in range(len(a)):
        if (round(a[i][0])==1 and a[i][1]==1):
            TP[i] = 1
        elif (round(a[i][0])==0 and a[i][1]==0):
            TN[i] = 1
        elif (round(a[i][0])==1 and a[i][1]==0):
            FP[i] = 1
        elif (round(a[i][0])==0 and a[i][1]==1):
            FN[i] = 1
            
    TPFPTNFN.TP = TP 
    TPFPTNFN.FP = FP 
    TPFPTNFN.TN = TN 
    TPFPTNFN.FN = FN 
                
    TPFPTNFN = TPFPTNFN.fillna(0, inplace=False)
    
    data['cost'] = TPFPTNFN.apply(lambda x: VALUE_TRUE_POSITIVE if x['TP'].astype(int) \
                                     else (VALUE_FALSE_POSITIVE if x['FP'].astype(int) \
                                      else (VALUE_TRUE_NEGATIVE if x['TN'].astype(int) \
                                      else VALUE_FALSE_NEGATIVE)), axis=1)
    
    data['cost'] *= total_no_queries / len(y_test_T)
    data['cumul_cost'] = data.cost.cumsum()

    data['saving'] = data['cumul_cost']/abs(Current_spent) * 100
    
#    if perc_contacted == 0:
#        print("Gain by automating %d%% of the queries: £%s" % (perc_contacted, Current_spent))
#    else:
#        perc_idx = int(perc_contacted*len(data)/100)
#        if perc_idx == 0:
#            perc_idx = 1
        #print(f'perc_idx={perc_idx}')
#        print("Gain by automating %d%% of the queries: £%s" 
#              % (perc_contacted,data['cumul_cost'][perc_idx-1]))  
        
    utility2 = [0] + list(data.saving/100)
    NUM_SAMPLES=len(y_test_T)
    #import train and test data sets:
    x_axis = np.linspace(0, 1, NUM_SAMPLES + 1)
    
    utility2 = [round(num, 3) for num in utility2]
    x_axis = [round(num, 3) for num in x_axis]

#    fig = fig.add_trace(go.Scatter(x=x_axis, y=utility2, fill='tozeroy',
#                    mode='none' # override default markers+lines
#                    ))
    
#    fig = fig.update_layout(
#    xaxis_title="Percentage of automated cases",
#    yaxis_title="% Utility (Saving/Loss)")
    
 #   if perc_contacted != 0:
 #       rounded = round(data['cumul_cost'][perc_idx-1],2)
 #   else:
  #      rounded = Current_spent
        
        
    lis = [(x_axis[i], utility2[i]*100) for i in range(len(x_axis))]
    lis = lis[1:]
    best = max(lis, key=lambda x: x[1])

    utility_number = [0] + list(data.cumul_cost)
    lis2 = [(x_axis[i], utility_number[i]/100) for i in range(len(x_axis))]
    lis2 = lis2[1:]

    sen = '''Current spending: £%s.  
             **Maximum saving** rate is **%s%%** if %s%% percent of queries is automated.  
             This accounts for **£%s saved**. ''' % (round(Current_spent,2),\
            best[1], best[0]*100, round((best[1]*(-Current_spent))/100, 2))
            
    #, round(best[1]*(-Current_spent)/100, 2)
    #sen = "Current spending: £%s.  \
    #Gain by automating %d%% of the queries: £%s. \
    #Maximum saving rate is %s percent if %s percent of queries is automated. \
    #This accounts for £%s saved. " % (round(Current_spent,2),perc_contacted,rounded,\
    #round(best[1]),round(best[0]), round(best[1]*(-Current_spent)/100, 2))
    
    df = pd.DataFrame(
    {'x_axis': x_axis,
     'utility2': utility2
    })
    
    df = pd.DataFrame(df.groupby(['x_axis'], sort=False)['x_axis','utility2'].max())
    print(df)
                        
    return [dcc.Markdown(sen,style={'color': '#332F33'})], df.to_dict('records')
                    

@app.callback(
      dash.dependencies.Output('example-graph', 'figure'),
      [dash.dependencies.Input('datatable-upload-container', 'data'),
       dash.dependencies.Input('non-linear-range-slider', 'value')
       ])

def update_g(data, slider):
    
    fig = go.Figure(
            layout=go.Layout(
                    xaxis={
                        ####'rangeslider': {'visible':True},
                        ####'rangeselector': {'visible':True, 'buttons':[{'step':'all'}], 'yanchor' : 'top'},
                        'tickformat' : '%',
                        'autorange' : True,
                        'range' : [0,1]
                    },
                    yaxis=dict(
                            autorange = True,
                            fixedrange= False,
                            tickformat= '%',
                            range = [0,1]
                            ), paper_bgcolor='rgba(0,0,0,0)'#,
        ,font=dict(
       #family="Courier New, monospace",
        size=13,
        color='#656365'
        )
      #plot_bgcolor='rgb(255, 255, 255)'
                )
           )   

    if not data:
        return fig
    
    else:
        df = pd.DataFrame(data, columns =['x_axis', 'utility2'])

        df = df[df.x_axis.between(slider[0],slider[1])]
                    
        fig = fig.add_trace(go.Scatter(x=df.loc[:,'x_axis'], y=df.loc[:,'utility2'], fill='tozeroy',
                                       hovertemplate = 'Price: %{y:%.3f%}<extra></extra>',
                    mode='none' # override default markers+lines
                    ))
    
        fig = fig.update_layout(
                xaxis_title="Percentage of automated cases",
                yaxis_title="% Utility (Saving/Loss)"
                #,hovertemplate = "Popularity: %{df.loc[:,'x_axis']}"
                )
        
        #print('You have selected "{}"'.format(slider))
        return fig

    
if __name__ == '__main__':
    app.run_server(debug=True)
      