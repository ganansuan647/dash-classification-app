import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
from dash import callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import numpy as np
from sklearn.manifold import TSNE
import umap

# Create Dash app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Function to load dataset and process it
def load_data(dataset_name, selected_features):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
        X, y = data.data, data.target
        feature_names, target_names = data.feature_names, data.target_names
    elif dataset_name == 'RC墩柱破坏模式':
        data = pd.read_excel('dataset/数据集2_RC墩柱破坏模式预测.xlsx')
        data = data.rename(columns={data.columns[0]: 'Index'}).drop(columns=['Index'])
        
        # Extract features and target
        X = data.drop(columns=['Failure Mode'])
        y = pd.Categorical(data['Failure Mode']).codes  # Convert to numerical codes
        
        # Convert categorical variables to numeric
        X = pd.get_dummies(X, columns=['Test configuration', 'Section shape', 'Hoop type'])
        feature_names = X.columns.tolist()
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        target_names = sorted(data['Failure Mode'].unique())
    elif dataset_name == '桥梁震后损伤状态':
        data = pd.read_excel('dataset/数据集6_桥梁震后损伤状态预测.xlsx')
        data = preprocess_bridge_data(data)
        
        # Extract features and target
        X = data.drop(columns=['Tag'])
        y = pd.Categorical(data['Tag']).codes  # Convert to numerical codes
        
        feature_names = X.columns.tolist()
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # target_names = sorted(data['Tag'].unique())
        target_names = ["绿", "黄", "红"]
    
    if selected_features is None:
        return feature_names
    else:
        # Filter selected features
        df = pd.DataFrame(X, columns=feature_names)
        X = df[selected_features].values
        return X, y, feature_names, target_names

# 添加字段选择组件到布局
@app.callback(
    Output('feature-selector-container', 'children'),
    Input('dataset-dropdown', 'value')
)
def create_feature_selector(dataset_name):
    # 根据数据集名称获取可用字段
    feature_names = load_data(dataset_name, None)
    available_features = feature_names
    
    # Get initial selected features (first 5 or all if less than 5)
    initial_features = available_features[:5] if len(available_features) > 5 else available_features

    return html.Div([
        html.H4("特征选择"),
        html.Div([
            dbc.Button("全选", id="select-all-btn", n_clicks=0, className="me-2"),
            dbc.Button("全不选", id="deselect-all-btn", n_clicks=0, className="me-2"),
            dbc.Button("反选", id="toggle-select-btn", n_clicks=0)
        ], className="mb-3"),
        dbc.Checklist(
            id='feature-checklist',
            options=[{'label': feat, 'value': feat} for feat in available_features],
            value=initial_features,  # Default to first 5 or all features
            inline=False
        ),
    ])
    
# function to update 
@app.callback(
    Output('feature-checklist', 'value'),
    Input('select-all-btn', 'n_clicks'),
    Input('deselect-all-btn', 'n_clicks'),
    Input('toggle-select-btn', 'n_clicks'),
    State('feature-checklist', 'options'),
    State('feature-checklist', 'value')
)
def update_feature_selection(select_all, deselect_all, toggle_select, options, current_value):
    # Get the ID of the button that triggered the callback
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Update the selected features based on the button clicked
    if button_id == 'select-all-btn':
        return [option['value'] for option in options]
    elif button_id == 'deselect-all-btn':
        return []
    elif button_id == 'toggle-select-btn':
        return [option for option in options if option['value'] not in current_value]

    return current_value

def create_pairplot(X, y, dataset_name, selected_features, target_names):
    # Create DataFrame from X and y
    df = pd.DataFrame(X, columns=selected_features)
    df['类别'] = pd.Categorical.from_codes(y, target_names)
    
    # Initialize figure with subplots
    fig_pairplot = make_subplots(rows=len(selected_features), cols=len(selected_features))
    
    # Define color scheme for all classes
    if dataset_name == '桥梁震后损伤状态':
        color_scheme = {target_names[0]: 'green', target_names[1]: 'yellow', target_names[2]: 'red'}
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Default plotly colors
        color_scheme = {name: color for name, color in zip(target_names, colors)}
    
    # Create scatter plots for pairs
    for i, feat1 in enumerate(selected_features):
        for j, feat2 in enumerate(selected_features):
            if i != j:  # Off-diagonal: scatter plots
                for target in target_names:
                    mask = df['类别'] == target
                    fig_pairplot.add_trace(
                        go.Scatter(
                            x=df[mask][feat1],
                            y=df[mask][feat2],
                            mode='markers',
                            name=target,
                            showlegend=(j == 0 and i == 1),  # Show legend only once
                            marker=dict(
                                size=6, 
                                opacity=0.6,
                                color=color_scheme[target]  # Use consistent color
                            )
                        ),
                        row=i+1,
                        col=j+1,
                    )
            else:  # Diagonal: density plots
                for target in target_names:
                    mask = df['类别'] == target
                    data = df[mask][feat1].values
                    if len(data) > 0:
                        fig_pairplot.add_trace(
                        go.Histogram(x=data,name=target,showlegend=False,marker=dict(color=color_scheme[target], opacity=0.6),nbinsx=30),row=i+1,col=j+1)
    
    # Update figure layout
    fig_pairplot.update_layout(
        title='特征配对图',
        height=1100,
        width=1200,
        showlegend=True,
        dragmode='select'
    )
    
    # Update axes labels
    for i, feat in enumerate(selected_features):
        fig_pairplot.update_xaxes(title_text=feat, row=len(selected_features), col=i+1)
        fig_pairplot.update_yaxes(title_text=feat, row=i+1, col=1)
    
    return fig_pairplot

def preprocess_bridge_data(data):
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    # 如果有未命名的列则舍弃
    data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col])
    return data

# Function to create classifier
def create_classifier(classifier_name, C_value):
    classifiers = {
        'logreg': LogisticRegression(C=C_value),
        'dt': DecisionTreeClassifier(),
        'rf': RandomForestClassifier(),
        'svm': SVC(probability=True),
        'knn': KNeighborsClassifier(),
        'gb': GradientBoostingClassifier()
    }
    return classifiers.get(classifier_name, LogisticRegression(C=C_value))

def create_prediction_plot(
    model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold
):
    # Get train and test score from model
    y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
    y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
    train_score = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    test_score = accuracy_score(y_true=y_test, y_pred=y_pred_test)

    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    range = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))

    # Colorscale
    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    cscale = [
        [0.0000000, "#ff744c"],
        [0.1428571, "#ff916d"],
        [0.2857143, "#ffc0a8"],
        [0.4285714, "#ffe7dc"],
        [0.5714286, "#e5fcff"],
        [0.7142857, "#c8feff"],
        [0.8571429, "#9af8ff"],
        [1.0000000, "#20e6ff"],
    ]

    # Create the plot
    # Plot the prediction contour of the SVM
    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - range,
        zmax=scaled_threshold + range,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=cscale,
        opacity=0.9,
    )

    # Plot the threshold
    trace1 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo="none",
        contours=dict(
            showlines=False, type="constraint", operation="=", value=scaled_threshold
        ),
        name=f"Threshold ({scaled_threshold:.3f})",
        line=dict(color="#708090"),
    )

    # Plot Training Data
    trace2 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(size=10, color=y_train, colorscale=bright_cscale),
    )

    # Plot Test Data
    trace3 = go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode="markers",
        name=f"Test Data (accuracy={test_score:.3f})",
        marker=dict(
            size=10, symbol="triangle-up", color=y_test, colorscale=bright_cscale
        ),
    )

    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0, trace1, trace2, trace3]
    figure = go.Figure(data=data, layout=layout)

    return figure

def create_cm_fig(confusion_matrix, target_names):
    fig_cm = px.imshow(confusion_matrix, labels=dict(x="Predicted", y="Actual"), x=target_names, y=target_names, title="Confusion Matrix")
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            fig_cm.add_annotation(
                x=j, y=i,
                text=str(confusion_matrix[i, j]),
                showarrow=False,
                font=dict(color="black" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "white")
            )
    return fig_cm

# Function to calculate ROC curve and AUC
def calculate_roc(y_test, y_scores, target_names):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

# Layout of the Dash app
app.layout = html.Div([
    html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Support Vector Machine (SVM) Explorer",
                                    href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("TBS-logo.png"),
                                         style={"background-color": "transparent",
                                                "transform": "scale(1.5)",
                                                "transform-origin": "right"},
                                         )
                            ],
                            href="https://faculty-civileng.tongji.edu.cn/wangxiaowei/",
                        ),
                    ],
                )
            ],
        ),
    # 左列
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='说明', children=[html.H3('说明'), html.P('这是一个用于展示不同参数对分类器性能影响的 Dash 应用程序。')]),
            dcc.Tab(label='数据集', children=[
                html.H3('选择数据集'),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[{'label': 'Iris Dataset', 'value': 'Iris'},
                            {'label': 'RC墩柱破坏模式', 'value': 'RC墩柱破坏模式'},
                            {'label': '桥梁震后损伤状态', 'value': '桥梁震后损伤状态'}],
                    value='Iris'
                ),
                html.Div(id='feature-selector-container'),  # Move feature selector container here
            ]),
            dcc.Tab(label='分类器', children=[
                html.H3('选择分类器'),
                dcc.Dropdown(
                    id='classifier-dropdown',
                    options=[{'label': 'Logistic Regression', 'value': 'logreg'},
                            {'label': 'Decision Tree', 'value': 'dt'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'SVM', 'value': 'svm'},
                            {'label': 'KNN', 'value': 'knn'},
                            {'label': 'Gradient Boosting', 'value': 'gb'}],
                    value='logreg'
                ),
                html.H3('选择绘图坐标参数'),
                html.Div([
                    html.Label('X轴:'),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[],
                        value=None,
                        clearable=False
                    ),
                    html.Label('Y轴:'),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[],
                        value=None,
                        clearable=False
                    )
                ]),
                html.H3('调整分类器参数'),
                html.Div(id='param-selector-container'),
            ]),
        ], style={'width': '100%', 'padding': '10px', 'vertical-align': 'top'}),
    ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'height': '100vh'}),
    # 右列
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='数据集分布图', children=[dcc.Graph(id='pairplot', style={'width': '100%'})]),
            dcc.Tab(label='分类边界图', children=[
                dcc.Graph(id='decision-boundary', style={'height': '50vh'}),
                html.Div([
                    html.Div([html.H4('ROC 曲线', style={'textAlign': 'center'}), dcc.Graph(id='roc-curve', style={'height': '45vh'})],
                             style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([html.H4('混淆矩阵', style={'textAlign': 'center'}), dcc.Graph(id='confusion-matrix', style={'height': '45vh'})],
                             style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginTop': '20px'})
            ])
        ])
    ], style={'width': '75%', 'display': 'inline-block', 'padding': '10px', 'vertical-align': 'top'})
])

@app.callback(
    [Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options')],
    [Input('dataset-dropdown', 'value'),
     Input('feature-checklist', 'value')],
)
def update_axis_options(dataset_name, selected_features):
    if dataset_name and selected_features:
        options = [{'label': feat, 'value': feat} for feat in selected_features]
        return options, options
    return [], []

@app.callback(
    [Output('x-axis-dropdown', 'value'),
     Output('y-axis-dropdown', 'value')],
    [Input('dataset-dropdown', 'value'),
     Input('feature-checklist', 'value')]
)
def update_axis_values(dataset_name, selected_features):
    if dataset_name and selected_features:
        return selected_features[0], selected_features[-1]
    return None, None

# 添加参数选择器布局到布局
@app.callback(
    Output('param-selector-container', 'children'),
    Input('classifier-dropdown', 'value')
)
def update_param_selector(classifier_name):
    if classifier_name == 'logreg':
        return html.Div([
            html.Label('C (正则化强度)'),
            dcc.Slider(id='C-slider', min=0.01, max=10, step=0.01, value=1, marks={i: str(i) for i in range(11)}),
            html.Label('solver (优化算法)'),
            dcc.Dropdown(id='solver-dropdown', options=[
                {'label': 'lbfgs', 'value': 'lbfgs'},
                {'label': 'liblinear', 'value': 'liblinear'},
                {'label': 'sag', 'value': 'sag'},
                {'label': 'saga', 'value': 'saga'}
            ], value='lbfgs')
        ])
    elif classifier_name == 'dt':
        return html.Div([
            html.Label('max_depth (最大深度)'),
            dcc.Slider(id='max-depth-slider', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(1, 21)}),
            html.Label('min_samples_split (内部节点再划分所需最小样本数)'),
            dcc.Slider(id='min-samples-split-slider', min=2, max=20, step=1, value=2, marks={i: str(i) for i in range(2, 21)}),
            html.Label('min_samples_leaf (叶子节点最少样本数)'),
            dcc.Slider(id='min-samples-leaf-slider', min=1, max=20, step=1, value=1, marks={i: str(i) for i in range(1, 21)})
        ])
    elif classifier_name == 'rf':
        return html.Div([
            html.Label('n_estimators (树的数量)'),
            dcc.Slider(id='n-estimators-slider', min=10, max=200, step=10, value=100, marks={i: str(i) for i in range(10, 201, 10)}),
            html.Label('max_depth (最大深度)'),
            dcc.Slider(id='max-depth-slider', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(1, 21)}),
            html.Label('min_samples_split (内部节点再划分所需最小样本数)'),
            dcc.Slider(id='min-samples-split-slider', min=2, max=20, step=1, value=2, marks={i: str(i) for i in range(2, 21)}),
            html.Label('min_samples_leaf (叶子节点最少样本数)'),
            dcc.Slider(id='min-samples-leaf-slider', min=1, max=20, step=1, value=1, marks={i: str(i) for i in range(1, 21)})
        ])
    elif classifier_name == 'svm':
        return html.Div([
            html.Label('C (正则化强度)'),
            dcc.Slider(id='C-slider', min=0.01, max=10, step=0.01, value=1, marks={i: str(i) for i in range(11)}),
            html.Label('kernel (核函数)'),
            dcc.Dropdown(id='kernel-dropdown', options=[
                {'label': 'linear', 'value': 'linear'},
                {'label': 'poly', 'value': 'poly'},
                {'label': 'rbf', 'value': 'rbf'},
                {'label': 'sigmoid', 'value': 'sigmoid'}
            ], value='rbf'),
            html.Label('gamma (核系数)'),
            dcc.Slider(id='gamma-slider', min=0.001, max=1, step=0.001, value=0.1, marks={i/100: str(i/100) for i in range(1, 101, 10)})
        ])
    elif classifier_name == 'knn':
        return html.Div([
            html.Label('n_neighbors (邻居数)'),
            dcc.Slider(id='n-neighbors-slider', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(1, 21)}),
            html.Label('weights (权重函数)'),
            dcc.Dropdown(id='weights-dropdown', options=[
                {'label': 'uniform', 'value': 'uniform'},
                {'label': 'distance', 'value': 'distance'}
            ], value='uniform'),
            html.Label('algorithm (算法)'),
            dcc.Dropdown(id='algorithm-dropdown', options=[
                {'label': 'auto', 'value': 'auto'},
                {'label': 'ball_tree', 'value': 'ball_tree'},
                {'label': 'kd_tree', 'value': 'kd_tree'},
                {'label': 'brute', 'value': 'brute'}
            ], value='auto')
        ])
    elif classifier_name == 'gb':
        return html.Div([
            html.Label('n_estimators (树的数量)'),
            dcc.Slider(id='n-estimators-slider', min=10, max=200, step=10, value=100, marks={i: str(i) for i in range(10, 201, 10)}),
            html.Label('learning_rate (学习率)'),
            dcc.Slider(id='learning-rate-slider', min=0.01, max=1, step=0.01, value=0.1, marks={i: str(i/100) for i in range(0, 101, 10)}),
            html.Label('max_depth (最大深度)'),
            dcc.Slider(id='max-depth-slider', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(1, 21)}),
            html.Label('min_samples_split (内部节点再划分所需最小样本数)'),
            dcc.Slider(id='min-samples-split-slider', min=2, max=20, step=1, value=2, marks={i: str(i) for i in range(2, 21)}),
            html.Label('min_samples_leaf (叶子节点最少样本数)'),
            dcc.Slider(id='min-samples-leaf-slider', min=1, max=20, step=1, value=1, marks={i: str(i) for i in range(1, 21)})
        ])
    return html.Div([])

@app.callback(
    Output('pairplot', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('feature-checklist', 'value')],
)
def update_pairplot(dataset_name, selected_features):
    if dataset_name and selected_features:
        X, y, _, target_names = load_data(dataset_name, selected_features)
        return create_pairplot(X, y, dataset_name, selected_features, target_names)
    return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)