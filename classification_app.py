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
from typing import List, Dict, Tuple

import utils.dash_reusable_components as drc

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
@app.callback(
    Output('classification-model-params', 'data'),
    Input('classifier-dropdown', 'value'),
    Input('dropdown-svm-parameter-kernel', 'value'),
    Input('slider-svm-parameter-C-power', 'value'),
    Input('slider-svm-parameter-C-coef', 'value'),
    Input('slider-svm-parameter-degree', 'value'),
    Input('slider-svm-parameter-gamma-power', 'value'),
    Input('slider-svm-parameter-gamma-coef', 'value'),
    Input('dt-max-depth-slider', 'value'),
    Input('rf-max-depth-slider', 'value'),
    Input('dt-min-samples-split-slider', 'value'),
    Input('rf-min-samples-split-slider', 'value'),
    Input('dt-min-samples-leaf-slider', 'value'),
    Input('rf-min-samples-leaf-slider', 'value'),
    Input('n-estimators-slider', 'value'),
    Input('n-neighbors-slider', 'value'),
    Input('weights-dropdown', 'value'),
    Input('algorithm-dropdown', 'value'),
)
def create_classifier(classifier_name, svm_kernel, svm_C_power, svm_C_coef, svm_degree, 
                      svm_gamma_power, svm_gamma_coef, dt_max_depth, rf_max_depth, 
                      dt_min_samples_split, rf_min_samples_split, dt_min_samples_leaf, 
                      rf_min_samples_leaf, n_estimators, n_neighbors, weights, algorithm):
    if classifier_name == 'svm':
        model = SVC(
            kernel=svm_kernel,
            C=svm_C_coef * 10 ** svm_C_power,
            degree=svm_degree,
            gamma=svm_gamma_coef * 10 ** svm_gamma_power,
            probability=True
        )
    elif classifier_name == 'dt':
        model = DecisionTreeClassifier(
            max_depth=dt_max_depth,
            min_samples_split=dt_min_samples_split,
            min_samples_leaf=dt_min_samples_leaf
        )
    elif classifier_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf
        )
    elif classifier_name == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    model_params = model.get_params()  # Get model parameters
    return model_params
    
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
def calculate_roc(y_true: np.ndarray, y_scores: np.ndarray, target_names: List[str]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    fpr, tpr, roc_auc = {}, {}, {}
    for i, name in enumerate(target_names):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

def roc_curve_fig(model, X_test: np.ndarray, y_test: np.ndarray, target_names: List[str]) -> go.Figure:
    y_scores = model.predict_proba(X_test)
    fpr, tpr, roc_auc = calculate_roc(y_test, y_scores, target_names)

    # Plotly color sequence
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    traces = []
    for i, name in enumerate(target_names):
        color = colors[i % len(colors)]  # Cycle through colors if more classes than colors
        trace = go.Scatter(
            x=fpr[i], 
            y=tpr[i], 
            mode="lines", 
            name=f"{name} (AUC = {roc_auc[i]:.3f})",
            line={"color": color}
        )
        traces.append(trace)

    layout = go.Layout(
        title="ROC Curve",
        xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
        yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=100, r=10, t=25, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
        showlegend=True
    )

    figure = go.Figure(data=traces, layout=layout)

    # Add diagonal line
    figure.add_shape(
        type='line', 
        line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )

    return figure

def create_prediction_plot(model, X_train, X_test, y_train, y_test, Z, xx, yy, x_axis_name, y_axis_name, target_names):
    # Create the plot
    fig = go.Figure()

    # Add decision boundary
    boundary = go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z.reshape(xx.shape),
        colorscale='RdBu',
        opacity=0.6,
        showscale=False,
        contours=dict(
            showlines=False,
        ),
        hoverinfo='none'
    )
    fig.add_trace(boundary)

    # Add scatter plot for training data
    for i, target in enumerate(np.unique(y_train)):
        indices = y_train == target
        scatter_train = go.Scatter(
            x=X_train[indices, 0],
            y=X_train[indices, 1],
            mode='markers',
            name=f'Train: {target_names[i]}',
            marker=dict(
                size=10,
                symbol='circle',
                color=f'rgba({50+i*50}, {100+i*50}, {150+i*50}, 0.8)',
                line=dict(width=1, color='white')
            )
        )
        fig.add_trace(scatter_train)

    # Add scatter plot for test data
    for i, target in enumerate(np.unique(y_test)):
        indices = y_test == target
        scatter_test = go.Scatter(
            x=X_test[indices, 0],
            y=X_test[indices, 1],
            mode='markers',
            name=f'Test: {target_names[i]}',
            marker=dict(
                size=10,
                symbol='star',
                color=f'rgba({50+i*50}, {100+i*50}, {150+i*50}, 0.8)',
                line=dict(width=1, color='white')
            )
        )
        fig.add_trace(scatter_test)

    # Update layout
    fig.update_layout(
        title=dict(text=f'Decision Boundary and Data Points ({type(model).__name__})', x=0.5),
        xaxis=dict(title=x_axis_name, gridcolor='#2f3445'),
        yaxis=dict(title=y_axis_name, gridcolor='#2f3445'),
        legend=dict(x=0, y=1.05, orientation='h'),
        margin=dict(l=50, r=10, t=50, b=50),
        plot_bgcolor='#282b38',
        paper_bgcolor='#282b38',
        font=dict(color='#a5b1cd'),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

    # Add model performance information
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        text=f'Train Accuracy: {train_accuracy:.2f}<br>Test Accuracy: {test_accuracy:.2f}',
        showarrow=False,
        font=dict(color='white'),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='white',
        borderwidth=1
    )

    return fig
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
                                    "Classification Explorer",
                                    href="https://github.com/ganansuan647/dash-classification-app",
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
            dcc.Tab(label='说明', children=[
                html.H3('说明'), html.P('这是一个用于展示不同参数对不同分类器性能影响的 Dash 应用程序。'),
                # 添加作者信息
                html.Div(
                    className="author-info",
                    children=[
                        html.P("脚本作者: 苟凌云"),
                        html.P("时间: 2024年11月"),
                        html.P("邮箱: 2410100@tongji.edu.cn"),
                        html.P("组员: 苟凌云、易航、齐新宇、何帅君、李鸿豪"),
                    ],
                    style={"text-align": "center", "margin-top": "20px"}
                ),
                ]),
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
                    options=[{'label': 'Decision Tree', 'value': 'dt'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'SVM', 'value': 'svm'},
                            {'label': 'KNN', 'value': 'knn'}],
                    value='knn'
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
                html.Details(
                    id="svm-parameter-control-div",
                    open=False,
                    children=[
                        html.Summary(
                            "SVM 模型参数",
                            style={"font-size": "24px",
                                "border-bottom": "1px solid #000",
                                "margin-bottom": "5px"},
                        ),
                        # SVM 参数选择器
                        drc.NamedSlider(
                            name="Threshold",
                            id="slider-threshold",
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01,
                        ),
                        html.Button(
                            "Reset Threshold",
                            id="button-zero-threshold",
                        ),
                        drc.NamedDropdown(
                            name="Kernel",
                            id="dropdown-svm-parameter-kernel",
                            options=[
                                {
                                    "label": "Radial basis function (RBF)",
                                    "value": "rbf",
                                },
                                {"label": "Linear", "value": "linear"},
                                {
                                    "label": "Polynomial",
                                    "value": "poly",
                                },
                                {
                                    "label": "Sigmoid",
                                    "value": "sigmoid",
                                },
                            ],
                            value="rbf",
                            clearable=False,
                            searchable=False,
                        ),
                        drc.NamedSlider(
                            name="Cost (C)",
                            id="slider-svm-parameter-C-power",
                            min=-2,
                            max=4,
                            value=0,
                            marks={
                                i: "{}".format(10 ** i)
                                for i in range(-2, 5)
                            },
                        ),
                        drc.FormattedSlider(
                            id="slider-svm-parameter-C-coef",
                            min=1,
                            max=9,
                            value=1,
                        ),
                        drc.NamedSlider(
                            name="Degree",
                            id="slider-svm-parameter-degree",
                            min=2,
                            max=10,
                            value=3,
                            step=1,
                            marks={
                                str(i): str(i) for i in range(2, 11, 2)
                            },
                        ),
                        drc.NamedSlider(
                            name="Gamma",
                            id="slider-svm-parameter-gamma-power",
                            min=-5,
                            max=0,
                            value=-1,
                            marks={
                                i: "{}".format(10 ** i)
                                for i in range(-5, 1)
                            },
                        ),
                        drc.FormattedSlider(
                            id="slider-svm-parameter-gamma-coef",
                            min=1,
                            max=9,
                            value=5,
                        ),
                    ],
                ),
                html.Details(
                    id="dt-parameter-control-div",
                    open=False,
                    children=[
                        html.Summary(
                            "Decision Tree 模型参数",
                            style={"font-size": "24px",
                                "border-bottom": "1px solid #000",
                                "margin-bottom": "5px"},
                        ),
                        drc.NamedSlider(
                            name="Max Depth",
                            id="dt-max-depth-slider",
                            min=1,
                            max=20,
                            value=5,
                            marks={i: str(i) for i in range(1, 21)},
                        ),
                        drc.NamedSlider(
                            name="Min Samples Split",
                            id="dt-min-samples-split-slider",
                            min=2,
                            max=20,
                            value=2,
                            marks={i: str(i) for i in range(2, 21)},
                        ),
                        drc.NamedSlider(
                            name="Min Samples Leaf",
                            id="dt-min-samples-leaf-slider",
                            min=1,
                            max=20,
                            value=1,
                            marks={i: str(i) for i in range(1, 21)},
                        ),
                    ],
                ),
                html.Details(
                    id="rf-parameter-control-div",
                    open=False,
                    children=[
                        html.Summary(
                            "Random Forest 模型参数",
                            style={"font-size": "24px",
                                "border-bottom": "1px solid #000",
                                "margin-bottom": "5px"},
                        ),
                        drc.NamedSlider(
                            name="Number of Estimators",
                            id="n-estimators-slider",
                            min=10,
                            max=200,
                            value=100,
                            marks={i: str(i) for i in range(10, 201, 10)},
                        ),
                        drc.NamedSlider(
                            name="Max Depth",
                            id="rf-max-depth-slider",
                            min=1,
                            max=20,
                            value=5,
                            marks={i: str(i) for i in range(1, 21)},
                        ),
                        drc.NamedSlider(
                            name="Min Samples Split",
                            id="rf-min-samples-split-slider",
                            min=2,
                            max=20,
                            value=2,
                            marks={i: str(i) for i in range(2, 21)},
                        ),
                        drc.NamedSlider(
                            name="Min Samples Leaf",
                            id="rf-min-samples-leaf-slider",
                            min=1,
                            max=20,
                            value=1,
                            marks={i: str(i) for i in range(1, 21)},
                        ),
                    ],
                ),
                html.Details(
                    id="knn-parameter-control-div",
                    open=False,
                    children=[
                        html.Summary(
                            "K-Nearest Neighbors 模型参数",
                            style={"font-size": "24px",
                                "border-bottom": "1px solid #000",
                                "margin-bottom": "5px"},
                        ),
                        drc.NamedSlider(
                            name="Number of Neighbors",
                            id="n-neighbors-slider",
                            min=1,
                            max=20,
                            value=5,
                            marks={i: str(i) for i in range(1, 21)},
                        ),
                        drc.NamedDropdown(
                            name="Weights",
                            id="weights-dropdown",
                            options=[
                                {"label": "Uniform", "value": "uniform"},
                                {"label": "Distance", "value": "distance"},
                            ],
                            value="uniform",
                            clearable=False,
                            searchable=False,
                        ),
                        drc.NamedDropdown(
                            name="Algorithm",
                            id="algorithm-dropdown",
                            options=[
                                {"label": "Auto", "value": "auto"},
                                {"label": "Ball Tree", "value": "ball_tree"},
                                {"label": "KD Tree", "value": "kd_tree"},
                                {"label": "Brute", "value": "brute"},
                            ],
                            value="auto",
                            clearable=False,
                            searchable=False,
                        ),
                    ],
                ),
            ]),
        ], style={'width': '100%', 'padding': '10px', 'vertical-align': 'top'}),
    ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'height': '100vh'}),
    # 右列
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='数据集分布图', children=[dcc.Graph(id='pairplot', style={'width': '100%'})]),
            dcc.Tab(label='分类器效果图', children=[
                html.H3('分类置信图', style={'textAlign': 'center'}),
                dcc.Graph(id='prediction-confidence', style={'height': '50vh'}),
                html.Div([
                    html.Div([html.H4('ROC 曲线', style={'textAlign': 'center'}), dcc.Graph(id='roc-curve', style={'height': '45vh'})],
                             style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([html.H4('混淆矩阵', style={'textAlign': 'center'}), dcc.Graph(id='confusion-matrix', style={'height': '45vh'})],
                             style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginTop': '20px'})
            ])
        ])
    ], style={'width': '75%', 'display': 'inline-block', 'padding': '10px', 'vertical-align': 'top'}),
    dcc.Store(id='classification-model-params'),
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

@app.callback(
    [Output('prediction-confidence', 'figure'),
     Output('roc-curve', 'figure'),
     Output('confusion-matrix', 'figure')],
    [Input('classifier-dropdown', 'value'),
     Input('classification-model-params', 'data'),
     Input('dataset-dropdown', 'value'),
     Input('feature-checklist', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')
    ]
)
def train_model_and_update_figure(model_name, model_params, dataset_name, selected_features,x_axis_name,y_axis_name):
    X, y, _, target_names = load_data(dataset_name, selected_features)
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    models = {
        'svm': SVC,
        'dt': DecisionTreeClassifier,
        'rf': RandomForestClassifier,
        'knn': KNeighborsClassifier
    }
    if model_name == 'svm':
        model_params['probability'] = True
    classifier = models[model_name](**model_params)
    classifier.fit(X_train, y_train)
    
    # Create meshgrid for decision boundary, x_axis_name and y_axis_name are the selected features
    index_x = selected_features.index(x_axis_name)
    index_y = selected_features.index(y_axis_name)
    x_min, x_max = X[:, index_x].min(), X[:, index_x].max()
    y_min, y_max = X[:, index_y].min(), X[:, index_y].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # 其他特征采用平均值来代替
    pred_point_list = []
    for i in range(len(selected_features)):
        if i != index_x and i != index_y:
            mean = np.mean(X[:, i])
            pred_point_list.append(np.full(xx.ravel().shape, mean))
        elif i == index_x:
            pred_point_list.append(xx.ravel())
        elif i == index_y:
            pred_point_list.append(yy.ravel())
    
    Z = classifier.predict(np.c_[pred_point_list].T)
    
    predict_fig =  create_prediction_plot(classifier, X_train, X_test, y_train, y_test, Z, xx, yy, x_axis_name, y_axis_name, target_names)
    
    roc_fig = roc_curve_fig(classifier, X_test, y_test, target_names)
    
    cm_fig = create_cm_fig(confusion_matrix(y_test, classifier.predict(X_test)), target_names)
    
    return predict_fig, roc_fig, cm_fig
    

if __name__ == '__main__':
    app.run_server(debug=True)