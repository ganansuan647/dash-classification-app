import os
from typing import Dict, List, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset_name == 'Iris':
        data = datasets.load_iris()
        X, y = data.data, data.target
        feature_names, target_names = data.feature_names, data.target_names
    elif dataset_name == 'RC_Pier_Column_Failure_Mode':
        file_path = os.path.join(base_dir, "dataset/dataset2_RC_Pier_Column_Failure_Mode.xlsx")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_excel(file_path)
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
    elif dataset_name == 'Bridge_Post_Earthquake_Damage_State':
        file_path = os.path.join(base_dir, "dataset/dataset6_Bridge_Post_Earthquake_Damage_State.xlsx")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_excel(file_path)
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

# function to update 
@app.callback(
    [Output('feature-checklist', 'options'),
    Output('feature-checklist', 'value'),
    Output('current-dataset','data')],
    Input('dataset-dropdown', 'value'),
    Input('current-dataset','data'),
    Input('select-all-btn', 'n_clicks'),
    Input('deselect-all-btn', 'n_clicks'),
    Input('toggle-select-btn', 'n_clicks'),
    Input('feature-checklist', 'options'),
    Input('feature-checklist', 'value'),
)
def update_feature_options_and_selection(dataset_name,current_dataset, select_all, deselect_all, toggle_select, options, current_value):
    if dataset_name!= current_dataset:
        # 根据数据集名称获取可用字段
        feature_names = load_data(dataset_name, None)
        available_features = feature_names
        
        # Get initial selected features (first 4 or all if less than 4)
        initial_features = available_features[:4] if len(available_features) > 4 else available_features
        
        selected_features = initial_features
    else:
        available_features = [option['value'] for option in options]
        # Get the ID of the button that triggered the callback
        ctx = callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Update the selected features based on the button clicked
        if button_id == 'select-all-btn':
            selected_features = [option['value'] for option in options]
        elif button_id == 'deselect-all-btn':
            selected_features = []
        elif button_id == 'toggle-select-btn':
            selected_features = [option['value'] for option in options if option['value'] not in current_value]
        else:
            selected_features = current_value
            
    return [{'label': feat, 'value': feat} for feat in available_features],selected_features,dataset_name

def create_pairplot(X, y, dataset_name, selected_features, target_names):
    # Create DataFrame from X and y
    df = pd.DataFrame(X, columns=selected_features)
    df['类别'] = pd.Categorical.from_codes(y, target_names)
    
    # Initialize figure with subplots
    fig_pairplot = make_subplots(rows=len(selected_features), cols=len(selected_features))
    
    # Define color scheme for all classes
    if dataset_name == 'Bridge_Post_Earthquake_Damage_State':
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
        height=1100,
        autosize=True,
        showlegend=True,
        dragmode='select',
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
    [Output('classification-model-params', 'data'),
    Output('current-model-params', 'data')],
    State('classifier-dropdown', 'value'),
    Input('current-model-params', 'data'),
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
def create_classifier(classifier_name, current_params, svm_kernel, svm_C_power, svm_C_coef, svm_degree, 
                      svm_gamma_power, svm_gamma_coef, dt_max_depth, rf_max_depth, 
                      dt_min_samples_split, rf_min_samples_split, dt_min_samples_leaf, 
                      rf_min_samples_leaf, n_estimators, n_neighbors, weights, algorithm):
    if classifier_name == 'svm':
        model = SVC(
            kernel=svm_kernel,
            C=svm_C_coef * 10 ** svm_C_power,
            degree=svm_degree,
            gamma=svm_gamma_coef * 10 ** svm_gamma_power,
            probability=True,
        )
        model_params = model.get_params()  # Get model parameters
    elif classifier_name == 'dt':
        model = DecisionTreeClassifier(
            max_depth=dt_max_depth,
            min_samples_split=dt_min_samples_split,
            min_samples_leaf=dt_min_samples_leaf
        )
        model_params = model.get_params()  # Get model parameters
    elif classifier_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf
        )
        model_params = model.get_params()  # Get model parameters
    elif classifier_name == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
        model_params = model.get_params()  # Get model parameters
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    model_params['name'] = classifier_name
    if current_params == model_params:
        raise PreventUpdate
    else:
        return model_params, model_params
    
    
def create_prediction_plot(classifier, X, y, X_train, X_test, y_train, y_test, x_axis_name, y_axis_name, selected_features, target_names):
    # Create meshgrid for decision boundary, x_axis_name and y_axis_name are the selected features
    index_x = selected_features.index(x_axis_name)
    index_y = selected_features.index(y_axis_name)
    x_min, x_max = X[:, index_x].min() - 1, X[:, index_x].max() + 1
    y_min, y_max = X[:, index_y].min() - 1, X[:, index_y].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    pred_point_list = [np.full(xx.ravel().shape, np.mean(X[:, i])) if i not in [index_x, index_y] 
                       else xx.ravel() if i == index_x else yy.ravel() 
                       for i in range(len(selected_features))]
   
    Z = classifier.predict(np.c_[pred_point_list].T)
    Z = Z.reshape(xx.shape)
    proba = classifier.predict_proba(np.c_[pred_point_list].T)
    proba = proba.reshape(xx.shape + (len(target_names),))
    
    proba_test = classifier.predict_proba(X_test)

    proba_test_names = np.array([target_names[idx] for idx in classifier.predict(X_test)])
    test_real_names = np.array([target_names[idx] for idx in y_test])
    
    # Get the index of the class with the highest probability for each point
    max_proba_class = np.argmax(proba, axis=-1)
    # Get the highest probability for each point
    max_proba_value = np.max(proba, axis=-1)
    max_proba_class_names = np.array([target_names[idx] for idx in max_proba_class.flatten()]).reshape(max_proba_class.shape)

    # Create color scale for each class
    # colors = px.colors.qualitative.Set1[:len(target_names)]

    color_scale = px.colors.get_colorscale('Viridis')
    hex_colors = [c[1] for c in color_scale]
    
    color_scale = []
    for i, colour in enumerate(hex_colors):
        color_scale.extend([(0.1, colour), (0.9, colour)])
    
    # 创建预测图
    predict_fig = go.Figure()

    # 添加概率热图
    predict_fig.add_trace(
        go.Heatmap(
            x=xx[0],
            y=yy[:, 0],
            z=max_proba_class+max_proba_value,
            customdata= np.dstack((proba, max_proba_class_names)),
            colorscale=color_scale,
            opacity=0.7,
            showscale=False,
            hovertemplate=(
                f'{x_axis_name}: %{{x:.2f}} <br>'
                f'{y_axis_name}: %{{y:.2f}} <br>'
                f'{target_names[0]}:' + '%{customdata[0]:.2f}<br>'
                f'{target_names[1]}:' + '%{customdata[1]:.2f}<br>'
                f'{target_names[2]}:' + '%{customdata[2]:.2f}<br>'
                '最可能类别: %{customdata[3]}<br>'
            )
        )
    )
    
    # TODO: Add decision boundary or Contour plot

    # 添加训练数据和测试数据散点图
    markers = ['circle', 'square', 'diamond', 'cross', 'x']
    for i, target in enumerate(target_names):
        train_mask = y_train == i
        predict_fig.add_trace(
            go.Scatter(
                x=X_train[train_mask, index_x],
                y=X_train[train_mask, index_y],
                mode='markers',
                marker=dict(size=10, symbol=markers[i], line=dict(width=2, color='DarkSlateGrey')),
                name=f'{target} (训练)',
                legendgroup='train',
                showlegend=True,
                marker_color='rgba(255, 255, 255, 0)',
                marker_line_color='rgb(0, 0, 0)',
                marker_line_width=2,
                customdata=np.array([target] * train_mask.sum()),
                hovertemplate=(
                    f'{x_axis_name}: %{{x:.2f}} <br>'
                    f'{y_axis_name}: %{{y:.2f}} <br>'
                    '训练点类别: %{customdata}<br>'
                )
            )
        )
    
    for i, target in enumerate(target_names):
        test_mask = y_test == i
        predict_fig.add_trace(
            go.Scatter(
                x=X_test[test_mask, index_x],
                y=X_test[test_mask, index_y],
                mode='markers',
                marker=dict(size=10, symbol=markers[i], line=dict(width=2, color='DarkSlateGrey')),
                name=f'{target} (测试)',
                legendgroup='test',
                showlegend=True,
                customdata=np.hstack((proba_test[test_mask], proba_test_names[test_mask].reshape(-1, 1),test_real_names[test_mask].reshape(-1,1))),  # 概率+类别名称
                hovertemplate=(
                    f'{x_axis_name}: %{{x:.2f}} <br>'
                    f'{y_axis_name}: %{{y:.2f}} <br>'
                    f'{target_names[0]}: %{{customdata[0]:.2f}}<br>'
                    f'{target_names[1]}: %{{customdata[1]:.2f}}<br>'
                    f'{target_names[2]}: %{{customdata[2]:.2f}}<br>'
                    '预测类别: %{customdata[3]}<br>'
                    '实际类别: %{customdata[4]}<br>'
                )
            )
        )

    predict_fig.update_layout(
        xaxis_title=x_axis_name,
        yaxis_title=y_axis_name,
        font_size=18,
        legend_title='类别',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="left",
            x=0.0
        )
    )
    return predict_fig

def create_cm_fig(confusion_matrix, target_names):
    fig_cm = px.imshow(confusion_matrix, labels=dict(x="Predicted", y="Actual"), x=target_names, y=target_names)
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
                    options=[{'label': '鸢尾花(Iris)数据集', 'value': 'Iris'},
                            {'label': 'RC墩柱失效模式数据集', 'value': 'RC_Pier_Column_Failure_Mode'},
                            {'label': '桥梁震后损伤状态数据集', 'value': 'Bridge_Post_Earthquake_Damage_State'}],
                    value='Iris'
                ),
                html.H4("特征选择"),
                html.Div([
                    dbc.Button("全选", id="select-all-btn", n_clicks=0),
                    dbc.Button("全不选", id="deselect-all-btn", n_clicks=0,),
                    dbc.Button("反选", id="toggle-select-btn", n_clicks=0,),
                ]),
                dcc.Checklist(
                    id='feature-checklist',
                    options=[],  # Options will be added dynamically
                    value=[],
                    inline=False
                ),
                # html.Div(id='feature-selector-container'),  # Move feature selector container here
            ]),
            dcc.Tab(label='分类器', children=[
                html.H3('选择分类器'),
                dcc.Dropdown(
                    id='classifier-dropdown',
                    options=[{'label': '支持向量机(SVM)', 'value': 'svm'},
                             {'label': '决策树(Decision Tree)', 'value': 'dt'},
                            {'label': '随机森林(Random Forest)', 'value': 'rf'},
                            {'label': 'K-近邻(K-Nearest Neighbors)', 'value': 'knn'}],
                    value='svm'
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
                        # SVM 参数选择
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
                            step = 1,
                        ),
                        drc.NamedSlider(
                            name="Min Samples Split",
                            id="dt-min-samples-split-slider",
                            min=2,
                            max=20,
                            value=2,
                            marks={i: str(i) for i in range(2, 21)},
                            step = 1,
                        ),
                        drc.NamedSlider(
                            name="Min Samples Leaf",
                            id="dt-min-samples-leaf-slider",
                            min=1,
                            max=20,
                            value=1,
                            marks={i: str(i) for i in range(1, 21)},
                            step = 1,
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
                            step=10,
                        ),
                        drc.NamedSlider(
                            name="Max Depth",
                            id="rf-max-depth-slider",
                            min=1,
                            max=20,
                            value=5,
                            marks={i: str(i) for i in range(1, 21)},
                            step = 1,
                        ),
                        drc.NamedSlider(
                            name="Min Samples Split",
                            id="rf-min-samples-split-slider",
                            min=2,
                            max=20,
                            value=2,
                            marks={i: str(i) for i in range(2, 21)},
                            step=1,
                        ),
                        drc.NamedSlider(
                            name="Min Samples Leaf",
                            id="rf-min-samples-leaf-slider",
                            min=1,
                            max=20,
                            value=1,
                            marks={i: str(i) for i in range(1, 21)},
                            step=1,
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
                            step=1,
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
                dcc.Graph(id='prediction-plot', style={'height': '50vh'}),
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
    dcc.Store(id='current-model-params'),
    dcc.Store(id='current-dataset')
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
    [Output('prediction-plot', 'figure'),
     Output('roc-curve', 'figure'),
     Output('confusion-matrix', 'figure')],
    [Input('classifier-dropdown', 'value'),
     Input('current-model-params', 'data'),
     Input('classification-model-params', 'data'),
     Input('dataset-dropdown', 'value'),
     Input('feature-checklist', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')
    ]
)
def train_model_and_update_figure(model_name, current_model, model_params, dataset_name, selected_features,x_axis_name,y_axis_name):
    if model_name != current_model['name']:
        return go.Figure(), go.Figure(),go.Figure()
    if len(selected_features)<2:
        return go.Figure(),go.Figure(),go.Figure()
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
        
    model_params.pop('name', None)
    classifier = models[model_name](**model_params)
    classifier.fit(X_train, y_train)
    
    predict_fig =  create_prediction_plot(classifier, X, y, X_train, X_test, y_train, y_test, x_axis_name, y_axis_name, selected_features, target_names)
    
    roc_fig = roc_curve_fig(classifier, X_test, y_test, target_names)
    
    cm_fig = create_cm_fig(confusion_matrix(y_test, classifier.predict(X_test)), target_names)
    
    return predict_fig, roc_fig, cm_fig

if __name__ == '__main__':
    app.run_server(debug=True)