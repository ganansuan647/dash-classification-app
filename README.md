# Classification Explorer

本项目为一个采用Dash编写的轻量级可视化APP，用于探索不同分类器在各种参数和不同数据集上的性能。该应用程序允许用户选择数据集、选择特征并配置分类器参数，以可视化分类结果。

## 功能

- **数据集选择**: 从多个数据集中选择，包括Iris、RC墩柱破坏模式和桥梁震后损伤状态。
- **特征选择**: 选择用于分类的特征。
- **分类器选择**: 从各种分类器中选择，如SVM、决策树、随机森林和K近邻。
- **参数调整**: 调整所选分类器的参数。
- **可视化**: 可视化数据分布、分类置信度、ROC曲线和混淆矩阵。

## 使用方法

1. **选择数据集**: 使用下拉菜单选择一个数据集。
2. **选择特征**: 从特征检查列表中选择特征。可以全选、取消全选或切换选择。
3. **选择分类器**: 从下拉菜单中选择一个分类器，目前支持SVM，DT，RF和KNN。
4. **配置参数**: 调整所选分类器的参数。
5. **选择绘图轴**: 选择用于绘图的X轴和Y轴特征。
6. **查看结果**: 应用程序将显示数据分布、分类置信度图、ROC曲线和混淆矩阵。

## 安装和运行

1. 克隆仓库:
    ```bash
    git clone https://github.com/ganansuan647/dash-classification-app.git
    ```
2. 进入项目目录:
    ```bash
    cd dash-classification-app
    ```
3. 安装所需依赖:
    可以简单的使用pip命令安装所需依赖:
    ```bash
    pip install -r requirements.txt
    ```

    也可以使用uv进行安装
    ```bash
    pip install uv
    uv venv
    ```

    使用以下命令激活venv
    ```bash
    # On macOS and Linux.
    source .venv/bin/activate
    
    # On Windows.
    .venv\Scripts\activate
    ```

    待虚拟环境激活后，再安装依赖
    ```bash
    uv pip sync requirements.txt
    ```
4. 运行应用程序:
    ```bash
    python classification_app.py
    ```
5. 打包为单一exe文件（可选）:
    ```bash
    pip install pyinstaller
    pyinstaller --onefile classification_app.py
    ```
6. 自行修改后更新uv依赖（可选）:
    ```bash
    # 添加你需要的包
    uv add <某个新模块>

    # 更新依赖，生成requirements.txt
    uv pip compile pyproject.toml -o requirements.txt
    ```


## 作者

- **苟凌云**
- **邮箱**: 2410100@tongji.edu.cn
- **小组成员**: 苟凌云、易航、齐新宇、何帅君、李鸿豪

## 许可证

此项目基于MIT许可证 - 详见[LICENSE](LICENSE)文件。