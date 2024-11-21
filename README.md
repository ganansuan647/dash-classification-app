# Classification Explorer

本项目为一个采用Dash编写的轻量级可视化APP，用于探索不同分类器在各种参数和不同数据集上的性能。该应用程序允许用户选择数据集、选择特征并配置分类器参数，以可视化分类结果。

## 界面概览
![image](https://github.com/user-attachments/assets/7c71812a-06a0-419e-866b-83f70882154a)

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
    # 使用pyinstaller打包比较简单，但文件会略大
    pip install pyinstaller
    pyinstaller --onefile classification_app.py
    ```
    **注意**：pyinstaller打包时对于用到的文件以及一些特定的包可能会有问题，需要对打包配置文件([classification_app.spec](/classification_app.spec))进行修改，修改完可以利用如下命令重新打包：
    ```bash
    pyinstaller classification_app.spec
    ```
    更详细的spec配置文件修改方法请参考[pyinstaller官方文档](https://pyinstaller.readthedocs.io/en/stable/usage.html)。

    也可以有其他的打包方式，如使用nuitka：
    ```bash
    # 使用nuitka打包，文件会小一些
    pip install nuitka
    nuitka --mingw64 --show-progress --plugin-enable=upx --standalone --onefile classification_app.py
    ```
    **注意**：nuitka打包可能需要下载[mingw64](https://winlibs.com/),并将mingw64/bin目录添加至环境变量 (*注意这里不要有中文*)
6. 自行修改后更新uv依赖（可选）:
    ```bash
    # 添加你需要的包
    uv add <某个新模块>

    # 更新依赖，生成requirements.txt
    uv pip compile pyproject.toml -o requirements.txt
    ```

## 作者

- [**苟凌云**](https://github.com/ganansuan647)
- **邮箱**: 2410100@tongji.edu.cn
- **小组成员**: 苟凌云、易航、齐新宇、何帅君、李鸿豪
