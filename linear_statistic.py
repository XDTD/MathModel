import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox_normmax
from scipy import special
from sklearn.decomposition import PCA


def Q_6_7():
    data = pd.read_csv('Data/LinearStatistic.csv')
    print('原始数据:\n',data)

    # 数据预处理
    data_list = data.to_dict()
    for i in range(0,len(data_list['x1'])):
        data_list['x1'][i] = data_list['x1'][i]*data_list['x1'][i]
    data = pd.DataFrame(data_list)
    # 数据描述
    print('数据描述:\n',data.describe())
    # 缺失值检验
    print('缺失值检验:\n',data[data.isnull() == True].count())

    data.boxplot()
    plt.savefig("result/线性统计/6.7/boxplot_linear.jpg")
    plt.title('数据特征分析')
    plt.show()
    # 相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
    # 相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
    print('相关系数: ',data.corr())


    # 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带。
    sns.pairplot(data, x_vars=['x1','x2'], y_vars='y', height=7, aspect=0.8,kind = 'reg')
    plt.savefig("result/线性统计/6.7/pairplot_linear.jpg")
    plt.title('不同因素影响图')
    plt.show()

    X_train,X_test,Y_train,Y_test = train_test_split(data.ix[:,:2],data.ix[:,2:3],train_size=.80)

    print("原始数据特征:",data.ix[:,:2].shape,
          ",训练数据特征:",X_train.shape,
          ",测试数据特征:",X_test.shape)

    print("原始数据标签:",data.ix[:,2:3].shape,
          ",训练数据标签:",Y_train.shape,
          ",测试数据标签:",Y_test.shape)

    model = LinearRegression()
    model.fit(X_train,Y_train)
    a = model.intercept_  # 截距
    b = model.coef_  # 回归系数
    print("最佳拟合线:截距", a, ",回归系数：",b)

    # R方检测
    # 决定系数r平方
    # 对于评估模型的精确度
    # y误差平方和 = Σ(y实际值 - y预测值)^2
    # y的总波动 = Σ(y实际值 - y平均值)^2
    # 有多少百分比的y波动没有被回归拟合线所描述 = SSE/总波动
    # 有多少百分比的y波动被回归线描述 = 1 - SSE/总波动 = 决定系数R平方
    # 对于决定系数R平方来说1） 回归线拟合程度：有多少百分比的y波动刻印有回归线来描述(x的波动变化)
    # 2）值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
    score = model.score(X_test,Y_test)
    print('相关系数',score)

    # 对线性回归进行预测
    Y_pred = model.predict(X_test)
    print(Y_pred)


    # 显示图像
    plt.figure()
    plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
    plt.plot(range(len(Y_pred)),Y_test,'r',label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("Y")
    plt.ylabel('the number of Y')
    plt.title('预测与源数据对比图')
    plt.savefig("result/线性统计/6.7/compare_linear.jpg")
    plt.show()


    # 残差预测值
    #  enumerate 函数可以把一个 list 变成索引-元素对
    y_dif = []
    for i in range(len(Y_pred)):
        y_dif.append(Y_pred[i,0]-Y_test['y'].values[i])
    tmp = {'x':range(len(y_dif)),'y':y_dif}
    df = pd.DataFrame(tmp)
    sns.residplot(x="x", y="y",data=df)
    plt.savefig("result/线性统计/6.7/残差图1.jpg")
    plt.title('残差图1')
    plt.show()

    # box-cox变换
    Y = data['y'].tolist()
    lam_best = boxcox_normmax(Y)
    print('lambda: ',lam_best)
    Y = special.boxcox1p(Y, lam_best)

    # box-cox变换后的回归
    for i in range(len(Y)):
        data_list['y'][i] = Y[i]
    data = pd.DataFrame(data_list)


    data.boxplot()
    plt.savefig("result/线性统计/6.7/boxplot_linear_bxo_cox.jpg")
    plt.title('bxo-cox后的数据特征分析')
    plt.show()
    # 相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
    # 相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
    print('相关系数: ',data.corr())


    # 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带。
    sns.pairplot(data, x_vars=['x1','x2'], y_vars='y', height=7, aspect=0.8,kind = 'reg')
    plt.savefig("result/线性统计/6.7/pairplot_linear_bxo_cox.jpg")
    plt.title('bxo-cox后的不同因素影响图')
    plt.show()

    X_train,X_test,Y_train,Y_test = train_test_split(data.ix[:,:2],data.ix[:,2:3],train_size=.80)
    model = LinearRegression()
    model.fit(X_train,Y_train)
    a = model.intercept_  # 截距
    b = model.coef_  # 回归系数
    print("bxo-cox后的最佳拟合线:截距", a, ",回归系数：",b)

    # R方检测
    # 决定系数r平方
    # 对于评估模型的精确度
    # y误差平方和 = Σ(y实际值 - y预测值)^2
    # y的总波动 = Σ(y实际值 - y平均值)^2
    # 有多少百分比的y波动没有被回归拟合线所描述 = SSE/总波动
    # 有多少百分比的y波动被回归线描述 = 1 - SSE/总波动 = 决定系数R平方
    # 对于决定系数R平方来说1） 回归线拟合程度：有多少百分比的y波动刻印有回归线来描述(x的波动变化)
    # 2）值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
    score = model.score(X_test,Y_test)
    print('R方检测 ',score)

    # 对线性回归进行预测
    Y_pred = model.predict(X_test)
    print(Y_pred)

    # 显示图像
    plt.figure()
    plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
    plt.plot(range(len(Y_pred)),Y_test,'r',label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.ylabel("Y")
    plt.xlabel('the number of Y')
    plt.title('bxo-cox后的预测与源数据对比图')
    plt.savefig("result/线性统计/6.7/compare_linear_box_cox.jpg")
    plt.show()


    # 残差预测值
    #  enumerate 函数可以把一个 list 变成索引-元素对
    y_dif = []
    for i in range(len(Y_pred)):
        y_dif.append(Y_pred[i,0]-Y_test['y'].values[i])
    tmp = {'x':range(len(y_dif)),'y':y_dif}
    df = pd.DataFrame(tmp)
    sns.residplot(x="x", y="y",data=df)
    plt.savefig("result/线性统计/6.7/残差图_bxo_cox.jpg")
    plt.title('bxo-cox后的残差图')
    plt.show()


def Q_6_12():
    X = pd.read_csv('Data/Q_6_12.csv')
    data = X
    X.drop(columns='Y')
    Y = data['Y']
    pca = PCA(n_components=2)
    pca.fit(X)
    newX = pca.fit_transform(X)
    print('贡献率:\n',pca.explained_variance_ratio_)  # 输出贡献率
    print('主成分分析后的自变量：\n',newX)
    X_train,X_test,Y_train,Y_test = train_test_split(newX,Y,train_size=.80)
    model = LinearRegression()
    model.fit(X_train,Y_train)
    a = model.intercept_  # 截距
    b = model.coef_  # 回归系数
    print("最佳拟合线:截距", a, ",回归系数：",b)

    # R方检测
    # 决定系数r平方
    # 对于评估模型的精确度
    # y误差平方和 = Σ(y实际值 - y预测值)^2
    # y的总波动 = Σ(y实际值 - y平均值)^2
    # 有多少百分比的y波动没有被回归拟合线所描述 = SSE/总波动
    # 有多少百分比的y波动被回归线描述 = 1 - SSE/总波动 = 决定系数R平方
    # 对于决定系数R平方来说1） 回归线拟合程度：有多少百分比的y波动刻印有回归线来描述(x的波动变化)
    # 2）值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
    score = model.score(X_test,Y_test)
    print('R方检测: ',score)

    # 对线性回归进行预测
    Y_pred = model.predict(X_test)
    print(Y_pred)

    # 显示图像
    plt.figure()
    plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
    plt.plot(range(len(Y_pred)),Y_test,'r',label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of Y")
    plt.ylabel('Y')
    plt.title('预测与源数据对比图')
    plt.savefig("result/线性统计/6.12/compare_linear.jpg")
    plt.show()

    #  残差预测值
    #  enumerate 函数可以把一个 list 变成索引-元素对
    y_dif = []
    for i in range(len(Y_pred)):
        y_dif.append(Y_pred[i]-Y_test.values[i])
    tmp = {'x':range(len(y_dif)),'y':y_dif}
    df = pd.DataFrame(tmp)
    sns.residplot(x="x", y="y",data=df)
    plt.savefig("result/线性统计/6.12/残差图.jpg")
    plt.title('残差图')
    plt.show()


if __name__ == '__main__':
   # Q_6_7()
    Q_6_12()