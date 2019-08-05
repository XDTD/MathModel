import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('Data/timeSerial.csv',encoding='ANSI')
data.plot()
plt.legend()
plt.savefig('result/时间序列/时序图.jpg')
plt.show()
# 绘制自相关图
plot_acf(data).show()
plot_acf(data).savefig('result/时间序列/自相关图.jpg')

