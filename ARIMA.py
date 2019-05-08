import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.api import tsa

def generate_data(start_date,end_date):
    df=pd.DataFrame([300+i*30+randrange(50)for i in range(31)],columns=['income'],
                    index=pd.date_range(start_date,end_date,freq='D'))
    return df
data=generate_data('20170601','20170701')
data['income']=data['income'].astype('float64')

data.plot()
plt.show()
plot_acf(data).show()

data_diff=data.diff()
data_diff=data_diff.dropna()
data_diff.plot()
plt.show()

plot_acf(data_diff).show()
plot_pacf(data_diff).show()

arima=ARIMA(data,order=(1,1,2))
result=arima.fit(disp=False)
print(result.aic,result.bic,result.hqic)
plt.plot(data_diff)
plt.plot(result.fittedvalues,color='red')
plt.title('ARIMA RSS:%.4f'%sum(result.fittedvalues-data_diff['income'])**2)
plt.show()

pred=result.predict('20170701','20170710',typ='levels')
print(pred)
x=pd.date_range('20170601','20170705')
plt.plot(x[:31],data['income'])
plt.plot(pred)
plt.show()
print('end')