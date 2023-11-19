import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from keras.callbacks import ModelCheckpoint 
from tensorflow.keras.models import load_model 

from keras.models import Sequential 
from keras.layers import LSTM 
from keras.layers import Dropout 
from keras.layers import Dense 

#kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_absolute_percentage_error 

#đọc data
arrDf = []

file_paths = ["dataset/FPT.csv" , "dataset/MSN.csv" , "dataset/PNJ.csv" , "dataset/VIC.csv"]
for path in file_paths:
  df = pd.read_csv(path)
  arrDf.append(df)


#hàm xử lý số liệu
def preprocess_data(df , index):
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%m/%d/%Y %H:%M")
    df["Ticker"] = index                                                              #chuyển nhán công ty thành một số
    df.dropna(inplace=True)
    df.sort_values(by='Date/Time',inplace = True)
    df['Next_Date/Time'] = df['Date/Time'].shift(-1)
    df["During"] = df['Next_Date/Time'] - df['Date/Time']                              # tình thời gian tương lai cách hiện tại là bao nhiêu
    df.dropna(inplace=True)
    df["During_minutes"] = df["During"].dt.total_seconds() / 60
    df["During_minutes"] = df["During_minutes"].astype(int)
    df = df.drop(columns=["Date/Time" , "During" , "Open Interest" , "Next_Date/Time"])

    return df

arrDfProcess = []
for i , df in enumerate(arrDf):
  result = preprocess_data(df,i)
  arrDfProcess.append(result)

df_Fpt = arrDfProcess[3]
print(df_Fpt.head(10) , "\n")
print(df_Fpt.info())


def built_Input(df,ratio , sc , time_step):
    x,y=[],[]
    data = df.values
    data = sc.transform(data)
    for i in range(time_step,len(data)):
        x.append(data[i-time_step:i])
        y.append(data[i,1:5])

    threshold = int(len(data) * ratio)
    x_train = np.array(x[:threshold])
    y_train = np.array(y[:threshold])
    x_test = np.array(x[threshold:])
    y_test =  np.array(y[threshold:])

    return x_train,y_train,x_test,y_test

Ratio = 0.7
TimeStep = 50
list_x_train = []
list_y_train = []
list_x_test = []
list_y_test = []
list_data = []

#chuẩn hóa dữ liệu
sc = MinMaxScaler(feature_range=(0,1))

for df in arrDfProcess:
    data = df.values
    list_data.append(data)

total_data = np.concatenate(list_data)
sc.fit(total_data)

for df in arrDfProcess:
    x_train,y_train,x_test,y_test = built_Input(df , Ratio , sc , TimeStep)
    list_x_train.append(x_train)
    list_y_train.append(y_train)
    list_x_test.append(x_test)
    list_y_test.append(y_test)


total_x_train = np.concatenate(list_x_train, axis=0)
total_y_train = np.concatenate(list_y_train, axis=0)
print(total_x_train.shape)
print(total_y_train.shape)


total_x_test = np.concatenate(list_x_test, axis=0)
total_y_test = np.concatenate(list_y_test, axis=0)

model = Sequential()
model.add(LSTM(units = 132 , input_shape = total_x_train[1].shape , return_sequences=True))
model.add(LSTM(units = 64))
model.add(Dropout(0.5))
model.add(Dense(28 , activation = "relu"))
model.add(Dense(total_y_train.shape[1]))
model.compile(loss='mean_absolute_error',optimizer='adam')
model.summary()

save_model = "save_model.hdf5"
best_model = ModelCheckpoint(save_model,monitor='loss',verbose=2,save_best_only=True,mode='auto')
model.fit(total_x_train,total_y_train,epochs=10,batch_size=50,verbose=2,callbacks=[best_model])

final_model = load_model("Downloads/save_model.hdf5")

#dự đoán
def convertY_lableArr(total_y_test):
  total_y_test1 = np.hstack((np.zeros((total_y_test.shape[0], 1)), total_y_test))
  total_y_test2 = np.hstack((total_y_test1, np.zeros((total_y_test.shape[0], 2))))
  total_y_test_real = sc.inverse_transform(total_y_test2)
  return total_y_test_real[:, 1:-2]

y_real = convertY_lableArr(total_y_test)
y__predict = final_model.predict(total_x_test)
y_predict = convertY_lableArr(y__predict)

print('Độ phù hợp tập :',r2_score(y_real,y_predict))
print('Sai số tuyệt đối trung bình:',mean_absolute_error(y_real,y_predict))
print('Phần trăm sai số tuyệt đối trung bình:',mean_absolute_percentage_error(y_real,y_predict))
