import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# __locations = None
# __State = None
# __District = None
# __Variety = None
# __Month = None

__model = None
model=None

def get_estimated_Quality(DO,BOD,totalCaliform,fecalCaliform):
    # c_District= __District.index(District.lower())
    # c_Variety=340+__Variety.index(Variety.lower())
    # c_Month=361+__Month.index(Month.lower())
    # c_State=369+__State.index(State.lower())
    # if(isProducing==0):
    #     c_isProducing=397
    # else:
    #     c_isProducing=398  
    # if(Harvesting_month==0):
    #     c_Harvesting_month=399
    # else:
    #     c_Harvesting_month=400
    # if(isDrought==0):
    #     c_isDrought=401
    # else:
    #     c_isDrought=402          
    # # c_isProducing=397+np.where(df6.columns==isProducing)[0][0]
    # # c_Harvesting_month=399+np.where(df7.columns==Harvesting_month)[0][0]
    # # c_isDrought=401+np.where(df8.columns==isDrought)[0][0]
    
    # pridictData=np.zeros(403)
    # pridictData[c_District]=1
    # pridictData[c_Variety]=1
    # pridictData[c_Month]=1
    # pridictData[c_State]=1
    # pridictData[c_isProducing]=1
    # pridictData[c_Harvesting_month]=1
    # pridictData[c_isDrought]=1
    pridictData=[DO,BOD,totalCaliform,fecalCaliform]
    with open('model.pkl', 'rb') as file:
     model = pickle.load(file)
       
    df = pd.read_csv('Months (4).csv')
    encoder = LabelEncoder()
    encoder.fit(df['E'])
    encoded_data = encoder.transform(df['E'])
    df['E'] = encoded_data
    df = df.dropna()
    X = df.iloc[:,:-1].values 
    y = df.iloc[:,-1].values
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)    
   
   
   
    y_pred = model.predict(sc.transform([pridictData])) 
    if y_pred == [0]:
      return "B"
    elif y_pred == [1]:
      return "C"
    elif y_pred == [3]:
      return "D"
    elif y_pred == [5]:
      return "E"
    else :
      return "A"
  

# def get_estimated_price(location,sqft,bhk,bath):
#     try:
#         loc_index = __data_columns.index(location.lower())
#     except:
#         loc_index = -1

#     x = np.zeros(len(__data_columns))
#     x[0] = sqft
#     x[1] = bath
#     x[2] = bhk
#     if loc_index>=0:
#         x[loc_index] = 1

#     return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    # global  __data_columns

    # with open("columns.json", "r") as f:
    #     # __data_columns = data['data_columns']
    #     data=json.load(f)
    #     __District = data['district'] 
    #     __State = data['States'] 
    #     __Month = data['Month'] 
    #     __Variety = data['variety'] 

        
        
       

    global model
    if model is None:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(model.predict([[10.42,1.6,920,540]])[0])
    print(get_estimated_Quality(10.0,23.0,700.0,1000.0))
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    # print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    # print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location