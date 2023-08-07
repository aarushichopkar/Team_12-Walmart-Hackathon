# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score

Data =pd.read_csv("C:/Users/aarus/Downloads/Walmart/train.csv")
Test_Data =pd.read_csv("C:/Users/aarus/Downloads/Walmart/test.csv")
Click_log =pd.read_csv("C:/Users/aarus/Downloads/archive/click_log.csv")
Data=Data[Data.isnull().sum(axis=1)<=17]

# Check the percetage of missing values in columns
null_val = Data.isnull().sum()
null_per = (Data.isnull().sum()/Data.shape[0])*100
dic ={'No. of Missing Values':null_val,'Percentage of Missing Values':null_per}
df = pd.DataFrame(dic,columns=['No. of Missing Values','Percentage of Missing Values'])
df.sort_values(by='Percentage of Missing Values',ascending=False)

record_id = Test_Data['record_id']

Data['Type'] = 1
Test_Data['Type'] = 0

Full_Data=pd.concat([Data,Test_Data],axis=0)

remove_col_lst=df[df['Percentage of Missing Values']>=50.0].index
Full_Data.drop(remove_col_lst,axis=1,inplace=True)

Full_Data = pd.merge(Full_Data, Click_log[["imprId", "clickIp"]], left_on="imprid_cr", right_on="imprId", how="left")
Full_Data['clickIp'] = Full_Data['clickIp'].astype(str)

not_relevant_col=["v_cr","record_id","templateid_cr","geodimid_cr","goalTypeId_cr"]
Full_Data.drop(not_relevant_col,axis=1,inplace=True)

Full_Data['conversion_fraud'].replace((True,False),(1,0),inplace=True)
Full_Data.drop(Full_Data.select_dtypes('object').columns,axis=1,inplace=True)

# Fill the values of data

Full_Data['cityGrpDimId_cr'].fillna(Full_Data['cityGrpDimId_cr'].mode()[0],inplace=True)

Full_Data['stateGrpDimId_cr'].fillna(Full_Data['stateGrpDimId_cr'].mode()[0],inplace=True)

Full_Data['clickTimeInMillis_cr'].fillna(Full_Data['clickTimeInMillis_cr'].mode()[0],inplace=True)

Full_Data['clickbid_cr'].fillna(Full_Data['clickbid_cr'].mode()[0],inplace=True)

# Stor column names
cloumn_names=Full_Data.columns

scaled_Full_Data = minmax_scale(Full_Data, feature_range=(0,1))
scaled_Full_Data=pd.DataFrame(scaled_Full_Data,columns=cloumn_names)
Data_Modified= scaled_Full_Data[scaled_Full_Data['Type']==1]
Test_Modified= scaled_Full_Data[scaled_Full_Data['Type']==0]
X = Data_Modified.drop(['conversion_fraud'],axis=1)
Y = Data_Modified['conversion_fraud']

from imblearn.over_sampling import SMOTE
x_res, y_res = SMOTE().fit_resample(X, Y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_res,y_res, test_size=0.20, random_state=0,shuffle= True, stratify=y_res)

from sklearn.ensemble import GradientBoostingClassifier

# Create  ML Model and fit the training data
GBC = GradientBoostingClassifier(learning_rate=0.2, max_depth=4, n_estimators=200, random_state=25)
GBC.fit(x_train, y_train)

# Predict Output and Store it 
y_pred_GBC= GBC.predict(x_test)

from sklearn.metrics import accuracy_score
print('Accuracy : %s '%'{0:.2%}'.format(accuracy_score(y_test, y_pred_GBC)))

result= GBC.predict(Test_Modified.drop(['conversion_fraud'],axis=1))
result = pd.DataFrame(result, columns=['conversion_fraud'])
result['conversion_fraud'].replace((1,0),(True,False),inplace=True)
# result.set_index(record_id, inplace=True)
Test_Modified = pd.merge(Test_Data, Click_log[["imprId", "clickIp"]], left_on="imprid_cr", right_on="imprId", how="left")
Test_Modified['clickIp'] = Test_Modified['clickIp'].astype(str)
result.set_index(Test_Modified['clickIp'], inplace=True)
result.to_csv(r"C:/Users/aarus/Downloads/Walmart/Result.csv")


num_unique_clicks = Test_Modified['clickIp'].count()
num_unique_impressions = Test_Modified['imprid_cr'].nunique()

result_copy = result.copy()

# Convert the column to integers
result["conversion_fraud"] = result["conversion_fraud"].astype(int)

# Calculate number of conversions and total spend using the copied DataFrame
num_conversions =  (result_copy["conversion_fraud"] == 0).sum()
total_spend = Test_Data['spend_cr'].sum()

# Calculate click-through rate (CTR)
ctr = round(num_unique_clicks / num_unique_impressions,2)

# Calculate conversion rate
conversion_rate = round(num_conversions / num_unique_clicks,2)

# Calculate cost per click (CPC)
cpc = round(total_spend / num_unique_clicks,2)

# Calculate cost per mille (CPM)
cpm = round((total_spend / num_unique_impressions) * 1000,2)

# Print the calculated metrics
print(f'Number of Unique Clicks: {num_unique_clicks}')
print(f'Number of Unique Impressions: {num_unique_impressions}')
print(f'Number of Conversions: {num_conversions}')
print(f'Total Spend: {total_spend}')
print(f'Click-Through Rate (CTR): {ctr}')
print(f'Conversion Rate: {conversion_rate}')
print(f'Cost Per Click (CPC): {cpc:.2f}')
print(f'Cost Per Mille (CPM): {cpm:.2f}')

metrics = {
    'tt' : num_unique_impressions,
    'ctr': ctr,
    'valid_conversions':num_conversions,
    'conversion_rate': conversion_rate,
    'cpc': cpc,
    'cpm': cpm
}
flagged_ips = Test_Modified['clickIp'].unique().tolist()



# Save metrics and flagged IPs to a pickle file
import pickle
with open('metrics_flagged_ips.pkl', 'wb') as f:
    pickle.dump({'metrics': metrics, 'flagged_ips': flagged_ips}, f)

# # Save the trained model
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(GBC, model_file)

# # Save the preprocessor
# with open('preprocessor.pkl', 'wb') as preprocessor_file:
#     pickle.dump(preprocessor, preprocessor_file)

def process_uploaded_file(file_path):
    # Load the trained model
    with open('trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the preprocessor
    with open('trained_model.pkl', 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

        # Load the uploaded CSV file into a DataFrame
        df_test = pd.read_csv(file_path)

        # Preprocess the test data
        X_test = preprocessor.transform(df_test)

        # Make predictions using the trained model
        predictions = model.predict(X_test)

        # Add predictions to the DataFrame
        df_test['conversion_fraud'] = predictions

        return df_test



# def preprocessor():
#     Data=Data[Data.isnull().sum(axis=1)<=17]
#     # Check the percetage of missing values in columns
#     null_val = Data.isnull().sum()
#     null_per = (Data.isnull().sum()/Data.shape[0])*100
#     dic ={'No. of Missing Values':null_val,'Percentage of Missing Values':null_per}
#     df = pd.DataFrame(dic,columns=['No. of Missing Values','Percentage of Missing Values'])
#     df.sort_values(by='Percentage of Missing Values',ascending=False)

#     # record_id = Test_Data['record_id']
#     remove_col_lst=df[df['Percentage of Missing Values']>=50.0].index
#     Data.drop(remove_col_lst,axis=1,inplace=True)

#     Data = pd.merge(Data, Click_log[["imprId", "clickIp"]], left_on="imprid_cr", right_on="imprId", how="left")
#     Data['clickIp'] = Data['clickIp'].astype(str)

#     not_relevant_col=["v_cr","record_id","templateid_cr","geodimid_cr","goalTypeId_cr"]
#     Data.drop(not_relevant_col,axis=1,inplace=True)

#     Data['conversion_fraud'].replace((True,False),(1,0),inplace=True)
#     Data.drop(Data.select_dtypes('object').columns,axis=1,inplace=True)

#     # Fill the values of data

#     Data['cityGrpDimId_cr'].fillna(Data['cityGrpDimId_cr'].mode()[0],inplace=True)
#     Data['stateGrpDimId_cr'].fillna(Data['stateGrpDimId_cr'].mode()[0],inplace=True)
#     Data['clickTimeInMillis_cr'].fillna(Data['clickTimeInMillis_cr'].mode()[0],inplace=True)
#     Data['clickbid_cr'].fillna(Data['clickbid_cr'].mode()[0],inplace=True)

#     # Stor column names
#     cloumn_names=Data.columns

#     Data = minmax_scale(Full_Data, feature_range=(0,1))
#     Data=pd.DataFrame(scaled_Full_Data,columns=cloumn_names)
#     return Data

# def pred(Test_Modified):
#     result= GBC.predict(Test_Modified.drop(['conversion_fraud'],axis=1))
#     result = pd.DataFrame(result, columns=['conversion_fraud'])
#     result['conversion_fraud'].replace((1,0),(True,False),inplace=True)
#     # result.set_index(record_id, inplace=True)
#     Test_Modified = pd.merge(Test_Data, Click_log[["imprId", "clickIp"]], left_on="imprid_cr", right_on="imprId", how="left")
#     Test_Modified['clickIp'] = Test_Modified['clickIp'].astype(str)
#     result.set_index(Test_Modified['clickIp'], inplace=True)
#     result.to_csv(r"C:/Users/aarus/Downloads/Walmart/Result.csv")
