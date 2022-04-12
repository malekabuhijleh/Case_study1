import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Reading Data
loan = pd.read_csv('loans_full_schema.csv')

#Data Visualization
sns.set(rc = {'figure.figsize':(12,6)})
sns.heatmap(loan.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
sns.distplot(loan['interest_rate'],bins=50,kde=False,color='red')
plt.show()
sns.color_palette("vlag", as_cmap=True)
sns.scatterplot(x= 'loan_amount', y ='interest_rate', data = loan , hue = 'term')
plt.show()
sns.set(rc = {'figure.figsize':(14,8)})
sns.countplot(x='loan_purpose',data=loan)
plt.show()
sns.distplot(loan['annual_income'],bins=50,kde=False,color='blue')
plt.show()

#To handle missing data in employement Length
def impute__emp(cols):
    emp_length = cols[0]
    if pd.isnull(emp_length):
        return np.random.randint(0,11)
    else:
        return emp_length
loan['emp_length'] = loan[['emp_length']].apply(impute__emp,axis=1)

#Remove non-numerical columns
loan.drop(['disbursement_method','sub_grade','grade','application_type','loan_purpose','verified_income','homeownership','state','num_accounts_120d_past_due','debt_to_income_joint', 'issue_month','loan_status', 'initial_listing_status','emp_title','annual_income_joint', 'verification_income_joint', 'months_since_last_delinq' , 'months_since_90d_late'], axis =1, inplace= True)
loan.dropna(inplace=True)
print(loan.columns)
print(loan.info())

sns.set(rc = {'figure.figsize':(10,4)})
X = loan.drop('interest_rate', axis = 1)
y = loan['interest_rate']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

rfc = RandomForestRegressor(n_estimators=1000)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)
plt.scatter(y_test,predictions)
plt.title('RF Regressor Predicted Vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print('MAE for the random forest regressor:', metrics.mean_absolute_error(y_test, predictions))
print('MSE for the random forest regressor:', metrics.mean_squared_error(y_test, predictions))
print('RMSE for the random forest regressor:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

svm_r = make_pipeline(StandardScaler(), SVR(C=20, epsilon=0.05))

svm_r.fit(X_train,y_train)
predictions_svm = svm_r.predict(X_test)
plt.scatter(y_test,predictions_svm)
plt.title('SVM Predicted Vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print('MAE for the Support vector machine regressor:', metrics.mean_absolute_error(y_test, predictions_svm))
print('MSE for the Support vector machine regressor:', metrics.mean_squared_error(y_test, predictions_svm))
print('RMSE for the Support vector machine regressor:', np.sqrt(metrics.mean_squared_error(y_test, predictions_svm)))

