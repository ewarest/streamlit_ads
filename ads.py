import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import xgboost as xgb
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

if 'initial_run' not in st.session_state:
    st.session_state.initial_run = 1

st.title('Interactive Hyperparameter tuning for a XGBoost model')

st.header('Ad click dataset')
st.write('The ad click dataset could be found on Kaggle and shows coversions (Purchased = [0,1]) for 400 customers. It also contains gender, age and the estimated salary of the customers. The goal is to predict if a customer will convert. \n\n First 5 lines:')

ads = load_data('Social_Network_Ads.csv')
st.write(ads.head())

ads['Gender_Flag']= ads['Gender'].map({'Male':0, 'Female':1})

st.header('Correlation Matrix')
st.write('Gender was converted to a 0/1 flag to consider it in the correlation matrix. The Correlation Matrix shows that age and the estimated salary have the highest potential as predictors for conversions (=Purchased).')
X, y = ads[['Gender_Flag', 'Age', 'EstimatedSalary']], ads['Purchased']
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.33, random_state=42)
corr = ads.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
fig1 = plt.figure(figsize=(10, 4))
sns.heatmap(corr, mask=mask, annot=True)
st.pyplot(fig1)

st.header('Model Training')
st.write('Find the best model by changing the hyperparameters, the initial run happens with default values from xgboost library.')

sel_col, disp_col, best_col = st.columns(3)

with sel_col:
    st.subheader('Parameters')
    st.write('Try different hyperparameters (model training is executed immediately after every change):')
    gamma = st.selectbox('Gamma', options=[0, 0.5, 1], index=0)
    max_depth = st.selectbox('Max Depth', options=[100,200,300], index=0)
    learning_rate = st.selectbox('Learning Rate', options=[0.01,0.05,0.1], index=0)

if st.session_state.initial_run:
    model = xgb.XGBClassifier()
else:
    model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, gamma=gamma)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

with disp_col:
    st.subheader('Last Model')
    if st.session_state.initial_run != 1:
        st.metric('Mean Squared Error',value= mse, delta= mse-st.session_state.best_mse, delta_color= "inverse")
        st.metric('Gamma',value= gamma)
        st.metric('Max Depth',value= max_depth)
        st.metric('Learning Rate',value= learning_rate)
    else:
        st.metric('Mean Squared Error',value= mse)
        st.metric('Gamma',value= model.get_xgb_params()['gamma'])
        st.metric('Max Depth',value= model.get_xgb_params()['max_depth'])
        st.metric('Learning Rate',value= model.get_xgb_params()['learning_rate'])

with best_col:
    st.subheader('Best Model so far')
    if st.session_state.initial_run != 1:
        st.metric('Mean Squared Error',value= st.session_state.best_mse)
        st.metric('Gamma',value= st.session_state.best_gamma)
        st.metric('Max Depth',value= st.session_state.best_depth)
        st.metric('Learning Rate',value= st.session_state.best_lr)
    else:
        st.metric('Mean Squared Error',value= '-')
        st.metric('Gamma',value= '-')
        st.metric('Max Depth',value= '-')
        st.metric('Learning Rate',value= '-')
    
if st.session_state.initial_run or mse < st.session_state.best_mse:
    st.session_state.best_lr = model.get_xgb_params()['learning_rate']
    st.session_state.best_gamma = model.get_xgb_params()['gamma']
    st.session_state.best_depth = model.get_xgb_params()['max_depth']
    st.session_state.best_mse = mse

st.subheader('Confusion Matrix')
st.write('Let\'s have a look at the confusion matrix to see how many predictions we predicted wrong in which category (confusion matrix shows data for the last trained model).')
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
fig2 = plt.figure(figsize=(10, 4))
sns.heatmap(cm, annot=True, cbar=False, cmap='Greys')
plt.xlabel('Predicted values')
plt.ylabel('True values')
st.pyplot(fig2)

st.session_state.initial_run = 0