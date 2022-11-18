import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import xgboost as xgb
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

header = st.container()
dataset = st.container()
correlation = st.container()
params = st.container()
metrics = st.container()
graphs = st.container()

if 'initial_run' not in st.session_state:
    st.session_state.initial_run = 1

with header:
    st.title('Interactive Hyperparameter tuning for a XGBoost model')

with dataset:
    st.header('Ad click dataset')
    st.write('The ad click dataset could be found on Kaggle and shows coversions (Purchased = [0,1]) for 400 customers. It also contains gender, age and the estimated salary of the customers. The goal is to predict if a customer will convert. \n\n First 5 lines:')
    ads = pd.read_csv('Social_Network_Ads.csv')
    st.write(ads.head())

ads['Gender_Flag']= ads['Gender'].map({'Male':0, 'Female':1})

with correlation:
    st.header('Correlation Matrix')
    st.write('Gender was converted to a 0/1 flag to consider it in the correlation matrix. The Correlation Matrix shows that age and the estimated salary have the highest potential as predictors for conversions (=Purchased).')
    X, y = ads[['Gender_Flag', 'Age', 'EstimatedSalary']], ads['Purchased']
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.33, random_state=42)
    corr = ads.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig1 = plt.figure(figsize=(10, 4))
    sns.heatmap(corr, mask=mask, annot=True)
    st.pyplot(fig1)

sel_col, disp_col, best_col = st.columns(3)

with params:
    ##################input variables######################
    sel_col.subheader('Parameters')
    sel_col.write('Try different hyperparameters (model training is executed immediately after every change):')
    gamma = sel_col.selectbox('Gamma', options=[0, 0.5, 1], index=0)
    max_depth = sel_col.selectbox('Max Depth', options=[100,200,300], index=0)
    learning_rate = sel_col.selectbox('Learning Rate', options=[0.01,0.05,0.1], index=0)

if st.session_state.initial_run:
    model = xgb.XGBClassifier()
else:
    model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, gamma=gamma)

with metrics:
    st.header('Model Training')
    st.write('Find the best model by changing the hyperparameters, the initial run happens with default values from xgboost library.')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    disp_col.subheader('Last Model')
    if st.session_state.initial_run != 1:
        disp_col.metric('Mean Squared Error',value= mse, delta= mse-st.session_state.best_mse, delta_color= "inverse")
        disp_col.metric('Gamma',value= gamma)
        disp_col.metric('Max Depth',value= max_depth)
        disp_col.metric('Learning Rate',value= learning_rate)
    else:
        disp_col.metric('Mean Squared Error',value= mse)
        disp_col.metric('Gamma',value= model.get_xgb_params()['gamma'])
        disp_col.metric('Max Depth',value= model.get_xgb_params()['max_depth'])
        disp_col.metric('Learning Rate',value= model.get_xgb_params()['learning_rate'])

    best_col.subheader('Best Model so far')
    if st.session_state.initial_run != 1:
        best_col.metric('Mean Squared Error',value= st.session_state.best_mse)
        best_col.metric('Gamma',value= st.session_state.best_gamma)
        best_col.metric('Max Depth',value= st.session_state.best_depth)
        best_col.metric('Learning Rate',value= st.session_state.best_lr)
    else:
        best_col.metric('Mean Squared Error',value= '-')
        best_col.metric('Gamma',value= '-')
        best_col.metric('Max Depth',value= '-')
        best_col.metric('Learning Rate',value= '-')
    
if st.session_state.initial_run or mse < st.session_state.best_mse:
    st.session_state.best_lr = model.get_xgb_params()['learning_rate']
    st.session_state.best_gamma = model.get_xgb_params()['gamma']
    st.session_state.best_depth = model.get_xgb_params()['max_depth']
    st.session_state.best_mse = mse

g,k = st.columns([99, 1])

with graphs:
    g.subheader('Confusion Matrix')
    g.write('Let\'s have a look at the confusion matrix to see how many predictions we predicted wrong in which category. (confusion matrix shows data for the last trained model)')
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig2 = plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, cbar=False, cmap='Greys')
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    g.pyplot(fig2)

st.session_state.initial_run = 0