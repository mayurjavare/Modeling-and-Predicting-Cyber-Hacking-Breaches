import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache(allow_output_mutation=True)
def load_data(file_path):
    column_names = ["duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes", "land", "wrongfragment", "urgent", "hot", "numfailedlogins", "loggedin", "numcompromised", "rootshell", "suattempted", "numroot", "numfilecreations", "numshells", "numaccessfiles", "numoutboundcmds", "ishostlogin", "isguestlogin", "count", "srvcount", "serrorrate", "srvserrorrate", "rerrorrate", "srvrerrorrate", "samesrvrate", "diffsrvrate", "srvdiffhostrate", "dsthostcount", "dsthostsrvcount", "dsthostsamesrvrate", "dsthostdiffsrvrate", "dsthostsamesrcportrate", "dsthostsrvdiffhostrate", "dsthostserrorrate", "dsthostsrvserrorrate", "dsthostrerrorrate", "dsthostsrvrerrorrate", "attack", "lastflag"]
    df = pd.read_csv(file_path, sep=",", names=column_names)
    return df

# Preprocess data
def preprocess_data(df):
    df['attack'].loc[df['attack'] != 'normal'] = 'attack'
    le = LabelEncoder()
    df['protocoltype'] = le.fit_transform(df['protocoltype'])
    df['service'] = le.fit_transform(df['service'])
    df['flag'] = le.fit_transform(df['flag'])
    return df

# Train model
def train_model(df, model_type='Random Forest'):
    X = df.drop(['attack'], axis=1)
    y = df['attack']
    scaler = StandardScaler()
    scaler.fit(X)
    X_transformed = scaler.transform(X)
    
    if model_type == 'Random Forest':
        model = RandomForestClassifier()
    elif model_type == 'Logistic Regression':
        model = LogisticRegression()
    
    model.fit(X_transformed, y)
    return model, scaler

# Predict using model
def predict(model, scaler, test_df):
    test_df['attack'].loc[test_df['attack'] != 'normal'] = 'attack'
    
    # Define LabelEncoder
    le = LabelEncoder()
    
    test_df['protocoltype'] = le.fit_transform(test_df['protocoltype'])
    test_df['service'] = le.fit_transform(test_df['service'])
    test_df['flag'] = le.fit_transform(test_df['flag'])
    X_test = test_df.drop(['attack'], axis=1)
    y_test = test_df['attack']
    X_test_transformed = scaler.transform(X_test)
    test_pred = model.predict(X_test_transformed)
    return test_pred, y_test

# Main function
def main():
    st.title('Network Suspicion Detection App')

    # Sidebar
    uploaded_file_train = st.sidebar.file_uploader("Upload Training File", type=['csv'])
    uploaded_file_test = st.sidebar.file_uploader("Upload Test File", type=['csv'])
    model_type = st.sidebar.radio("Select Model", ('Random Forest', 'Logistic Regression'))

    if uploaded_file_train and uploaded_file_test:
        train_df = load_data(uploaded_file_train)
        test_df = load_data(uploaded_file_test)

        st.write("## Training Data")
        st.write(train_df.head())

        st.write("## Test Data")
        st.write(test_df.head())

        st.write("### Preprocessing Data...")
        train_df = preprocess_data(train_df)
        test_df = preprocess_data(test_df)

        st.write("### Training Model...")
        model, scaler = train_model(train_df, model_type)

        st.write("### Making Predictions...")
        test_pred, y_test = predict(model, scaler, test_df)

        st.write("### Model Evaluation")
        st.write("Accuracy:", accuracy_score(y_test, test_pred))
        st.write("Classification Report:")
        st.write(classification_report(y_test, test_pred))

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, test_pred))

if __name__ == '__main__':
    main()
