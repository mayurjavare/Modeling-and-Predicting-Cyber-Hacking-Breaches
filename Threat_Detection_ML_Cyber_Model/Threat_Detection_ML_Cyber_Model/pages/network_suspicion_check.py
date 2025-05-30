import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training data
@st.cache(allow_output_mutation=True)
def load_train_data(file_path):
    column_names = ["duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes", "land", "wrongfragment", "urgent", "hot", "numfailedlogins", "loggedin", "numcompromised", "rootshell", "suattempted", "numroot", "numfilecreations", "numshells", "numaccessfiles", "numoutboundcmds", "ishostlogin", "isguestlogin", "count", "srvcount", "serrorrate", "srvserrorrate", "rerrorrate", "srvrerrorrate", "samesrvrate", "diffsrvrate", "srvdiffhostrate", "dsthostcount", "dsthostsrvcount", "dsthostsamesrvrate", "dsthostdiffsrvrate", "dsthostsamesrcportrate", "dsthostsrvdiffhostrate", "dsthostserrorrate", "dsthostsrvserrorrate", "dsthostrerrorrate", "dsthostsrvrerrorrate", "attack", "lastflag"]
    df = pd.read_csv(file_path, sep=",", names=column_names)
    return df

# Load network data
@st.cache(allow_output_mutation=True)
def load_network_data(file_path):
    column_names = ["duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes", "land", "wrongfragment", "urgent", "hot", "numfailedlogins", "loggedin", "numcompromised", "rootshell", "suattempted", "numroot", "numfilecreations", "numshells", "numaccessfiles", "numoutboundcmds", "ishostlogin", "isguestlogin", "count", "srvcount", "serrorrate", "srvserrorrate", "rerrorrate", "srvrerrorrate", "samesrvrate", "diffsrvrate", "srvdiffhostrate", "dsthostcount", "dsthostsrvcount", "dsthostsamesrvrate", "dsthostdiffsrvrate", "dsthostsamesrcportrate", "dsthostsrvdiffhostrate", "dsthostserrorrate", "dsthostsrvserrorrate", "dsthostrerrorrate", "dsthostsrvrerrorrate", "attack", "lastflag"]
    df = pd.read_csv(file_path, sep=",", names=column_names)
    return df

# Preprocess training data
def preprocess_train_data(df):
    df['attack'].loc[df['attack'] != 'normal'] = 'attack'
    le = LabelEncoder()
    df['protocoltype'] = le.fit_transform(df['protocoltype'])
    df['service'] = le.fit_transform(df['service'])
    df['flag'] = le.fit_transform(df['flag'])
    return df

# Preprocess network data
def preprocess_network_data(df):
    le = LabelEncoder()
    df['protocoltype'] = le.fit_transform(df['protocoltype'])
    df['service'] = le.fit_transform(df['service'])
    df['flag'] = le.fit_transform(df['flag'])
    return df

# Train model
def train_model(df):
    X = df.drop(['attack'], axis=1)
    y = df['attack']
    scaler = StandardScaler()
    scaler.fit(X)
    X_transformed = scaler.transform(X)
    model = RandomForestClassifier()
    model.fit(X_transformed, y)
    return model, scaler

# Predict using model
def predict(model, scaler, test_df):
    test_df['attack'].loc[test_df['attack'] != 'normal'] = 'attack'
    X_test = test_df.drop(['attack'], axis=1)
    y_test = test_df['attack']
    X_test_transformed = scaler.transform(X_test)
    test_pred = model.predict(X_test_transformed)
    return test_pred, y_test

# Main function
def main():
    st.title('Network Anomaly Detection App')

    # Sidebar
    uploaded_train_file = st.sidebar.file_uploader("Upload Training File", type=['csv'])
    uploaded_network_file = st.sidebar.file_uploader("Upload Network File", type=['log'])

    if uploaded_train_file and uploaded_network_file:
        train_df = load_train_data(uploaded_train_file)
        network_df = load_network_data(uploaded_network_file)

        st.write("## Training Data")
        st.write(train_df.head())

        st.write("## Network Data")
        st.write(network_df.head())

        st.write("### Preprocessing Training Data...")
        train_df = preprocess_train_data(train_df)

        st.write("### Training Model...")
        model, scaler = train_model(train_df)

        st.write("### Preprocessing Network Data...")
        network_df = preprocess_network_data(network_df)

        st.write("### Checking Suspicious Activity on Network Data...")
        test_pred, y_test = predict(model, scaler, network_df)

        # Determine if suspicious activity is present or network is normal
        if 'attack' in network_df.columns:
            if 'attack' in y_test.unique() and len(y_test.unique()) == 1:
                result = y_test.unique()[0]
            else:
                result = 'suspicious activity detected' if 'attack' in test_pred else 'normal network'
            # Display the result with color
            if result == 'suspicious activity detected':
                st.write("### Result:", "<span style='color:red'>{}</span>".format(result), unsafe_allow_html=True)
            else:
                st.write("### Result:", "<span style='color:green'>{}</span>".format(result), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
