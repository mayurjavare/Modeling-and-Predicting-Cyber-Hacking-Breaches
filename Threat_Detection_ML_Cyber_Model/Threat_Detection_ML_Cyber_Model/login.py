import streamlit as st
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect(r'C:\Users\mayur\OneDrive\Desktop\BE Project 1\Threat_Detection_ML_Cyber_Model\Threat_Detection_ML_Cyber_Model\Threat_detection_Super_Admin_Panel\user_accounts.db')
c = conn.cursor()

# Create a table to store user accounts if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users (
             id INTEGER PRIMARY KEY,
             username TEXT NOT NULL,
             password TEXT NOT NULL,
             email TEXT NOT NULL,
             role TEXT NOT NULL
             )''')
conn.commit()

# Function to authenticate user login
def authenticate_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    return user

# Custom CSS for appealing dark cyber theme
dark_cyber_css = """
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            font-family: 'Arial', sans-serif;
        }
        .login-container {
            background-color: #1f1f1f;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 400px;
        }
        .stTextInput {
            background-color: #37474F;
            color: #ffffff;
            border: 1px solid #03DAC6;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 15px;
        }
        .stTextInput:hover {
            border: 1px solid #03DAC6;
        }
        .stButton {
            background-color: #03DAC6;
            color: #ffffff;
            border: 2px solid #03DAC6;
            border-radius: 5px;
            padding: 10px;
            cursor: pointer;
        }
        .sponsor-line {
            text-align: center;
            margin-top: 20px;
            color: #03DAC6;
        }
    </style>
"""

# Apply custom CSS
st.markdown(dark_cyber_css, unsafe_allow_html=True)

# Create login form
st.title("Cyber Hacking Breaches Prediction using Machine Learning")

with st.form(key='login_form', clear_on_submit=True):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_button = st.form_submit_button("Login")

# Check login credentials
if login_button:
    user = authenticate_user(username, password)
    if user:
        st.success("Login Successful! Welcome to Cyber Hacking Breaches Prediction.")
        st.markdown("[Go to Dashboard](streamlit_app)")
    else:
        st.error("Invalid username or password. Please try again.")

# Sponsor line
#st.markdown("<div class='sponsor-line'>Project Sponsored By D-Soft Technology</div>", unsafe_allow_html=True)
