import streamlit as st
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('user_accounts.db')
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

# Functions for CRUD operations
def create_user(username, password, email, role):
    c.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)", (username, password, email, role))
    conn.commit()

def read_users():
    c.execute("SELECT * FROM users")
    return c.fetchall()

def update_user(id, username, password, email, role):
    c.execute("UPDATE users SET username=?, password=?, email=?, role=? WHERE id=?", (username, password, email, role, id))
    conn.commit()

def delete_user(id):
    c.execute("DELETE FROM users WHERE id=?", (id,))
    conn.commit()

# Streamlit UI
def main():
    st.title("Super Admin Panel")

    menu = ["Create User", "View Users", "Update User", "Delete User"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Create User":
        st.subheader("Create User")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["User", "Admin"])
        if st.button("Create"):
            create_user(username, password, email, role)
            st.success("User created successfully!")

    elif choice == "View Users":
        st.subheader("Read Users")
        users = read_users()
        for user in users:
            st.write(f"ID: {user[0]}, Username: {user[1]}, Email: {user[3]}, Role: {user[4]}")

    elif choice == "Update User":
        st.subheader("Update User")
        users = read_users()
        user_options = [user[1] for user in users]
        selected_user = st.selectbox("Select User", user_options)
        user_data = [user for user in users if user[1] == selected_user][0]
        username = st.text_input("Username", user_data[1])
        password = st.text_input("Password", user_data[2], type="password")
        email = st.text_input("Email", user_data[3])
        role = st.selectbox("Role", ["User", "Admin"], index=0 if user_data[4] == "User" else 1)
        if st.button("Update"):
            update_user(user_data[0], username, password, email, role)
            st.success("User updated successfully!")

    elif choice == "Delete User":
        st.subheader("Delete User")
        users = read_users()
        user_options = [user[1] for user in users]
        selected_user = st.selectbox("Select User", user_options)
        user_data = [user for user in users if user[1] == selected_user][0]
        if st.button("Delete"):
            delete_user(user_data[0])
            st.success("User deleted successfully!")

if __name__ == "__main__":
    main()
