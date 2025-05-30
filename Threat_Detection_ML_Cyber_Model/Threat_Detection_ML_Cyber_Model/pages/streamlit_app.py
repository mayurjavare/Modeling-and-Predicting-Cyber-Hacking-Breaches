import os
import streamlit as st

from file_checker import checkFile

st.title("Malware Detection using Random Forest Algorithm")
st.markdown(""" 
Malwares can wreak havoc on a computer and its network. Hackers use it to steal passwords, delete files and render computers inoperable. A malware infection can cause many problems that affect daily operation and the long-term security of your company.

Python program for detecting whether a given file is 
a probable malware or not! using [Random Forest Algorithm](https://en.wikipedia.org/wiki/Random_forest) for 
classification.""")

st.subheader("Scan your files:-")

file = st.file_uploader("Upload a file to check for malwares:", accept_multiple_files=True)
if len(file):
    with st.spinner("Checking..."):
        for i in file:
            open('malwares/tempFile', 'wb').write(i.getvalue())
            legitimate = checkFile("malwares/tempFile")
            os.remove("malwares/tempFile")
            if legitimate:
                st.write(f"File {i.name} seems *LEGITIMATE*!")
            else:
                st.markdown(f"File {i.name} is probably a **MALWARE**!!!")

# # Add Visualizations button
# if st.button("Visualizations"):
#     # Replace with code for visualizations
#     st.write("Visualizations will be displayed here.")

# # Add Accuracy Reports button
# if st.button("Accuracy Reports"):
#     # Replace with code for accuracy reports
#     st.write("Accuracy reports will be displayed here.")
