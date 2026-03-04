# PROJECT SETUP & USAGE GUIDE

Follow the steps below to get the application running on your local machine.

---

1. GET THE CODE

---

Clone this repository
OR
Download the source code as a ZIP file and extract it to a folder of your choice.

---

2. OPEN TERMINAL & NAVIGATE

---

Open your terminal (Command Prompt, PowerShell, or Terminal)
Navigate to the project folder:

cd path/to/ev_stock_regime_prediction

---

3. CREATE A VIRTUAL ENVIRONMENT

---

It is recommended to use a virtual environment to keep dependencies organized.

For Windows:

python -m venv venv
venv\Scripts\activate

For Linux / macOS:

python3 -m venv venv
source venv/bin/activate

You should see (venv) appear in your terminal prompt.

---

4. INSTALL DEPENDENCIES

---

Once the environment is active, install the required libraries:

pip install -r requirements.txt

---

5. RUN THE APPLICATION

---

Launch the Streamlit server:

streamlit run app.py

Note: If your main script is named differently (e.g., main.py),
replace app.py with your filename.

---

6. ACCESS THE APP

---

After running the command, the terminal will provide a local URL.

Open your web browser and navigate to:

http://localhost:8501

The application should now be loaded and ready for use!

==================================================
