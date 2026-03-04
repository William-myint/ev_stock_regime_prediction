Project Setup & Usage Guide

Follow these steps to get the application running on your local machine.

1. Get the Code

Clone this repository or download the source code as a ZIP file and extract it to a folder of your choice.

2. Open Terminal & Navigate

Open your terminal (Command Prompt, PowerShell, or Terminal) and navigate to the project folder:

cd path/to/ev_stock_regime_prediction


3. Create a Virtual Environment

It is recommended to use a virtual environment to keep dependencies organized.

For Windows:

python -m venv venv
venv\Scripts\activate


For Linux / macOS:

python3 -m venv venv
source venv/bin/activate


(You should see (venv) appear in your terminal prompt.)

4. Install Dependencies

Once the environment is active, install the required libraries:

pip install -r requirements.txt


5. Run the Application

Launch the Streamlit server with the following command:

streamlit run app.py


Note: If your main script is named differently (e.g., main.py), replace app.py with your filename.

6. Access the App

After the command runs, the terminal will provide a local URL. Open your web browser and navigate to:
http://localhost:8501

The application should now be loaded and ready for use!
