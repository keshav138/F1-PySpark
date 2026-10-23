import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # This tricks Python into running Streamlit as if typed in the terminal
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())