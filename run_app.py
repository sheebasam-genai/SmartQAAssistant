import os
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = [
        "streamlit", 
        "run", 
        os.path.join("C:\\Users\\Sheeba\\GENAIPOC\\SmartQAAssistant\\src\\ai_chatbot", "app_streamlit.py")
    ]
    sys.exit(stcli.main())
