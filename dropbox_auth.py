import os
from dropbox import DropboxOAuth2FlowNoRedirect
import json
from dotenv import load_dotenv

load_dotenv()

def generate_refresh_token():
    """One-time script to get refresh token and save credentials"""
    auth_flow = DropboxOAuth2FlowNoRedirect(
        os.getenv("DROPBOX_APP_KEY"),
        os.getenv("DROPBOX_APP_SECRET"),
        token_access_type='offline'
    )

    authorize_url = auth_flow.start()
    print("1. Go to: " + authorize_url)
    print("2. Click \"Allow\" (you might have to log in first).")
    print("3. Copy the authorization code.")
    auth_code = input("Enter the authorization code here: ").strip()

    try:
        oauth_result = auth_flow.finish(auth_code)
        
        # Save credentials to a file
        credentials = {
            "refresh_token": oauth_result.refresh_token,
            "access_token": oauth_result.access_token,
            "app_key": os.getenv("DROPBOX_APP_KEY"),
            "app_secret": os.getenv("DROPBOX_APP_SECRET")
        }
        
        # Save to a secure location
        creds_path = '.dropbox_credentials.json'
        with open(creds_path, 'w') as f:
            json.dump(credentials, f)
        
        print(f"Credentials saved to {creds_path}")
        return oauth_result.refresh_token
        
    except Exception as e:
        print('Error: %s' % (e,))
        raise

if __name__ == "__main__":
    generate_refresh_token()