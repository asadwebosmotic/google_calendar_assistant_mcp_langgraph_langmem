'''first-time setup where you authorize the app and obtain both access token and refresh token.'''

import json
from google_auth_oauthlib.flow import InstalledAppFlow

# Path to your downloaded credentials.json
CREDENTIALS_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/calendar']

def main():
    # Create the flow using client secrets file
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_FILE, SCOPES)

    # Run local server to get authorization code
    creds = flow.run_local_server(port=0)

    # Save the credentials for later use
    with open('token.json', 'w') as token_file:
        token_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        json.dump(token_data, token_file, indent=4)

    print("Access Token:", creds.token)
    print("Refresh Token:", creds.refresh_token)

if __name__ == '__main__':
    main()
