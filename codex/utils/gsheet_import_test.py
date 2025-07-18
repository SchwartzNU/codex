import requests, json

ENDPOINT = "https://codex.flywire.ai/api/fetch_google_spreadsheet_data"

# This is the ID in the google spreadsheet url (typically found between /d/ and /edit)
#gsheet_id = "1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk"

# For the CellTypes sheet
gsheet_id = "1PnJ9vyK7T7Z2QThWXJ_K34BbqR7IQrRX9jgTBX32CLY"

# Security by obscurity - do not share with others :)
user_id = "gregs_eyewire2"

url = f"{ENDPOINT}/{gsheet_id}/{user_id}"

# This will return a dictionary, with one entry for each tab. Keys are tab names, values are tables of data for that tab.
# If something is wrong, this will crash or return error message (e.g. spreadsheet id is incorrect, it is unaccessable, user id is incorect etc.)
data = requests.get(url).json()
