import requests
import pandas as pd

ENDPOINT = "https://codex.flywire.ai/api/fetch_google_spreadsheet_data"


def seg_ids_matching_gsheet(search_string,
                            gsheet_id,
                            user_id,
                            column_name):
    url = f"{ENDPOINT}/{gsheet_id}/{user_id}"
    data = requests.get(url).json()
    # turn each tab into a DataFrame
    dfs = { tab: pd.DataFrame(rows) 
        for tab, rows in data.items() }

    T = dfs["Sheet1"]
    # 1) Take the first row as column names
    new_columns = T.iloc[0].tolist()     # grab row-0 values
    T.columns = new_columns              # assign them as the header

    # 2) Drop that first row from the data
    T = T.drop(index=0).reset_index(drop=True)
    # 3) Filter the DataFrame based on the search string in the specified column
    filtered_df = T[T[column_name].str.contains(search_string, na=False, case=False)]  
    # 4) Return the 'Segment ID' column as a list
    seg_id_list = filtered_df['Final SegID'].tolist() if 'Final SegID' in filtered_df.columns else []
    # Optionally, you can filter out None or non-integer values if needed
    seg_id_list = [x for x in seg_id_list if x is not None and str(x).isdigit()]  # Ensure all IDs are valid integers
    return seg_id_list
