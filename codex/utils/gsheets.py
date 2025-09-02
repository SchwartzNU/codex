
import requests
import pandas as pd
import numpy as np

ENDPOINT = "https://codex.flywire.ai/api/fetch_google_spreadsheet_data"


def seg_ids_and_soma_pos_matching_gsheet(search_string,
                            gsheet_id,
                            user_id,
                            column_name):
    """
    Searches a Google Sheet for segment IDs matching a search string in a specified column.

    Args:
        search_string (str): The string to search for in the specified column.
        gsheet_id (str): The Google Sheet ID.
        user_id (str): The user ID for authentication.
        column_name (str): The name of the column to search.

    Returns:
        list: A list of valid segment IDs (as strings) matching the search criteria.
    """
    url = f"{ENDPOINT}/{gsheet_id}/{user_id}"
    #url = f"{ENDPOINT}/{gsheet_id}"

    data = requests.get(url).json()
    print(f"Fetched data from Google Sheet: {gsheet_id}")
    print(data.keys())
    print(f"Tabs: {list(data.keys())}")

    # Load each sheet into a DataFrame
    dfs = { tab: pd.DataFrame(rows) 
        for tab, rows in data.items() }

    #T = dfs["Sheet1"]
    T = dfs[list(dfs.keys())[0]]  # Use the first sheet if "Sheet1" not present
    # Use the first row as header
    new_columns = T.iloc[0].tolist()
    T.columns = new_columns

    # Remove header row from data
    T = T.drop(index=0).reset_index(drop=True)
    # Filter rows by search string
    filtered_df = T[T[column_name].str.contains(search_string, na=False, case=False)]
    # Extract and clean up segment IDs
    seg_id_list = filtered_df['Final SegID'].tolist() if 'Final SegID' in filtered_df.columns else []
    seg_id_list = [x for x in seg_id_list if x is not None and str(x).isdigit()]

    nuc_coords = filtered_df['Nuc Coords'].tolist() if 'Nuc Coords' in filtered_df.columns else []
    # Parse each entry as comma-separated x,y,z and extract [x, y]
    soma_pos = []
    for entry in nuc_coords:
        if isinstance(entry, str):
            parts = entry.split(',')
            if len(parts) >= 2:
                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    soma_pos.append([x, y])
                except Exception:
                    continue
    soma_pos = np.array(soma_pos)
    print(f"Soma positions: {soma_pos.shape}, Segment IDs: {len(seg_id_list)}")
    return seg_id_list, soma_pos
