
import requests
import pandas as pd
import numpy as np
from typing import Dict, List
from typing import Optional, Tuple, List

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


def seg_ids_and_soma_pos_matching_gsheet_multi(
    gsheet_id: str,
    user_id: str,
    human_cell_type: Optional[str] = None,
    machine_cell_type: Optional[str] = None,
    cell_class: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Filter a Google Sheet by multiple optional conditions combined with AND:
      - human_cell_type matches column "Cell Type" via case-insensitive contains
      - machine_cell_type matches column "Machine label" via case-insensitive contains
      - cell_class matches column "Cell Class" via case-insensitive equality

    Returns (seg_id_list, soma_pos) like seg_ids_and_soma_pos_matching_gsheet.
    """
    url = f"{ENDPOINT}/{gsheet_id}/{user_id}"
    data = requests.get(url).json()

    dfs = {tab: pd.DataFrame(rows) for tab, rows in data.items()}
    T = dfs[list(dfs.keys())[0]]  # Use the first sheet if specific not present

    # Use the first row as header
    new_columns = T.iloc[0].tolist()
    T.columns = new_columns
    # Remove header row from data
    T = T.drop(index=0).reset_index(drop=True)

    # Start with all rows and apply filters
    if len(T) == 0:
        return [], np.array([])

    mask = pd.Series([True] * len(T))

    if human_cell_type:
        col = "Cell Type"
        if col in T.columns:
            mask &= T[col].astype(str).str.contains(human_cell_type, case=False, na=False)
        else:
            mask &= False

    if machine_cell_type:
        col = "Machine label"
        if col in T.columns:
            mask &= T[col].astype(str).str.contains(machine_cell_type, case=False, na=False)
        else:
            mask &= False

    if cell_class:
        col = "Cell Class"
        if col in T.columns:
            # exact match ignoring case and leading/trailing whitespace
            mask &= (
                T[col].astype(str).str.strip().str.casefold()
                == str(cell_class).strip().casefold()
            )
        else:
            mask &= False

    filtered_df = T[mask]

    seg_id_list = (
        filtered_df['Final SegID'].tolist() if 'Final SegID' in filtered_df.columns else []
    )
    seg_id_list = [x for x in seg_id_list if x is not None and str(x).isdigit()]

    nuc_coords = (
        filtered_df['Nuc Coords'].tolist() if 'Nuc Coords' in filtered_df.columns else []
    )
    soma_pos_list: List[List[float]] = []
    for entry in nuc_coords:
        if isinstance(entry, str):
            parts = entry.split(',')
            if len(parts) >= 2:
                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    soma_pos_list.append([x, y])
                except Exception:
                    continue
    soma_pos = np.array(soma_pos_list)
    return seg_id_list, soma_pos


def cell_types_by_class_map(
    gsheet_id: str,
    user_id: str,
    tab_name: str = "Cell types and properties",
) -> Dict[str, List[str]]:
    """
    Read the "Cell types and properties" tab and build a mapping
    from Cell Class to a sorted list of Cell Type (human) names.

    Assumes the first row is a header and the first two columns correspond to
    cell type names and cell class, respectively.
    """
    url = f"{ENDPOINT}/{gsheet_id}/{user_id}"
    data = requests.get(url).json()

    if tab_name not in data:
        # Fallback: try to find a tab with a similar name (case-insensitive contains)
        for tab in data.keys():
            if tab.lower().strip() == tab_name.lower().strip():
                tab_name = tab
                break

    rows = data.get(tab_name)
    if not rows:
        return {}

    T = pd.DataFrame(rows)
    if T.empty:
        return {}

    # Use first row as header
    new_columns = T.iloc[0].tolist()
    T.columns = new_columns
    # Drop header row
    T = T.drop(index=0).reset_index(drop=True)

    # Use first two columns by position to avoid header name dependency
    if T.shape[1] < 2:
        return {}

    first_col_name = T.columns[0]
    second_col_name = T.columns[1]

    mapping: Dict[str, List[str]] = {}
    for _, row in T.iterrows():
        ct = str(row.get(first_col_name, "")).strip()
        cc = str(row.get(second_col_name, "")).strip()
        if not ct or not cc or ct.lower() == "nan" or cc.lower() == "nan":
            continue
        mapping.setdefault(cc, []).append(ct)

    # Deduplicate and sort
    for k in list(mapping.keys()):
        uniq_sorted = sorted({v for v in mapping[k] if v and v.lower() != "nan"})
        mapping[k] = uniq_sorted

    return mapping
