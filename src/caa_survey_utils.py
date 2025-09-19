import pandas as pd
import numpy as np

def process_dummy_records(caa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove dummy records from the CAA DataFrame and uplift remaining records to maintain the original population.

    This function processes the Civil Aviation Authority (CAA) survey by removing dummy records
    and adjusting the population of remaining records to maintain the original total population.

    Parameters
    ----------
    caa_df : pd.DataFrame
        Input DataFrame containing CAA data, including dummy records.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with dummy records removed and population uplifted.

    Notes
    -----
    The function performs the following steps:
    1. Identifies and removes dummy records.
    2. Calculates a global population uplift factor.
    3. Applies the uplift factor to maintain the original total population.
    4. Converts the 'APT_TERMINAL' column from string to integer.

    The 'POP' column in the returned DataFrame is adjusted to maintain the original total population.
    """

    # Remove dummy records and uplift remaining records to maintain CAA 2023 population

    caa_df.copy()

    # CAA population for London Heathrow (before removing dummy records)
    caa_lhr_pop = caa_df['POP'].sum()

    # Isolate dummy records
    dummy_record_idx = caa_df[caa_df['DUMMY_FLAG']=='Dummy Record'].index # Index of dummy records
    dummy_record = caa_df.loc[dummy_record_idx] # dataframe containing all dummy records
    dummy_record.reset_index(drop=True, inplace=True)

    # Remove dummy records from CAA 2023 data for London Heathrow
    caa_df.drop(dummy_record_idx, inplace=True)
    caa_df.reset_index(drop=True, inplace=True)
    # CAA population for London Heathrow (after removing dummy records)
    caa_lhr_reduced_pop = caa_df['POP'].sum()

    # global uplift factor to apply to maintain the original population total
    global_pop_uplift = caa_lhr_pop/caa_lhr_reduced_pop

    # Uplift CAA 2023 population for London Heathrow
    caa_df['POP'] = caa_df['POP'] * global_pop_uplift

    print(f'{len(dummy_record_idx)} dummy records removed and reminaing population uplifted by {global_pop_uplift}')

    # Reformat Terminal column from string to int
    caa_df['APT_TERMINAL'] = caa_df['APT_TERMINAL'].astype(int)
    
    return caa_df


def remove_interline_pax(caa_df: pd.DataFrame) -> pd.DataFrame:
    # remove interline passengers from the caa data
    caa_df = caa_df.copy()

    interline_passenger_idx = caa_df[caa_df['SYSTEM_TI']=='Interline'].index
    interline_passenger = caa_df.loc[interline_passenger_idx]
    interline_passenger.reset_index(drop=True, inplace=True)

    print(f'removed {len(interline_passenger_idx)} rows with interline passengers')

    caa_df.drop(interline_passenger_idx, inplace=True)
    caa_df.reset_index(drop=True, inplace=True)

    return caa_df


def apply_last_mode(row: pd.Series) -> pd.Series:
    # if MODEC and MODEB are empty the last mode should be MODEA
    if row['MODEC'] in ['No Mode', np.NaN] and row['MODEB'] in ['No Mode', np.NaN]:
        return row['MODEA']
    # if only MODEC is empty the last mode should be MODEB
    elif row['MODEC'] in ['No Mode', np.NaN]:
        return row['MODEB']
    # if MODEC has a value, use that for the last mode
    else:
        return row['MODEC']
    
def apply_2ndlast_mode(row: pd.Series) -> str:
    # if MODEC and MODEB are empty the second-to-last mode should be MODEA
    if row['MODEC'] in ['No Mode', np.NaN] and row['MODEB'] in ['No Mode', np.NaN]:
        return 'No Mode'
    # if only MODEC is empty the second-to-last mode should be MODEB
    elif row['MODEC'] in ['No Mode', np.NaN]:
        return row['MODEA']
    # if MODEB has a value, use that for the last mode
    else:
        return row['MODEB']
    
def apply_3rdlast_mode(row):
    # if MODEC is empty there is no third-to-last mode
    if row['MODEC'] in ['No Mode', np.NaN]:    
        return 'No Mode'
    # if MODEC is not empty use MODEA as the third-to-last mode
    else:
        return row['MODEA']
    


def apply_contains_mode(row: pd.Series, mode: str|list[str]) -> pd.Series:
    # returns true or false based on wether any of the modes contain the mode passed as an argument

    if type(mode) != list:
        mode = [mode]
    
    return (
        row['Last'] in mode or
        row['2ndLast'] in mode or
        row['3rdLast'] in mode
    )