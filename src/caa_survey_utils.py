import pandas as pd

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