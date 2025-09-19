import pandas as pd
import numpy as np
import logging

import sys
sys.path.append('..\..')
from src import config
logging.basicConfig(level=logging.ERROR)

def error_handling_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return np.nan
    return wrapper

def auto_apply_decorator(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith('__'):  # Ignore built-in methods
            setattr(cls, attr_name, error_handling_decorator(attr_value))
    return cls

@auto_apply_decorator
# Define the class
class ModeConditionMapper:
    def __init__(self, dataframe):
        """Initialize with the DataFrame."""
        self.df = dataframe

        self.mode_condition_lu = pd.read_csv(rf'{config.DATA_DIR}\mode_conditions\version2\caa_mode_allocation_lasam_mode_lu.csv')
        
    def step_1(self):
        conditions = [
            (self.df['Last'] == "Other") & (self.df['2ndLast'].isin(["Other", "No Mode"])) & (self.df['3rdLast'].isin(["Other", "No Mode"])),
            (self.df['Last'] == "Other") & (self.df['2ndLast'].isin(["Other", "No Mode"])) & (~self.df['3rdLast'].isin(["Other", "No Mode"])),
            (self.df['Last'] == "Other") & (~self.df['2ndLast'].isin(["Other", "No Mode"])),
            (self.df['Last'] != "Other")
        ]
        
        choices = [
            "Other",
            self.df['3rdLast'],
            self.df['2ndLast'],
            self.df['Last']
        ]
        
        return np.select(conditions, choices, default=np.nan)
    
    def step_2(self):
        conditions = [
            ((self.df['Step_1'].isin(["Cycle", "Walk (where only mode)"])) & 
            (self.df['2ndLast'].isin(["Cycle", "Walk (where only mode)"])) & 
            (self.df['3rdLast'] == 'No Mode')),
            
            ((self.df['Step_1'].isin(["Cycle", "Walk (where only mode)"])) & 
            (self.df['2ndLast'].isin(["Cycle", "Walk (where only mode)"])) & 
            (self.df['3rdLast'] != 'No Mode')),
            
            ((self.df['Step_1'].isin(["Cycle", "Walk (where only mode)"])) & 
            (self.df['2ndLast'] == 'No Mode')),
            
            ((self.df['Step_1'].isin(["Cycle", "Walk (where only mode)"])) & 
            (self.df['2ndLast'] != 'No Mode')),
            
            (~self.df['Step_1'].isin(["Cycle", "Walk (where only mode)"]) & 
             (self.df['Step_1'] == 'No Mode')),
            
            (~self.df['Step_1'].isin(["Cycle", "Walk (where only mode)"]) & 
             (self.df['Step_1'] != 'No Mode'))
        ]
        
        choices = [
            "Other",
            self.df['3rdLast'],
            "Other",
            self.df['2ndLast'],
            "No Mode",
            self.df['Step_1']
        ]
        
        return np.select(conditions, choices, default=np.nan)
    
    def step_3(self):
        conditions = [
            (self.df['Step_2'] == "Tube/Metro/Subway") & (self.df['Contains_Heathrow_Express'] == True),
            (self.df['Step_2'] == "Tube/Metro/Subway") & (self.df['2ndLast'] == "Elizabeth Line")
        ]

        choices = [
            "Heathrow Express",
            "Elizabeth Line",
        ]

        return np.select(conditions, choices, default=self.df['Step_2'])

    def step_4(self):
        conditions = [
            (self.df['Contains_Heathrow_Express'] == True) & (self.df['Contains_Elizabeth_Line'] == False)
        ]

        choices = [
            "Heathrow Express"
        ]

        return np.select(conditions, choices, default=self.df['Step_3'])
        
    def step_5(self):
        conditions = [
            (self.df['Contains_Heathrow_Express'] == True) & 
            (self.df['Contains_Elizabeth_Line'] == True) & 
            (self.df['Terminal'] == 5),

            (self.df['Contains_Heathrow_Express'] == True) & 
            (self.df['Contains_Elizabeth_Line'] == True) & 
            (self.df['Terminal'] != 5)
        ]

        choices = [
            "Elizabeth Line",
            "Heathrow Express"
        ]

        return np.select(conditions, choices, default=self.df['Step_4'])

    def step_6(self):
        conditions = [
            (self.df['Contains_Rental'] == True) & 
            ~(self.df['Contains_Heathrow_Express'] | self.df['Contains_Elizabeth_Line'] | self.df['Contains_Tube'])
        ]

        choices = [
            "Rentals"
        ]

        return np.select(conditions, choices, default=self.df['Step_5'])
    
    def step_7(self):
        
        included_mode = ['Private car - driven away', 'Uber', 'Minicab', 'Taxi', 'Chauffer', 'Taxi/Minicab Unspecified']
        incuded_2ndlast_mode = ['National Rail', 'London Underground', 
                                'London bus companies', 'Local bus companies', 'Bus Unspecified', 'Charter coach', 
                                'National Express Coach', 'Other National/Regional coach service', 
                                'Bus/coach company unknown', 'RailAir Bus (Reading/Woking/Feltham)', 'Elizabeth Line', 
                                'Tube/Metro/Subway', 'LHR-LTN Coach Service', 'Airport to airport coach service']

        conditions = [
            (self.df['Step_6'].isin(included_mode)) & 
            (self.df['2ndLast'].isin(incuded_2ndlast_mode))
        ]

        choices = [
            self.df['2ndLast']
        ]
        
        return np.select(conditions, choices, default=self.df['Step_6'])

    def step_8(self):
        conditions = [
            (self.df['Last']=='Hotel bus') & (self.df['2ndLast']=='Charter coach')
        ]

        choices = [
            "Charter coach"
        ]

        return np.select(conditions, choices, default=self.df['Step_7'])

    def step_9(self):
        conditions = [
            (self.df['Step_8'] == 'Car Unspecified') & (self.df['SYSTEM_COUNTRY']=='UK'),
            (self.df['Step_8'] == 'Car Unspecified') & (self.df['SYSTEM_COUNTRY']=='Foreign'),
            (self.df['Step_8'] != 'Car Unspecified')
        ]

        choices = [
            "Car Unspecified UK",
            "Car Unspecified Foreign",
            self.df['Step_8']
        ]

        return np.select(conditions, choices, default=np.nan)
    
    def step_10(self):
        conditions = [
            ((self.df['Step_9'] == 'Airport to airport coach service') & 
            (
                (self.df['Origin'] == 'AIRPORT') | 
                (self.df['SYSTEM_District'].str.contains('airport', case=False, na=False)) |
                (self.df['SYSTEM_District']=='Crawley District (SE)')
            )),

            (self.df['Step_9'] == 'Airport to airport coach service')
        ]

        choices = [
            'Airport to airport coach service',
            'National Express Coach'
        ]

        return np.select(conditions, choices, default=self.df['Step_9'])
    
    def step_11(self):
        def apply_condition(row):
            columns_to_check = ['Last', '2ndLast', '3rdLast']
            preceding_modes = ["Tube/Metro/Subway", "Elizabeth Line", "TfL Rail (formerly Heathrow Connect)", "National railways", "Rail Unspecified"]
            
            railair_bus = 'RailAir Bus (Reading/Woking/Feltham)'
            
            for i, col in enumerate(columns_to_check):
                if railair_bus in str(row[col]):
                    # Check preceding columns for preceding_modes
                    for prev_col in columns_to_check[i+1:]:
                        if any(mode in str(row[prev_col]) for mode in preceding_modes):
                            return railair_bus
                    # If we've checked all preceding columns and found no preceding_modes
                    return 'Other National/Regional coach service'
            
            # If RailAir Bus is not found in any column
            return row['Step_10']
        
        return self.df.apply(apply_condition, axis=1)

    def apply_steps(self):
        
        steps = [self.step_1, self.step_2, self.step_3, self.step_4, self.step_5, self.step_6, self.step_7, self.step_8, self.step_9, self.step_10, self.step_11]
        
        for i, step in enumerate(steps, 1):
            result = step()
            self.df[f'Step_{i}'] = result

        return self.df
    
    def assign_lasam_mode(self):
        self.df = self.df.merge(self.mode_condition_lu, 
                                left_on='Step_11',
                                right_on='Mode_Allocated', how='left')

        return self.df
    

    def main_run_all(self):

        # Step 1: apply conditions
        self.df = self.apply_steps()

        # Step 2: assign LASAM main mode and mode based on mode allocated
        self.df = self.assign_lasam_mode()

        # # drop condition columns
        # df_mode_mapped.drop(columns=self.step_columns, inplace=True)

        return self.df

