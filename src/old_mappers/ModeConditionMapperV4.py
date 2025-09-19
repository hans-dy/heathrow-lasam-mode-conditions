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
        self.minicab = ['Minicab', 'Uber']
        self.hotel_bus = ['Courtesy bus (travel agent)', 'Hotel bus']
        self.national_coach = ['LHR-LTN Coach Service', 'National Express Coach', 'Other National/Regional coach service']
        self.local_bus = ['Local bus companies', 'Luton airport parkway DART']
        self.tube = ['Docklands Light Railway', 'Tram', 'Tube/Metro/Subway']
        self.national_railways = ['National railways', 'National railways (MAN only) - changed trains', 'National railways (MAN only) - not changed trains']
        self.unspecified_modes = ['Car Unspecified', 'Bus Unspecified', 'Taxi/Minicab Unspecified', 'Rail Unspecified']
        
        self.mode_condition_lu = pd.read_excel(rf'{config.DATA_DIR}\mode_conditions\version1\mode_condition_mapping.xlsx', sheet_name='Mode_Conditions', usecols = ['Condition_Id', 'LASAM_Main_Mode_2024', 'LASAM_Mode_2024', 'LASAM_Mode_Code_2024', 'LASAM_Mode_Priority_2024'])
        self.mode_condition_lu.columns = ['Condition ID', 'LASAM Main Mode', 'LASAM Mode', 'LASAM Mode Code', 'LASAM Mode Priority']

        # columns 'condition_1' to 'condition_109'
        self.number_of_conditions = 109
        self.condition_columns = [f"Condition_{i}" for i in range(1, self.number_of_conditions + 1)]


    def condition_1(self, row):
        return np.where(
            (row['Last'] == 'Charter coach') | 
            (
                (row['2ndLast'] == 'Charter coach') & 
                (row['Origin'] == 'NonLDN') & 
                (row['Last'] != 'Heathrow Express')), 
            1, 
            0
        )

    def condition_2(self, row):
        return np.where(
            (row['Last'] == 'Airport to airport coach service') & 
            (row['2ndLast'] == 'No Mode') & 
            (
                (row['Origin'] == 'AIRPORT') | 
                ('airport' in str(row['SYSTEM_District']).lower()) |
                (row['SYSTEM_District']=='Crawley District (SE)')
            ), 
            2, 
            0
        )
    
    def condition_3(self, row):
        return np.where(
            (
                (row['Last'] in self.hotel_bus) | 
                (
                    (row['Last'] in self.minicab) & 
                    (row['2ndLast'] in self.hotel_bus) & 
                    (row['Origin'] == 'NonLDN')
                )
            ) & 
            (row['2ndLast'] not in ['Charter coach', 'Heathrow Express', 'Stansted Express', 'Gatwick Express']),
            3,  
            0   
        )

    def condition_4(self, row):
        return np.where(
            (row['Last'] == 'Taxi') & 
            (row['Origin'] == 'LDN') & 
            (row['2ndLast'] not in ['Heathrow Express']),
            4,
            0
        )

    def condition_5(self, row):
        return np.where(
            (row['Last'] == 'Taxi') & 
            (row['2ndLast'] == 'No Mode') & 
            (row['3rdLast'] == 'No Mode'),
            5,
            0
        )

    def condition_6(self, row):
        return np.where(
            (row['Last'] in self.minicab) & 
            (row['Origin'] == 'LDN') & 
            (row['2ndLast'] != 'Heathrow Express'),
            6,
            0
        )

    def condition_7(self, row):
        return np.where(
            (row['Last'] in self.minicab) & 
            (row['2ndLast'] == 'No Mode') & 
            (row['3rdLast'] == 'No Mode'),
            7,
            0
        )
    
    def condition_8(self, row):
        return np.where(
            (row['Last'] == 'Airline courtesy car'),
            8,
            0
        )

    def condition_9(self, row):
        excluded_modes = (
            self.national_coach + self.tube + self.local_bus + self.national_railways +
            ['Airport to airport coach service', 'London bus companies', 'Bus/coach company unknown', 'Heathrow Express',
             'Elizabeth Line', 'Stansted Express', 'Gatwick Express']
        )

        return np.where(
            (row['Last'] in ['Private car - short term car park', 'Private car - short term car park - meet/greet']) & 
            (row['Segment_4_ID'] < 3) & 
            # We do not know the trip total (days), could set to 999999
            # (
            #     ((row['AIRPORT_Prefix'] == 'LHR') & (row['trip_total_days'] <= 1)) |
            #     ((row['AIRPORT_Prefix'] == 'LGW') & (row['trip_total_days'] <= 2)) |
            #     ((row['AIRPORT_Prefix'] == 'STN') & (row['trip_total_days'] <= 2))
            # ) & 
            (
                (row['Origin'] == 'LDN') |
                (
                    (row['Origin'] == 'NonLDN') & 
                    (row['2ndLast'] not in excluded_modes)
                )
            ),
            9,
            0
        )
    
    def condition_10(self, row):
        return np.where(
            (row['Last'] == 'Rental car - short term car park'),
            10,
            0
        )
    
    def condition_11(self, row):
        included_modes = ['Private car - valet service - Off airport', 'Private car - valet service - On airport', 'Private car - airport long term car park bus',
                          'Private car - private long term car park bus', 'Private car - business car park', 'Private car - mid stay car park bus',
                          'Private car - staff car park bus', 'Private car - hotel car park bus', 'Private car - type of car park unknown'
                          ]

        excluded_modes = (
            self.national_coach + self.local_bus + self.tube + self.national_railways +
            ['Airport to airport coach service', 'London bus companies', 'Bus/coach company unknown', 
             'Heathrow Express', 'Elizabeth Line', 'Gatwick Express']
        )

        return np.where(
            (row['Last'] in included_modes) & 
            (
                (row['Origin'] == 'LDN') | 
                (
                    (row['Origin'] == 'NonLDN') & 
                    (row['2ndLast'] not in excluded_modes)
                )
            ),
            11,
            0
        )
    
    def condition_12(self, row):
        excluded_modes = self.national_coach + self.national_railways
        
        return np.where(
            (row['Last'] == 'Rental car - hire car courtesy bus') & 
            (
                (row['Origin'] == 'LDN') | 
                (
                    (row['Origin'] == 'NonLDN') & 
                    (row['2ndLast'] not in excluded_modes)
                )
            ),
            12,
            0
        )
    
    def condition_13(self, row):
        excluded_modes = (
            self.national_coach + self.local_bus + self.tube + self.national_railways +
            ['Charter coach', 'Airport to airport coach service', 'London bus companies', 'Bus/coach company unknown',
             'Heathrow Express', 'Elizabeth Line']
        )
        
        return np.where(
            (row['Last'] in ['Chauffer', 'Private car - driven away']) & 
            (
                (row['Origin'] == 'LDN') | 
                (
                    (row['Origin'] == 'NonLDN') & 
                    (row['2ndLast'] not in excluded_modes)
                ) | 
                (
                    (row['Origin'] == 'AIRPORT') & 
                    (row['2ndLast'] == 'No Mode') & 
                    (row['3rdLast'] == 'No Mode')
                )
            ),
            13,
            0
        )
    
    def condition_14(self, row):
        included_modes =['Private car - short term car park', 'Private car - short term car park - meet/greet']
        
        excluded_modes = (
            self.national_coach + self.local_bus + self.tube + self.national_railways +
            ['Airport to airport coach service', 'London bus companies', 'Bus/coach company unknown',
             'Heathrow Express', 'Elizabeth Line', 'Stansted Express', 'Gatwick Express']
        )

        return np.where(
            # Cannot do this entire block as we do not have trip_total_days
            # Taken (row['Last'] in included_modes) out to ensure the condition still works correctly
            # (
            #     (
            #         (row['Last'] in included_modes) & 
            #         (row['trip_total_days'] > 1)
            #     ) | 
            #     (
            #         (row['Last'] in included_modes) & 
            #         (row['Segment_4_ID'] > 2) & 
            #         (row['trip_total_days'] <= 1)
            #     )
            # ) &
            (row['Last'] in included_modes) &  
            (
                (row['Origin'] == 'LDN') | 
                (
                    (row['Origin'] == 'NonLDN') & 
                    (row['2ndLast'] not in excluded_modes)
                )
            ),
            14,
            0
        )
    
    def condition_15(self, row):
        return np.where(
            (row['Last'] == 'Heathrow Express') |
            (row['2ndLast'] == 'Heathrow Express'),
            15,
            0
        )
    
    # condition_16 not coded as it is specific to LGW
    def condition_16(self, row):
        return 0

    def condition_17(self, row):
        included_modes = self.tube + self.national_railways

        return np.where(
            (row['Last'] == 'RailAir Bus (Reading/Woking/Feltham)') & 
            (
                (row['2ndLast'] in included_modes) |  
                (row['3rdLast'] in included_modes) 
            ),
            17,
            0
        )
    
    def condition_18(self, row):
        return np.where(
            (row['AIRPORT_Prefix'] == 'LHR') & 
            (row['Last'] in self.tube) & 
            (row['Origin'] == 'LDN') & 
            (row['2ndLast'] != 'Heathrow Express'),
            18,
            0
        )
    
    def condition_19(self, row):
        excluded_2ndlast_modes = ['Charter coach', 'Airport to airport coach service', 'Heathrow Express']
        
        return np.where(
            (row['AIRPORT_Prefix'] == 'LHR') & 
            (row['Last'] in self.tube) & 
            (row['2ndLast'] not in excluded_2ndlast_modes) &
            (row['Origin'] == 'NonLDN'),
            19,
            0
        )
    
    def condition_20(self, row):
        return 0
    
    def condition_21(self, row):
        return 0
    
    def condition_22(self, row):
        return 0
    
    def condition_23(self, row):
        return np.where(
            (row['Last'] in self.national_coach) & 
            (
                (row['Origin'] == 'LDN') | 
                (
                    (row['Origin'] != 'LDN') & 
                    (row['2ndLast'] != 'Heathrow Express')
                )
            ),
            23,
            0
        )

    def condition_24(self, row):
        excluded_modes = self.tube + self.national_railways

        return np.where(
            (row['Last'] == 'RailAir Bus (Reading/Woking/Feltham)') & 
            (row['2ndLast'] not in excluded_modes) & 
            (row['3rdLast'] not in excluded_modes),
            24,
            0
        )
    
    def condition_25(self, row):
        included_2ndlast_modes = self.national_coach + ['Airport to airport coach service', 'Bus/Coach company unknown']
        included_3rdlast_modes = (
            self.national_coach + self.local_bus + self.national_railways +  self.minicab + 
            ['Airport to airport coach service', 'Bus/Coach company unknown', 'No Mode', 'Private car - driven away', 'Chauffer',
             'RailAir Bus (Reading/Woking/Feltham)']
        )

        excluded_2ndlast_modes = self.national_railways + ['Heathrow Express']

        return np.where(
            (row['AIRPORT_Prefix'] == 'LHR') & 
            (row['Last'] in self.tube) & 
            (row['2ndLast'] in [included_2ndlast_modes]) & 
            (row['2ndLast'] not in excluded_2ndlast_modes) & 
            (row['3rdLast'] in included_3rdlast_modes) & 
            (row['Origin'] == 'NonLDN'),
            25,
            0
        )
    
    def condition_26(self, row):
        return 0
    
    def condition_27(self, row):
        return np.where(
            (row['Last'] == 'Airport to airport coach service') & 
            (row['2ndLast'] != 'No Mode') & 
            (row['Origin'] != 'AIRPORT'),
            27,
            0
        )
    
    def condition_28(self, row):
        return 0
    
    def condition_29(self, row):
        included_last_modes = ['London bus companies'] + self.local_bus
        excluded_2ndlast_modes = self.national_railways + ['Heathrow Express', 'Stansted Express', 'Gatwick Express']

        return np.where(
            (row['Last'] in included_last_modes) & 
            (
                (
                    (row['2ndLast'] not in excluded_2ndlast_modes) & 
                    (row['3rdLast'] not in self.national_railways) & 
                    (row['Origin'] == 'NonLDN')
                ) | 
                (
                    (row['Origin'] == 'LDN') & 
                    (row['2ndLast'] not in ['Heathrow Express', 'Stansted Express', 'Gatwick Express'])
                )
            ),
            29,
            0
        )
    
    def condition_30(self, row):
        included_last_modes = [
            'Boat', 'Walk (where only mode)', 'Cycle', 'Motorcycle', 'Car Unspecified', 'Bus Unspecified', 'Taxi/Minicab Unspecified', 
            'Rail Unspecified', 'Other'
        ]

        return np.where(
            (row['Last'] in included_last_modes) & 
            (row['2ndLast'] not in ['Heathrow Express', 'Elizabeth Line', 'Stansted Express', 'Gatwick Express']),
            30,
            0
        )
    
    def condition_31(self, row):
        # combined condition 72 with 15
        return 0
    
    def condition_32(self, row):
        excluded_2ndlast_modes = ['Charter coach'] + self.national_railways

        return np.where(
            (row['Last'] == 'Bus/coach company unknown') & 
            (
                (
                    (row['2ndLast'] not in excluded_2ndlast_modes) & 
                    (row['3rdLast'] not in self.national_railways) & 
                    (row['Origin'] == 'NonLDN')
                ) | 
                (row['Origin'] == 'LDN')
            ),
            32,
            0
        )    
    
    def condition_33(self, row):
        return 0
    
    def condition_34(self, row):
        return np.where(
            (row['AIRPORT_Prefix'] == 'LHR') & 
            (row['Last'] in ['Private car - driven away', 'Chauffer']) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in self.tube),
            34,  
            0
        )    
    
    def condition_35(self, row):
        return 0
    
    def condition_36(self, row):
        included_2ndlast_modes = self.national_railways + ['Airport to airport coach service', 'Bus/coach company unknown']

        return np.where(
            (row['Last'] in ['Private car - driven away', 'Chauffer']) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            36,  
            0 
        )    
    
    def condition_37(self, row):
        included_2ndlast_modes = self.local_bus + ['London bus companies']

        return np.where(
            (row['Last'] in ['Private car - driven away', 'Chauffer']) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            37,
            0
        )

    # condition_38 not coded as it is specific to STN
    def condition_38(self, row):
        return 0
    
    def condition_39(self, row):
        return 0
    
    def condition_40(self, row):
        return 0
       
    def condition_41(self, row):
        return 0

    def condition_42(self, row):
        return np.where(
            # Cannot do this entire block as we do not have trip_total_days
            # Taken (row['Last'] in ['Private car - short term car park', 'Private car - short term car park - meet/greet']) out to ensure the condition still works correctly
            # (
            #     ((row['Last'] in ['Private car - short term car park', 'Private car - short term car park - meet/greet']) & (row['trip_total_days'] > 1)) | 
            #     ((row['Last'] in ['Private car - short term car park', 'Private car - short term car park - meet/greet']) & (row['Segment_4_ID'] > 2) & (row['trip_total_days'] <= 1))
            # ) & 
            (row['Last'] in ['Private car - short term car park', 'Private car - short term car park - meet/greet']) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in self.tube),
            42,
            0
        )
    
    def condition_43(self, row):
        return 0
    
    def condition_44(self, row):
        included_2ndlast_modes = self.national_coach + ['Airport to airport coach service', 'Bus/coach company unknown']

        return np.where(
            (row['Last'] in ['Private car - short term car park', 'Private car - short term car park - meet/greet']) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            44,
            0
        )

    def condition_45(self, row):
        return 0

    def condition_46(self, row):
        included_2ndlast_modes = self.local_bus + ['London bus companies']

        return np.where(
            (row['Last'] in ['Private car - short term car park', 'Private car - short term car park - meet/greet']) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            46,
            0
        )

    def condition_47(self, row):
        return 0

    def condition_48(self, row):
        return 0

    def condition_49(self, row):
        included_modes = ['Private car - valet service - Off airport', 'Private car - valet service - On airport', 'Private car - airport long term car park bus',
                          'Private car - private long term car park bus', 'Private car - business car park', 'Private car - mid stay car park bus',
                          'Private car - staff car park bus', 'Private car - hotel car park bus', 'Private car - type of car park unknown'
                          ]

        return np.where(
            (row['AIRPORT_Prefix'] == 'LHR') & 
            (row['Last'] in included_modes) &
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in self.tube),
            49,
            0
        )    

    def condition_50(self, row):
        return 0

    def condition_51(self, row):
        included_modes = ['Private car - valet service - Off airport', 'Private car - valet service - On airport', 'Private car - airport long term car park bus',
                          'Private car - private long term car park bus', 'Private car - business car park', 'Private car - mid stay car park bus',
                          'Private car - staff car park bus', 'Private car - hotel car park bus', 'Private car - type of car park unknown'
                          ]
        
        included_2ndlast_modes = self.national_coach + ['Airport to airport coach service', 'London bus companies']
        
        return np.where(
            (row['Last'] in included_modes) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            51,  
            0
        )

    def condition_52(self, row):
        included_last_modes = ['Private car - valet service - Off airport', 'Private car - valet service - On airport', 'Private car - airport long term car park bus',
                               'Private car - private long term car park bus', 'Private car - business car park', 'Private car - mid stay car park bus',
                               'Private car - staff car park bus', 'Private car - hotel car park bus', 'Private car - type of car park unknown'
                               ]
        
        included_2ndlast_modes = self.local_bus + ['London bus companies']

        return np.where(
            (row['Last'] in included_last_modes) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            52,
            0
        )

    def condition_53(self, row):
        return 0
    
    def condition_54(self, row):
        return 0

    def condition_55(self, row):
        return 0
    
    def condition_56(self, row):
        return 0

    def condition_57(self, row):
        return np.where(
            (row['Last'] == 'Taxi') & 
            (row['Origin'] == 'NonLDN') &
            (row['2ndLast'] not in ['Heathrow Express', 'RailAir Bus (Reading/Woking/Feltham)']),
            57,
            0
        )
    
    def condition_58(self, row):
        return np.where(
            (row['Last'] == 'Taxi') & 
            (row['2ndLast'] in ['Private car - driven away', 'Chauffer']) & 
            (row['Origin'] == 'NonLDN'),
            58,
            0
        )

    def condition_59(self, row):
        return np.where(
            (row['Last'] == 'Taxi') & 
            (row['2ndLast'] == 'Private car - hotel car park bus') & 
            (row['Origin'] == 'NonLDN'),
            59,  
            0  
        )

    def condition_60(self, row):
        return 0

    def condition_61(self, row):
        return np.where(
            (row['Last'] == 'Taxi') & 
            (
                (row['2ndLast'] in self. national_coach) | 
                (row['2ndLast'] == 'Airport to airport coach service') | 
                (row['2ndLast'] == 'Bus/coach company unknown')
            ) & 
            (row['Origin'] == 'NonLDN'),
            61,  
            0   
        )

    def condition_62(self, row):
        included_2ndlast_modes = self.local_bus + ['London bus companies']

        return np.where(
            (row['Last'] == 'Taxi') & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            62,  
            0    
        )

    def condition_63(self, row):
        return 0  # Always returns 0

    def condition_64(self, row):
        return 0  # Always returns 0

    def condition_65(self, row):
        return np.where(
            (row['Last'] in self.minicab) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] not in ['Heathrow Express', 'RailAir Bus (Reading/Woking/Feltham)']),
            65,
            0
        )
    
    def condition_66(self, row):
        return 0    

    def condition_67(self, row):
        return np.where(
            (row['Last'] in self.minicab) & 
            (row['2ndLast'] in ['Private car - hotel car park bus', 'Private car - private long term car park bus']) & 
            (row['Origin'] == 'NonLDN'),
            67,  
            0   
        )

    def condition_68(self, row):
        return np.where(
            (row['Last'] in self.minicab) & 
            (row['2ndLast'] in ['Rental car - short term car park', 'Rental car - hire car courtesy bus']) & 
            (row['Origin'] == 'NonLDN') & 
            (row['3rdLast'] == 'No Mode'),
            68, 
            0 
        )

    def condition_69(self, row):
        return np.where(
            (row['Last'] in self.minicab) & 
            (
                (row['2ndLast'] in self.national_coach) | 
                (row['2ndLast'] in ['Airport to airport coach service', 'Bus/coach company unknown'])
            ) & 
            (row['Origin'] == 'NonLDN'),
            69,  #
            0 
        )

    def condition_70(self, row):
        included_2ndlast_modes = self.local_bus + ['London bus companies']

        return np.where(
            (row['Last'] in self.minicab) & 
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] in included_2ndlast_modes),
            70,
            0 
        )

    def condition_71(self, row):
        return 0

    def condition_72(self, row):
        # combined condition 72 with 15
        return 0

    # Not included as it is specific to LGW
    def condition_73(self, row):
        return 0
    
    # Not included as it is specific to STN
    def condition_74(self, row):
        return 0    

    def condition_75(self, row):
        return 0   

    def condition_76(self, row):
        return 0

    def condition_77(self, row):
        return 0

    def condition_78(self, row):
        return 0

    def condition_79(self, row):
        # No included as Demand Responsive Coach doesn't exist
        return 0

    def condition_80(self, row):
        # Formely Heathrow Connect
        return 0

    def condition_81(self, row):
        # Formely Heathrow Connect
        return 0
    
    def condition_82(self, row):
        # Formely Heathrow Connect
        return 0

    def condition_83(self, row):
        # Formely Heathrow Connect
        return 0
    
    def condition_84(self, row):
        # Formely Heathrow Connect
        return 0
    
    def condition_85(self, row):
        # Formely Heathrow Connect
        return 0

    def condition_86(self, row):
        return np.where(
            (row['Last'] in self.tube) & 
            (row['3rdLast'] in self.national_coach) & 
            (row['Origin'] == 'NonLDN'),
            86,  
            0    
        )

    def condition_87(self, row):
        return 0
    
    # Not included as it does not apply to LHR
    def condition_88(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_89(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_90(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_91(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_92(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_93(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_94(self, row):
        return 0
    
    # Not included as it does not apply to LHR
    def condition_95(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_96(self, row):
        return 0

    # Not included as it does not apply to LHR
    def condition_97(self, row):
        return 0

    # Not included as it requires the column 'trip_total_days' which we do not have
    def condition_98(self, row):
        # return np.where(
        #     (
        #         ((row['AIRPORT_Prefix'] == 'LHR') & (row['trip_total_days'] > 1) & (row['Segment_4_ID'] < 3)) | 
        #         ((row['AIRPORT_Prefix'] != 'LHR') & (row['trip_total_days'] > 2) & (row['Segment_4_ID'] < 3))
        #     ),
        #     98,
        #     0
        # )
        return 0
    
    def condition_99(self, row):
        return np.where(
            (row['Last'] == 'Car Unspecified') & 
            (row['Segment_4_ID'] > 2),
            99,  
            0    
        )

    def condition_100(self, row):
        return np.where(
            (row['Last'] == 'Car Unspecified') & 
            # Not included as it requires the column 'trip_total_days' which we do not have
            # (
            #     ((row['AIRPORT_Prefix'] == 'LHR') & (row['trip_total_days'] <= 1) & (row['trip_total_days'] > 0) & (row['Segment_4_ID'] < 3)) | 
            #     ((row['AIRPORT_Prefix'] != 'LHR') & (row['trip_total_days'] <= 2) & (row['trip_total_days'] > 0) & (row['Segment_4_ID'] < 3))
            # ),
            (row['Segment_4_ID'] < 2),
            100,  # Value if condition is met
            0     # Value if condition is not met
        )

    def condition_101(self, row):
        return 0

    def condition_102(self, row):
        return 0

    def condition_103(self, row):
        return np.where(
            (row['Last'] == 'Taxi/Minicab Unspecified'),
            103,  
            0  
        )
    
    def condition_104(self, row):
        return np.where(
            (row['Last'] == 'Bus Unspecified'),
            104,  # Value if condition is met
            0     # Value if condition is not met
        )
    
    def condition_105(self, row):
        return np.where(
            (row['Last'] == 'Car Unspecified') & 
            # Not included as it requires the column 'trip_total_days' which we do not have
            # (
            #     (row['trip_total_days'] == 0) | 
            #     (pd.isnull(row['trip_total_days']))
            # ) & 
            (row['Segment_4_ID'] < 3),
            105,  
            0     
        )

    def condition_106(self, row):
        included_last_modes = self.national_railways + ['Rail Unspecified']
        
        return np.where(
            (row['AIRPORT_Prefix'] == 'LHR') & 
            (row['Last'] in included_last_modes),
            106, 
            0   
        )

    def condition_107(self, row):
        return 0
    
    def condition_108(self, row):
        return np.where(
            (row['Last'] == 'Elizabeth Line') &
            (row['2ndLast'] != 'Heathrow Express'),
            108,
            0
        )    

    def condition_109(self, row):
        # K&F, P&F or Taxi modes
        included_last_modes = (
            self.minicab + self.tube + 
            ['Private car - driven away', 'Chauffer','Private car - short term car park', 'Private car - short term car park - meet/greet',
            'Private car - valet service - Off airport', 'Private car - valet service - On airport', 'Private car - airport long term car park bus',
            'Private car - private long term car park bus', 'Private car - business car park', 'Private car - mid stay car park bus', 
            'Private car - staff car park bus', 'Private car - hotel car park bus', 'Private car - type of car park unknown'
            'Taxi']
            )
        
        return np.where(
            (row['Last'] in included_last_modes) &
            (row['Origin'] == 'NonLDN') & 
            (row['2ndLast'] == 'Elizabeth Line'),
            109, 
            0   
        )

    def apply_conditions(self, dataframe):
        df = dataframe.copy()

        # Temporary storage for all new columns
        condition_columns = {}
        
        # Loop through all conditions dynamically
        for i in range(1, self.number_of_conditions + 1):
            condition_method = getattr(self, f'condition_{i}')
            # Compute the condition using apply and store in dictionary
            condition_columns[f'Condition_{i}'] = df.apply(condition_method, axis=1)
        
        # Add all new columns to the DataFrame at once using pd.concat
        df = pd.concat([df, pd.DataFrame(condition_columns)], axis=1)
        
        return df
    
    def mode_process_check(self, dataframe):
        df = dataframe.copy()

        # Dataframe for condition columns only
        condition_df = df[self.condition_columns]

        # Count the non-zero values in these columns for each row
        non_zero_count_conditions = (condition_df != 0).sum(axis=1)
        condition_sum = (condition_df).sum(axis=1)

        df['Conditions Met'] = non_zero_count_conditions
        df['Condition Sum'] = condition_sum

        # def process_assignment_status(row):
        #     return np.where(
        #         (row['Conditions Met'] == 1), 
        #         'Correctly Assigned',
        #         np.where(
        #             (row['Conditions Met'] > 0),
        #             'Duplicates Assigned',
        #             np.where(
        #                 (row['Conditions Met'] == 0) &
        #                 (   
        #                     (row['SYSTEM_District'] != 'Heathrow Airport (SE)') &
        #                     (
        #                         (row['Last'] != 'No Mode') |
        #                         (row['2ndLast'] != 'No Mode') | 
        #                         (row['3rdLast'] != 'No Mode')
        #                     )    
        #                 ),
        #                 'Not assigned - Logic',
        #                 'Not Assigned - Data'
                        
        #             )
        #         ),
        #     )

        def process_assignment_status(row):
            return np.where(
                (row['Conditions Met'] == 1), 
                'Correctly Assigned',
                np.where(
                    (row['Conditions Met'] > 0),
                    'Duplicates Assigned',
                    np.where(
                        (row['Conditions Met'] == 0) &
                        (   
                            (row['Last']=='Airport to airport coach service') |
                            (row['Last'] == 'No Mode')   
                        ),
                        'Not Assigned - Data',
                        'Not Assigned - Logic'
                    )
                ),
            )

        df['Mode Process Check'] = df.apply(process_assignment_status, axis=1)
        df['Mode Process Check'] = df['Mode Process Check'].astype(str)

        return df
    
    def get_condition_id(self, dataframe):
        df = dataframe.copy()

        # Assign condition ID's to the correctly assigned rows
        df_correctly_assigned = df[df['Mode Process Check']=='Correctly Assigned'].copy()
        df_correctly_assigned['Condition ID'] = df_correctly_assigned['Condition Sum']
        
        # Process and assigned condition ID's to rows with duplicate modes
        # The process looks at all the conditions that passed then return the condition ID with the highest priority (1=highest priority)
        df_duplicates_assigned = df[df['Mode Process Check']=='Duplicates Assigned'].copy()
        
        # Function to determine the lowest priority LASAM mode
        def get_lowest_priority_mode_condition(row, condition_columns):
            # Filter condition columns in the row
            condition_values = row[condition_columns]
            
            # Extract non-zero conditions from the filtered columns
            non_zero_conditions = {col: condition_values[col] for col in condition_values.index if condition_values[col] != 0}
            
            if non_zero_conditions:
                # Convert non-zero values to a dataframe for merging with lookup_df
                conditions_df = pd.DataFrame({"Condition ID": list(non_zero_conditions.values())}, dtype=int)
                # Merge with the lookup dataframe to get priority and LASAM_mode
                merged_df = conditions_df.merge(self.mode_condition_lu, on="Condition ID", how="left")
                # Get the row with the lowest priority
                lowest_priority_row = merged_df.loc[merged_df["LASAM Mode Priority"].idxmin()]
                # Return the corresponding LASAM_mode
                return lowest_priority_row["Condition ID"]
            else:
                return -1

        # Apply the function row-wise
        df_duplicates_assigned["Condition ID"] = df_duplicates_assigned.apply(lambda row: get_lowest_priority_mode_condition(row, self.condition_columns), axis=1)
        
        df_not_assigned = df[df['Mode Process Check'].isin(['Not Assigned - Data', 'Not Assigned - Logic'])].copy()
        df_not_assigned['Condition ID'] = -1

        df_condition_id = pd.concat([df_correctly_assigned, df_duplicates_assigned, df_not_assigned], ignore_index=True)

        return df_condition_id
    
    def assign_lasam_mode(self, dataframe):
        df = dataframe.copy()

        df = df.merge(self.mode_condition_lu, on='Condition ID', how='left')

        return df
    
    def update_lasam_mode_using_final_mode(self, dataframe):
        df = dataframe.copy()

        df.loc[
            df['Mode Process Check'] == 'Not Assigned - Logic',
            ['LASAM Mode', 'LASAM Mode Code']
        ] = df.loc[
            df['Mode Process Check'] == 'Not Assigned - Logic',
            ['SYSTEM_FINALMODE_LASAM_Mode', 'SYSTEM_FINALMODE_LASAM_Mode_Code']
        ].values

        return df

    def main_mode_condition_mapping(self):

        # Step 1: apply conditions
        self.df = self.apply_conditions(self.df)

        # Step 2: mode process check
        self.df = self.mode_process_check(self.df)

        # Step 3: get condition ID
            # If there is only 1 condition then the condition ID is the value of the ID
            # If more than one condition was met then we need to run a separate function to identify the condition with the highest priority
            # If no conditions where met then the condition ID is -1.
        self.df = self.get_condition_id(self.df)

        # Step 4: assign LASAM main mode and mode based on the condition ID
        self.df = self.assign_lasam_mode(self.df)

        # Step 5: assign LASAM mode based on system final mode where there is a logic error
        self.df = self.update_lasam_mode_using_final_mode(self.df)

        return self.df
    
    def main_run_all(self):
        df_mode_mapped = self.main_mode_condition_mapping()

        # drop condition columns
        df_mode_mapped.drop(columns=self.condition_columns, inplace=True)

        return df_mode_mapped


