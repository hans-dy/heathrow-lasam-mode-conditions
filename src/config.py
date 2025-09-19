import pandas as pd

#################
##### PATHS #####
#################
MAIN_DIR= r'\\GBLON7VS01.europe.jacobs.com\Projects\UNIF\Projects\60H700SA - Heathrow SAS 2024\04 Technical\03 LASAM Development\2024 Base Mtx\Matrix Development'
DATA_DIR = rf'{MAIN_DIR}\02_data'
LOOKUP_DIR = rf'{MAIN_DIR}\03_lookups'

###################
##### LOOKUPS #####
###################

caa_final_mode_lasam_mode_lu = pd.read_csv(rf'{LOOKUP_DIR}\caa_final_mode_lasam_mode_lu.csv')
cube_segment_mode_index_lu = pd.read_csv(rf'{LOOKUP_DIR}\cube_segment_mode_index_lu.csv')
lasam_zone_district_lu = pd.read_csv(rf'{LOOKUP_DIR}\lasam_zone_district_lu.csv')
segment_lu = pd.read_csv(rf'{LOOKUP_DIR}\segment_lu.csv')
caa_mode_allocation_lasam_mode_lu = pd.read_csv(rf'{DATA_DIR}\mode_conditions\version2\caa_mode_allocation_lasam_mode_lu.csv')