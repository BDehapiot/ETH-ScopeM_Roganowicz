inputs = {

#%% 2025-03_mutants_norfloxacin -----------------------------------------------

"2025-03_mutants_norfloxacin" : {
    
    "parameters" : {
        
        "lmax_dist" : 4,
        "lmax_prom" : 0.6,
        "C2_dog_sigma1" : 1,
        "C2_dog_sigma2" : 8,
        "C2_dog_thresh" : 1,
        "C2_min_area" : 32,
        "C2_min_mean_int" : 24,
        "C2_min_mean_edt" : 20,
        
        },

    "plate_mapping" : {
        
        'p1_r1_2025-03-12_b2_control'          : 'control 0.1% DMSO',
        'p2_r1_2025-03-12_b2_norfloxacin-0002' : 'norfloxacin 0.002 µg/ml',
        'p3_r1_2025-03-12_b2_norfloxacin-0008' : 'norfloxacin 0.008 µg/ml',
        'p4_r1_2025-03-12_b2_norfloxacin-0032' : 'norfloxacin 0.032 µg/ml',
            
        'p1_r2_2025-03-13_b2_control'          : 'control 0.1% DMSO',
        'p2_r2_2025-03-13_b2_norfloxacin-0002' : 'norfloxacin 0.002 µg/ml',
        'p3_r2_2025-03-13_b2_norfloxacin-0008' : 'norfloxacin 0.008 µg/ml',
        'p4_r2_2025-03-13_b2_norfloxacin-0032' : 'norfloxacin 0.032 µg/ml',
                
        'p1_r3_2025-03-17_b2_control'          : 'control 0.1% DMSO',
        'p2_r3_2025-03-17_b2_norfloxacin-0002' : 'norfloxacin 0.002 µg/ml',
        'p3_r3_2025-03-17_b2_norfloxacin-0008' : 'norfloxacin 0.008 µg/ml',
        'p4_r3_2025-03-17_b2_norfloxacin-0032' : 'norfloxacin 0.032 µg/ml',
    
        },
    
    "well_mapping" : {
        
        'A01': 'parental', 'A02': 'ΔfimH', 'A03': 'ΔmotA', 
        'A04': 'ΔmotB'   , 'A05': 'ΔfliA', 'A06': 'ΔfliC',
        'B01': 'ΔcsgA'   , 'B02': 'ΔcsgB', 'B03': 'Δkps' ,
        'B04': 'Δneu'    , 'B05': 'Δgspl', 'B06': 'ΔyhjM',    
        'C01': 'Δfiu'    , 'C02': 'ΔnikA', 'C03': 'Δyhjk', 
        'C04': 'ΔsfaA'   , 'C05': 'ΔyeaP', 'C06': 'ΔyagX',
        'D01': 'ΔluxS'   , 'D02': 'ΔcheA', 'D03': 'Δtsr' , 
        'D04': 'ΔrfaQ'   , 'D05': 'ΔyaiW', 'D06': 'ΔompW',
        
        },
    
    },
        
#%% 2025-04_mutants_nitrofurantoin --------------------------------------------

"2025-04_mutants_nitrofurantoin" : {

    "parameters" : {
        
        "lmax_dist" : 4,
        "lmax_prom" : 0.6,
        "C2_dog_sigma1" : 1,
        "C2_dog_sigma2" : 8,
        "C2_dog_thresh" : 1,
        "C2_min_area" : 32,
        "C2_min_mean_int" : 24,
        "C2_min_mean_edt" : 20,
        
        },    

    "plate_mapping" : {
        
        'p1_r1_2025-04-03_b2_control'             : 'control 0.1% DMSO',
        'p2_r1_2025-04-03_b2_nitrofurantoin-0004' : 'nitrofurantoin 0.004 µg/ml',
        'p3_r1_2025-04-03_b2_nitrofurantoin-0064' : 'nitrofurantoin 0.064 µg/ml',
        'p4_r1_2025-04-03_b2_nitrofurantoin-1000' : 'nitrofurantoin 1.000 µg/ml',
            
        'p1_r2_2025-04-10_b2_control'             : 'control 0.1% DMSO',
        'p2_r2_2025-04-10_b2_nitrofurantoin-0004' : 'nitrofurantoin 0.004 µg/ml',
        'p3_r2_2025-04-10_b2_nitrofurantoin-0064' : 'nitrofurantoin 0.064 µg/ml',
        'p4_r2_2025-04-10_b2_nitrofurantoin-1000' : 'nitrofurantoin 1.000 µg/ml',
                
        'p1_r3_2025-04-22_b2_control'             : 'control 0.1% DMSO',
        'p2_r3_2025-04-22_b2_nitrofurantoin-0004' : 'nitrofurantoin 0.004 µg/ml',
        'p3_r3_2025-04-22_b2_nitrofurantoin-0064' : 'nitrofurantoin 0.064 µg/ml',
        'p4_r3_2025-04-22_b2_nitrofurantoin-1000' : 'nitrofurantoin 1.000 µg/ml',
    
        },

    "well_mapping" : {
        
        'A01': 'parental', 'A02': 'ΔfimH', 'A03': 'ΔmotA', 
        'A04': 'ΔmotB'   , 'A05': 'ΔfliA', 'A06': 'ΔfliC',
        'B01': 'ΔcsgA'   , 'B02': 'ΔcsgB', 'B03': 'Δkps' ,
        'B04': 'Δneu'    , 'B05': 'Δgspl', 'B06': 'ΔyhjM',    
        'C01': 'Δfiu'    , 'C02': 'ΔnikA', 'C03': 'Δyhjk', 
        'C04': 'ΔsfaA'   , 'C05': 'ΔyeaP', 'C06': 'ΔyagX',
        'D01': 'ΔluxS'   , 'D02': 'ΔcheA', 'D03': 'Δtsr' , 
        'D04': 'ΔrfaQ'   , 'D05': 'ΔyaiW', 'D06': 'ΔompW',
        
        },
    
    },
    
#%% 2025-09_parental_yhjC -----------------------------------------------------
    
"2025-09_parental_yhjC" : {
    
    "parameters" : {
        
        "lmax_dist" : 4,
        "lmax_prom" : 0.6,
        "C2_dog_sigma1" : 1,
        "C2_dog_sigma2" : 8,
        "C2_dog_thresh" : 0.5, # changed parameter
        "C2_min_area" : 32,
        "C2_min_mean_int" : 24,
        "C2_min_mean_edt" : 20,
        
        },    
    
    "plate_mapping" : {
        
        'p1_r1_2025-09-01_b2_control' : 'control',
        'p1_r2_2025-09-01_b2_control' : 'control',
        'p1_r3_2025-09-01_b2_control' : 'control',

        },
    
    "well_mapping" : {
        
        'B02': 'parental', 'B03': 'parental', 'B04': 'parental',
        'C02': 'ΔyhjC'   , 'C03': 'ΔyhjC'   , 'C04': 'ΔyhjC'   ,
        
        },
    
    },
    
#%% 2026-05_bcsA_complemented_strains -----------------------------------------

"2026-05_bcsA_complemented_strains" : {
    
    "parameters" : {
        
        "lmax_dist" : 4,
        "lmax_prom" : 0.6,
        "C2_dog_sigma1" : 1,
        "C2_dog_sigma2" : 8,
        "C2_dog_thresh" : 1,
        "C2_min_area" : 32,
        "C2_min_mean_int" : 24,
        "C2_min_mean_edt" : 20,
        
        },    
    
    "plate_mapping" : {
        
        'p1_r1_2026-05-21_bcsA_b2_complemented_strains' : 'control',

        },
    
    "well_mapping" : {
        'A01': 'parental', 'A02': 'parental', 'A03': 'parental',
        'B01': 'ΔbcsA'   , 'B02': 'ΔbcsA'   , 'B03': 'ΔbcsA'   ,
        'B04': '+yhjK'   , 'B05': '+yhjK'   , 'B06': '+yhjK'   ,
        'C01': 'empty'   , 'C02': 'empty'   , 'C03': 'empty'   ,
        'C04': '+yhjM'   , 'C05': '+yhjM'   , 'C06': '+yhjM'   ,
        
        },
    
    },

#%% ??? -----------------------------------------------------------------------

"???" : {
    
    "parameters" : {
        
        "lmax_dist" : 4,
        "lmax_prom" : 0.6,
        "C2_dog_sigma1" : 1,
        "C2_dog_sigma2" : 8,
        "C2_dog_thresh" : 1,
        "C2_min_area" : 32,
        "C2_min_mean_int" : 24,
        "C2_min_mean_edt" : 20,
        
        },    
    
    "plate_mapping" : {
        
        '???' : '???',

        },
    
    "well_mapping" : {
        'A01': 'parental', 'A02': 'parental', 'A03': 'parental',
        'A04': 'ΔbcsA'   , 'A05': 'ΔbcsA'   , 'A06': 'ΔbcsA'   ,
        'B01': 'ΔyhjK'   , 'B02': 'ΔyhjK'   , 'B03': 'ΔyhjK'   ,
        'B04': 'ΔfimH'   , 'B05': 'ΔfimH'   , 'B06': 'ΔfimH'   ,
        'C01': 'ΔyhjM'   , 'C02': 'ΔyhjM'   , 'C03': 'ΔyhjM'   ,
        'C04': 'ΔcsgA'   , 'C05': 'ΔcsgA'   , 'C06': 'ΔcsgA'   ,
        
        },
    
    },

#%%

}