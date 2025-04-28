import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import shutil
import pandas as pd
import numpy as np
from octolyzer import key_descriptions

KEY_LAYER_DICT = {"ILM": "Inner Limiting Membrane",
                  "RNFL": "Retinal Nerve Fiber Layer",
                  "GCL": "Ganglion Cell Layer",
                  "IPL": "Inner Plexiform Layer",
                  "INL": "Inner Nuclear Layer",
                  "OPL": "Outer Plexiform Layer",
                  "ELM": "External Limiting Membrane", # Outer nuclear layer
                  "PR1": "Photoreceptor Layer 1",
                  "PR2": "Photoreceptor Layer 2",
                  "RPE": "Retinal Pigment Epithelium",
                  "BM": "Bruch's Membrane Complex", 
                  "CHORupper": "Bruch's Membrane - Choroid boundary",
                  "CHORlower": "Choroid - Sclera boundary"}

# Load in previously analysed results (if saved out)
def load_files(fname_path, logging_list=[], analyse_square=0, only_slo=0, only_oct = 0, verbose=1):
    """
    Loads previously analysed SLO and OCT measurement data for a previously analysed file. 

    This function attempts to retrieve and collate measurement data saved as an Excel file
    in the save directory.

    Parameters:
    ----------
    fname_path : str or pathlib.Path
        The directory path where the analyzed results are stored for the file.
    logging_list : list, optional
        A list to append log messages about the loading process. Defaults to an empty list.

    Returns:
    -------
    meta_df : pandas.DataFrame
        Metadata associated with the analysed file, loaded from the "metadata" sheet.
    slo_dfs : list of pandas.DataFrame
        A list of DataFrames containing SLO measurement data. The content varies 
        depending on the scan location type (e.g., "Macular" or "Peripapillary").
    oct_dfs : list of pandas.DataFrame
        A list of DataFrames containing OCT measurement data, with sheet names 
        depending on the B-scan pattern.
    logging_list : list
        Updated list of log messages, including success or error information.
    """
    msg = "Loading in measurements..."
    if verbose:
        print(msg)
    logging_list.append(msg)
    fname = os.path.split(fname_path)[-1]
    output_fname = os.path.join(fname_path, f"{fname}_output.xlsx")

    # Load metadata
    meta_df = pd.read_excel(output_fname, sheet_name="metadata")
    outputs = [meta_df]
    oct_dfs = []
    slo_dfs = []

    # Load in SLO measurements
    if not only_oct:
        if meta_df.bscan_type.iloc[0] == "Peripapillary":
            for r in ["B", "C", "whole"]:
                try:
                    df = pd.read_excel(output_fname, sheet_name=f"slo_measurements_{r}")
                except:
                    df = pd.DataFrame()
                slo_dfs.append(df)
        else:
            try:
                df = pd.read_excel(output_fname, sheet_name=f"slo_measurements_whole")
            except:
                df = pd.DataFrame()
            slo_dfs.append(df)
        outputs.append(slo_dfs)

    # Load in OCT measurements
    if not only_slo:
        if meta_df.bscan_type.iloc[0] == "Ppole":
            if analyse_square:
                keys = ['etdrs_measurements', 'etdrs_volume_measurements', 'square_measurements', 'square_volume_measurements']
            else:
                keys = ['etdrs_measurements', 'etdrs_volume_measurements']
        else:
            keys = ['oct_measurements']
        for key in keys:
            try:
                df = pd.read_excel(output_fname, sheet_name=key)
                oct_dfs.append(df)
            except:
                if verbose:
                    print(f'{key} does not exist. Skipping.')
        outputs.append(oct_dfs)
    
    msg = "Successfully loaded in all measurements!\n"
    if verbose:
        print(msg)
    logging_list.append(msg)
    outputs.append(logging_list)

    return outputs



def collate_results(result_dict, save_directory, analyse_choroid=1, analyse_slo=0):
    """
    Consolidate all analysis results for SLO and OCT data into a unified Excel output file
    `analysis_output.xlsx` in `save_directory`.

    This function processes individual result dictionaries containing metadata, SLO, 
    and OCT measurements. It flattens and organizes the data into global DataFrames, 
    saves results into an Excel file, and writes associated logs for each analyzed 
    dataset.

    Parameters:
    ----------
    result_dict : dict
        A dictionary where keys represent filenames and values contain analysis outputs 
        with the following structure:
            - 'metadata': pandas.DataFrame with metadata about the scan.
            - 'slo': list of pandas.DataFrames for SLO measurements (or empty DataFrame if missing).
            - 'oct': list of pandas.DataFrames for OCT measurements, or a string if analysis failed.
            - 'log': list of strings representing log messages for the corresponding file.

    save_directory : str or pathlib.Path
        The directory path where the consolidated Excel output and log files will be saved.

    analyse_choroid : int, optional, default=1
        Flag to include choroid-related measurements (1 to include, 0 to exclude).

    analyse_slo : int, optional, default=0
        Flag to indicate whether SLO analysis is included (1 to include, 0 to ignore).

    Returns:
    -------
    None, as it only saves results out.

    Outputs:
    -------
    - Consolidated Excel file `analysis_output.xlsx` in `save_directory` containing:
        - Metadata ('metadata' sheet).
        - SLO measurements ('SLO_measurements' sheet, if available).
        - OCT measurements categorized by scan type:
            - Linescan measurements ('OCT_Linescan_measurements').
            - Radial scan measurements ('OCT_Radial_measurements').
            - Peripapillary measurements ('OCT_Peripapillary_measurements').
            - Posterior pole measurements ('OCT_Ppole_ETDRS_measurements', 'OCT_Ppole_Square_measurements').
        - Metadata key and descriptions ('metadata_keys' sheet).
    - Individual log files for each analyzed dataset and a global `analysis_log.txt` file.

    Notes:
    -----
    - SLO data is flattened and arranged based on academic literature-relevant feature order.
    - OCT data is processed differently for each scan type (e.g., peripapillary, radial, posterior pole).
    - Missing or failed analyses are logged, and placeholder DataFrames are created if necessary.
    - The function handles cases where specific measurement types or layers (e.g., choroid) 
      are excluded based on input flags.

    Examples:
    --------
    >>> result_dict = {
    ...     "file1.vol": {
    ...         "metadata": pd.DataFrame(...),
    ...         "slo": [pd.DataFrame(...)],
    ...         "oct": [pd.DataFrame(...)],
    ...         "log": ["Processing file1"]
    ...     },
    ...     "file2.vol": {
    ...         "metadata": pd.DataFrame(...),
    ...         "slo": [pd.DataFrame(...)],
    ...         "oct": "Error: Missing data",
    ...         "log": ["Error in file2 processing"]
    ...     }
    ... }
    >>> collate_results(result_dict, "output_directory")
    """
    
    print(f"Collecting all results together into one output file.")

    # Initialise all empty measurement DataFrames and list of logs to save out.
    all_logging_list = []
    all_slo_df = pd.DataFrame()
    all_ind_df = pd.DataFrame()
    all_oct_linescan_df = pd.DataFrame()
    all_oct_ppole_df = pd.DataFrame()
    all_oct_sq_ppole_df = pd.DataFrame()
    all_oct_peri_df = pd.DataFrame()
    all_oct_radial_df = pd.DataFrame()
    for fname_type, output in result_dict.items():
        
        fname = fname_type.split(".")[0]
        all_logging_list.append(f"\n\nANALYSIS LOG OF {fname_type}")

        # Create default dataframe for failed individuals and save out error
        if isinstance(output['oct'], str):
            log = output['log']
            metadata = output['metadata']
            slo_dfs = pd.DataFrame()
            oct_dfs = pd.DataFrame()
            fname_path = os.path.join(save_directory, fname)
            all_logging_list.extend(log)    
            with open(os.path.join(fname_path, f"{fname}_log.txt"), "w") as f:
                for line in log:
                    f.write(line+"\n")
            ind_df = pd.DataFrame(metadata, index=[0])
        
        # Otherwise, collate measurements and save out segmentations if specified
        else:    
            
            ind_df = output['metadata']
            bscan_type = ind_df.bscan_type.iloc[0]
            slo_dfs = output['slo']
            oct_dfs = output['oct']
            N_oct = len(oct_dfs)
            logging_list = output['log']

            # If cannot find measurements/etadata, create default dataframe and bypass
            # the segmentation visualisation
            all_logging_list.extend(logging_list)

            # process SLO measurement DFs and save out global dataframe
            flat_slo_df = pd.DataFrame()
            if len(slo_dfs[0]) > 0:
                rtypes = []
                for df in slo_dfs:
                    if len(df) == 0:
                        continue
                    # flatten
                    dfarr = df.values.flatten()
    
                    # Collect all columns in flattened DF
                    dfcols = list(df.columns)
                    rtype = df.zone.iloc[0]
                    rtypes.append(rtype)
                    vtypes = df.vessel_map.values
                    dfcols = [col+f"_{vtype}_{rtype}" for vtype in vtypes for col in dfcols]
                    df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])
    
                    # Remove indexing columns and concatenate with different zones
                    cols_to_drop = df_flatten.columns[df_flatten.columns.str.contains("vessel_map|zone")]
                    df_flatten = df_flatten.drop(cols_to_drop, axis=1, inplace=False)
                    flat_slo_df = pd.concat([flat_slo_df, df_flatten], axis=1)    

                # Order feature columns by importance in literature
                order_str = ["fractal_dimension", "vessel_density", "tortuosity_density", 'average_global_calibre', 
                             'average_local_calibre', 'CRAE_Knudtson', 'CRVE_Knudtson', 'AVR']
                order_str_rv = [col+f"_{vtype}_{rtype}" 
                                    for rtype in rtypes[::-1] 
                                        for vtype in vtypes for col in order_str]
                flat_slo_df = flat_slo_df[order_str_rv]
                flat_slo_df = flat_slo_df.loc[:, ~(flat_slo_df == -1).any()]
                flat_slo_df = flat_slo_df.rename({f"AVR_artery-vein_{rtype}":f"AVR_{rtype}" 
                                                  for rtype in rtypes}, inplace=False, axis=1)

                # Concatenate measurements with metadata filename
                ind_slo_df = pd.concat([ind_df['Filename'], flat_slo_df], axis=1)

            # process OCT measurement DFs and save out global dataframe
            flat_oct_df = pd.DataFrame()
            flat_sq_oct_df = pd.DataFrame()

            # For Peripapillary, H-line/V-line and Radial scans
            if bscan_type != 'Ppole':
                for idx, df in enumerate(oct_dfs):
                    
                    # Remove choroid measurements if not analysing choroid
                    if not analyse_choroid:
                        df = df[df['layer'] != 'CHORupper_CHORlower']

                    # Collect all layers
                    ltypes = df.layer.values
                    
                    # Flatten and collect all flattened columns through combining
                    # layers, features and potentially scan number
                    if bscan_type == "Peripapillary":
                        dfarr = df.values[:,1:].flatten()
                        dfcols = list(df.columns[1:])
                        dfcols = [f"{layer}_{col}" for layer in ltypes for col in dfcols]
                    
                    else:
                        dfarr = df.values[:,2:].flatten()
                        ltypes = df.layer.drop_duplicates().values
                        dfcols = list(df.columns[2:])
                        scan_idxs = df.scan_number.drop_duplicates().values
                        dfcols = [f"{layer}_{col}_{idx}" for layer in ltypes for idx in scan_idxs for col in dfcols]
                    
                    df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])
    
                    # Concatenate to flatten dF
                    flat_oct_df = pd.concat([flat_oct_df, df_flatten], axis=1).dropna(axis=1)
            else:

                # Loop across measurement DataFrames for Ppole features
                for idx, df in enumerate(oct_dfs[:2]):

                    # Remove choroid measurements if not analysing choroid
                    if not analyse_choroid:
                        df = df[~df['map_name'].isin(['choroid_vessel', 'choroid_CVI', 'choroid'])]
                    
                    # flatten
                    dfarr = df.values[:,2:].flatten()
                
                    # Collect all columns in flattened DF. 
                    dfcols = list(df.columns[2:])
                    mtypes = df.map_name.values
                    retinal_maps = [mtype for mtype in mtypes if "choroid" not in mtype]
                    for key in ['retina', 'inner_retina', 'outer_retina']:
                        if key in retinal_maps:
                            retinal_maps.remove(key)
                    
                    # For thickness/area, then for volume
                    if idx == 0:
                        dfcols = [col+f"_{mtype}_[um]" 
                                          if mtype not in ['choroid_CVI', 'choroid_vessel'] 
                                          else [col+f"_{mtype}", col+f"_{mtype}_[um2]"][mtype=='vessel'] 
                                    for mtype in mtypes for col in dfcols]
                    else:
                        dfcols = [col+f"_{mtype}_[mm3]" for mtype in mtypes for col in dfcols]

                    df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])

                    # Concatenate to flatten dataframe
                    flat_oct_df = pd.concat([flat_oct_df, df_flatten], axis=1).dropna(axis=1)

                # This is for Ppole scans where we have thickness and volume measurements
                if N_oct > 2:
                    for idx, df in enumerate(oct_dfs[2:]):
                        # flatten
                        dfarr = df.values[:,2:].flatten()
                    
                        # Collect all columns in flattened DF. 
                        dfcols = list(df.columns[2:])
                        mtypes = df.map_name.values
                        retinal_maps = [mtype for mtype in mtypes if "choroid" not in mtype]
                        for key in ['retina', 'inner_retina', 'outer_retina']:
                            if key in retinal_maps:
                                retinal_maps.remove(key)
                        
                        # For thickness/area, then for volume
                        if idx == 0:
                            dfcols = [col+f"_{mtype}_[um]" 
                                              if mtype not in ['choroid_CVI', 'choroid_vessel'] 
                                              else [col+f"_{mtype}", col+f"_{mtype}_[um2]"][mtype=='vessel'] 
                                        for mtype in mtypes for col in dfcols]
                        else:
                            dfcols = [col+f"_{mtype}_[mm3]" for mtype in mtypes for col in dfcols]
    
                        df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])
    
                        # Concatenate to flatten dataframe
                        flat_sq_oct_df = pd.concat([flat_sq_oct_df, df_flatten], axis=1).dropna(axis=1)
                        
            # Concatenate measurements with metadata
            ind_oct_df = pd.concat([ind_df['Filename'], flat_oct_df], axis=1)
            if N_oct > 2:
                ind_sq_oct_df = pd.concat([ind_df['Filename'], flat_sq_oct_df], axis=1)

            # Concatenate to create global dataframe of SLO results
            if len(slo_dfs[0]) > 0:
                all_slo_df = pd.concat([all_slo_df, ind_slo_df], axis=0)
                
            # Append row to global dataframe of OCT results dependent on bscan type
            if bscan_type == 'Ppole':
                all_oct_ppole_df = pd.concat([all_oct_ppole_df, ind_oct_df], axis=0)
                if N_oct > 2:
                    all_oct_sq_ppole_df = pd.concat([all_oct_sq_ppole_df, ind_sq_oct_df], axis=0)
            elif bscan_type == 'Peripapillary':
                all_oct_peri_df = pd.concat([all_oct_peri_df, ind_oct_df], axis=0)
            elif bscan_type == 'Radial':
                all_oct_radial_df = pd.concat([all_oct_radial_df, ind_oct_df], axis=0)
            else:
                all_oct_linescan_df = pd.concat([all_oct_linescan_df, ind_oct_df], axis=0)

        # Concenate metadata to global dataframe of metadata, robust to suppress NaN warnings
        # from different Bscan types having different columns
        if (len(all_ind_df)) > 0 & (len(ind_df) > 0):
            all_ind_df = pd.concat([all_ind_df.fillna(""), ind_df.fillna("")], axis=0)
        else:
            all_ind_df = ind_df.copy()

    # Reset index
    all_ind_df = all_ind_df.reset_index(drop=True)
    all_slo_df = all_slo_df.reset_index(drop=True)
    all_oct_ppole_df = all_oct_ppole_df.reset_index(drop=True)
    all_oct_sq_ppole_df = all_oct_sq_ppole_df.reset_index(drop=True)
    all_oct_peri_df = all_oct_peri_df.reset_index(drop=True)
    all_oct_linescan_df = all_oct_linescan_df.reset_index(drop=True)
    all_oct_radial_df = all_oct_radial_df.reset_index(drop=True)

    # Define columns which indicate any processing failures
    fail_cols = []
    if all_oct_linescan_df.shape[0] > 0 or all_oct_radial_df.shape[0] > 0:
        fail_cols.extend(['bscan_missing_fovea', 
                          'slo_missing_fovea', 
                          'missing_retinal_oct_measurements',
                          'missing_choroid_oct_measurements',])
    if all_oct_peri_df.shape[0] > 0:
        fail_cols.extend(['optic_disc_overlap_warning'])
    if 'FAILED' in all_ind_df.columns:
        fail_cols.append('FAILED')
    fail_df = all_ind_df[all_ind_df[fail_cols].any(axis=1)][['Filename'] + fail_cols]
    fail_df.replace(1, True, inplace=True)
    fail_df.replace(False, '', inplace=True)
    fail_df.replace(0, '', inplace=True)   

    # Remove any rows in linescan and radial dataframes and a which are just -1s, i.e. fovea was not detected
    all_oct_linescan_df = all_oct_linescan_df[~(all_oct_linescan_df.iloc[:, 1:]==-1).all(axis=1)]
    all_oct_radial_df = all_oct_radial_df[~(all_oct_radial_df.iloc[:, 1:]==-1).all(axis=1)]

    # Layer keys, ordered anaomtically
    key_df = pd.DataFrame({"key":KEY_LAYER_DICT.keys(),
                        "layer":KEY_LAYER_DICT.values()})
    if not analyse_choroid:
        key_df = key_df[~key_df.key.str.contains("CHOR")]
  
    # save out global metadata and measurements
    with pd.ExcelWriter(os.path.join(save_directory, f'analysis_output.xlsx')) as writer:
        
        # write metadata
        all_ind_df.to_excel(writer, sheet_name='metadata', index=False)

        # SLO
        if all_slo_df.shape[0] > 0:
            all_slo_df.to_excel(writer, sheet_name='SLO_measurements', index=False)
        else:
            if analyse_slo:
                print('WARNING: analyse_slo flag is 1, but there are no SLO measurements loaded!')

        # OCT measurements, save out sheets if populated
        # Linescan measurements
        if all_oct_linescan_df.shape[0] > 0:
            all_oct_linescan_df.to_excel(writer, sheet_name='OCT_Linescan_measurements', index=False)

        # Radial scan measurements
        if all_oct_radial_df.shape[0] > 0:
            all_oct_radial_df.to_excel(writer, sheet_name='OCT_Radial_measurements', index=False)

        # Ppole ETDRS measurements
        if all_oct_ppole_df.shape[0] > 0: 
            all_oct_ppole_df.to_excel(writer, sheet_name='OCT_Ppole_ETDRS_measurements', index=False)
            img_path = os.path.join(MODULE_PATH, os.path.join('figures','etdrs_posterior_pole_grid.png'))
            fname = os.path.split(img_path)[1]
            shutil.copy(img_path, os.path.join(save_directory, fname))

        # Ppole Posterior Pole Grid measurements
        if all_oct_sq_ppole_df.shape[0] > 0:
            all_oct_sq_ppole_df.to_excel(writer, sheet_name='OCT_Ppole_Square_measurements', index=False)
            img_path = os.path.join(MODULE_PATH, os.path.join('figures','square_posterior_pole_grid.png'))
            fname = os.path.split(img_path)[1]
            shutil.copy(img_path, os.path.join(save_directory, fname))

        # Peripapillary measurements
        if all_oct_peri_df.shape[0] > 0:
            all_oct_peri_df.to_excel(writer, sheet_name='OCT_Peripapillary_measurements', index=False)

        # Save out datasheet specifying any files which failed for some reason
        if fail_df.shape[0] > 0:
            fail_df.to_excel(writer, sheet_name='Failed', index=False)

        # Save out metadata key and descriptions
        key_descriptions.metakey_df.to_excel(writer, sheet_name='metadata_keys', index=False)

        # write out layer keys
        key_df.to_excel(writer, sheet_name="layer_keys", index=False)

        # Save out column descriptions if necessary
        if all_oct_linescan_df.shape[0] > 0 or all_oct_radial_df.shape[0] > 0:
            key_descriptions.linescanradial_df.to_excel(writer, sheet_name='OCT_Linescan_keys', index=False)
        if all_oct_ppole_df.shape[0] > 0:
            key_descriptions.ppole_df.to_excel(writer, sheet_name='OCT_Ppole_ETDRS_keys', index=False)
        if all_oct_peri_df.shape[0] > 0:
            key_descriptions.peripapillary_df.to_excel(writer, sheet_name='SLOCT_Peripapillary_keysO_keys', index=False)
        if all_slo_df.shape[0] > 0:
            key_descriptions.slo_df.to_excel(writer, sheet_name='SLO_keys', index=False)
   
    # save out log
    with open(os.path.join(save_directory, f"analysis_log.txt"), "w") as f:
        for line in all_logging_list:
            f.write(line+"\n")