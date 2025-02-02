import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import shutil
import sys
import time
import pandas as pd
import pprint
import numpy as np
from tqdm import tqdm
from pathlib import Path, PosixPath, WindowsPath

from octolyzer import analyse, collate_data, utils
from octolyzer.segment.octseg import choroidalyzer_inference, deepgpet_inference
from octolyzer.segment.sloseg import slo_inference, avo_inference, fov_inference
from octolyzer.measure.bscan.thickness_maps import grid


# Abbreviated versions of different layers in the OCT
KEY_LAYER_DICT = {"ILM": "Inner Limiting Membrane",
                  "GCL": "Ganglion Cell Layer",
                  "RNFL": "Retinal Nerve Fiber Layer",
                  "IPL": "Inner Plexiform Layer",
                  "INL": "Inner Nuclear Layer",
                  "OPL": "Outer Plexiform Layer",
                  "ELM": "External Limiting Membrane", # Outer nuclear layer
                  "PR1": "Photoreceptor Layer 1",
                  "PR2": "Photoreceptor Layer 2",
                  "RPE": "Retinal Pigment Epithelium",
                  "BM": "Bruch's Membrane Complex", 
                  "CHORupper": "Choroid - Sclera boundary",
                  "CHORlower": "Bruch's Membrane - Choroid boundary"}

def run(args):
    """
    Analyze .vol files from a specified directory using the OCT and SLO analysis pipeline.

    This function processes OCT and SLO data from Heidelberg `.vol` files. It performs segmentation, 
    measurement, and data collation based on the configuration provided in the `args` dictionary input. 
    Results and logs are saved to the specified output directory.

    Parameters:
    ----------
    args : dict
        Configuration dictionary containing the following keys:
        - `analysis_directory` (str): Path to the directory containing `.vol` files.
        - `output_directory` (str): Path to the directory where results will be saved.
        - `robust_run` (int): 1 to skip files with unexpected errors; 0 to halt and allow debugging.
        - `save_individual_segmentations` (int): Save segmentation masks per individual (1 to save, 0 to skip).
        - `save_individual_images` (int): Save SLO/OCT images for each individual (1 to save, 0 to skip).
        - `preprocess_bscans` (int): Preprocess B-scans to enhance choroid visibility (1 to enable, 0 to disable).
        - `analyse_choroid` (int): Include choroid-related results (1 to include, 0 to exclude).
        - `analyse_slo` (int): Analyze the accompanying SLO image (1 to include, 0 to exclude).
        - `custom_maps` (int): Specify custom retinal thickness maps (1 to enable, 0 to disable).
        - `analyse_all_maps` (int): Analyze all retinal layer thickness maps for Ppole data (1 to enable, 0 to disable).
        - `analyse_square_grid` (int): Analyze the square posterior pole grid (1 to enable, 0 to disable).
        - `choroid_measure_type` (str): Type of choroid measurement ("perpendicular" or "vertical").
        - `linescan_roi_distance` (int): Distance (in microns) for structure measurement around the fovea 
                                         in H-line/V-line/Radial scans.

    Outputs:
    -------
    - Results saved in the `output_directory`:
        - Individual segmentation masks and raw images, if specified.
        - Segmentation visualisations to demonstrate features measured.
        - Consolidated Excel file with metadata, SLO, and OCT measurements.
        - Configuration file (`configuration_used.txt`) copied for reproducibility.
        - Log files for individual and batch processing.

    Notes:
    -----
    - If `robust_run` is set to 1, unexpected errors for specific files will be logged, and analysis 
      will continue with other files.
    - Choroid analysis can be toggled with `analyse_choroid`.
    - Composite SLO visualisations, if enabled, are saved in the `slo_segmentations` subdirectory.
    - Composite OCT visualisations are saved in the `oct_segmentations` subdirectory.

    Examples:
    --------
    >>> args = {
    ...     "analysis_directory": "/path/to/input",
    ...     "output_directory": "/path/to/output",
    ...     "robust_run": 1,
    ...     "save_individual_segmentations": 1,
    ...     "save_individual_images": 1,
    ...     "preprocess_bscans": 1,
    ...     "analyse_choroid": 1,
    ...     "analyse_slo": 1,
    ...     "custom_maps": 0,
    ...     "analyse_all_maps": 0,
    ...     "analyse_square_grid": 0,
    ...     "choroid_measure_type": "perpendicular",
    ...     "linescan_roi_distance": 3000
    ... }
    >>> run(args)
    Analyzing .vol files from `/path/to/input` and saving results to `/path/to/output`.
    """
    
    # analysis directory
    analysis_directory = args["analysis_directory"]
    if not os.path.exists(analysis_directory):
        print("Cannot find directory images/ with files to analyse.")
        print("Please create directory and place images inside. Exiting analysis")
        sys.exit()

    # Detect .vol files from analysis_directory
    vol_paths = sorted(Path(analysis_directory).glob("*.vol"))
    N = len(vol_paths)
    if N > 0:
        print(f"Found {len(vol_paths)} to analyse.")
    else:
        print(f'Cannot find any supported files in {analysis_directory}. Please check directory. Exiting analysis')
        return

    # output directory
    save_directory = args["output_directory"]
    if not os.path.exists(save_directory):
        print("Cannot find directory output/ to store results.")
        print('Creating folder.')
        os.mkdir(save_directory)

    # Copy the config file over into the folder to take a screenshot of the configuration used during this batch processing
    shutil.copy(os.path.join(MODULE_PATH, 'config.txt'), os.path.join(save_directory, 'configuration_used.txt'))

    # This is a particularly helpful parameter when running large batches, and ignored
    # any unexpected errors from a particular file. Setting it as 0 will throw up errors
    # for debugging
    robust_run = args["robust_run"]
    analyse_slo_flag = args['analyse_slo'] 
    collate_segs = True
    analyse_choroid = True

    # create segmentations directory if specified
    if analyse_slo_flag:
        slo_segmentation_directory = os.path.join(save_directory, "slo_segmentations")
        if collate_segs and not os.path.exists(slo_segmentation_directory):
            os.mkdir(slo_segmentation_directory)
    oct_segmentation_directory = os.path.join(save_directory, "oct_segmentations")
    if collate_segs and not os.path.exists(oct_segmentation_directory):
        os.mkdir(oct_segmentation_directory)

    # construct param dicts
    param_keys = ["save_individual_segmentations",
                "save_individual_images",
                "preprocess_bscans",
                "custom_maps",
                "analyse_all_maps",
                "analyse_choroid",
                'analyse_slo',
                "analyse_square_grid",
                "choroid_measure_type",
                "linescan_roi_distance"]
    param_dict = {key:args[key] for key in param_keys}

    # Instantiate SLO binary/AVOD/Fovea segmentation model
    print(f"\nLoading SLO and OCT models.")
    slosegmenter = slo_inference.SLOSegmenter()
    avosegmenter = avo_inference.AVOSegmenter()
    fovsegmenter = fov_inference.FOVSegmenter()
    choroidalyzer = choroidalyzer_inference.Choroidalyzer()
    deepgpet = deepgpet_inference.DeepGPET()

    if analyse_choroid:
        print(f"\nRunning choroid and retinal analysis.")
    else:
        print(f"\nRunning OCTolyzer...")

    # Loop through .vol files, segment, measure and save out in analyse()
    st = time.time()
    oct_slo_result_dict = {}
    for path in tqdm(vol_paths, desc='Analysing...', leave=False):
        
        # Initialise results dictionary for .vol file and create paths to results
        fname_type = os.path.split(path)[-1]
        oct_slo_result_dict[fname_type] = {}
        fname = fname_type.split(".")[0]
        fname_path = os.path.join(save_directory, fname)
        output_fname = os.path.join(fname_path, f"{fname}_output.xlsx")

        # Check for pre-existing manual segmentations of SLO 
        slo_mannotations = ( len(list(Path(fname_path).glob("*.nii.gz"))) - len(list(Path(fname_path).glob("*_used.nii.gz"))) )  > 0
        param_dict['manual_annotation'] = int(slo_mannotations)

        # Check to see if '{fname}_output.xlsx' exists (i.e. file has already been processed)
        # If so, load in results and skip to next file.
        if os.path.exists(output_fname) and not slo_mannotations:
            print(f"Previously analysed {fname}.")
            ind_df, slo_dfs, oct_dfs, log = collate_data._load_files(fname_path, logging_list=[])
            oct_slo_result_dict[fname_type]['metadata'] = ind_df
            if analyse_slo_flag:
                oct_slo_result_dict[fname_type]['slo'] = slo_dfs
            else:
                oct_slo_result_dict[fname_type]['slo'] = [pd.DataFrame()]
            oct_slo_result_dict[fname_type]['oct'] = oct_dfs
            oct_slo_result_dict[fname_type]['log'] = log

        # Skip if .vol file is an OCT-A scan
        elif "_ANGIO" in fname:
            print(f"{fname} is an OCT-A scan. Skipping.\n\n")

        # If unprocessed and valid, apply pipeline
        else:

            # For batch processing
            if robust_run:

                # Catch any exceptions 
                try:

                    # Analyse file
                    output = analyse.analyse(path, 
                                    save_directory, 
                                    choroidalyzer, 
                                    slosegmenter, 
                                    avosegmenter,
                                    fovsegmenter,
                                    deepgpet,
                                    param_dict)
                    slo_analysis_output, oct_analysis_output = output

                    # Store results for collating
                    oct_slo_result_dict[fname_type]['metadata'] = oct_analysis_output[0]
                    if slo_analysis_output is not None and analyse_slo_flag:
                        oct_slo_result_dict[fname_type]['slo'] = slo_analysis_output[1]
                    else:
                        oct_slo_result_dict[fname_type]['slo'] = [pd.DataFrame()]
                    oct_slo_result_dict[fname_type]['oct'] = oct_analysis_output[3]
                    oct_slo_result_dict[fname_type]['log'] = oct_analysis_output[-1]

                # Unexpected error
                except Exception as e:
                    
                    # print and log error
                    user_fail = f"\nFailed to analyse {fname}."
                    log = utils.print_error(e)
                    logging_list = [user_fail] + log
                    skip = "Skipping and moving to next file.\nCheck data input and/or set robust_run to 0 to debug code.\n"
                    print(skip)

                    # Try at least save out metadata from loading volfile for failed
                    # file - making sure to mark in FAILED column
                    try:
                        _, metadata, _, _, _ = utils.load_volfile(path, verbose=False)
                        metadata['FAILED'] = True
                        if metadata["bscan_type"] == 'Peripapillary':
                            del metadata['stxy_coord']

                    # Catch any exceptions with failing to even load image and metadata from 
                    # volfile
                    except:
                        metadata = {'Filename':os.path.split(path)[1]}
                        fail_load = "Failed to even load path, check utils.load_volfile"
                        print(fail_load)
                        log.append(fail_load)

                    # Store results for collating
                    oct_slo_result_dict[fname_type]['metadata'] = metadata
                    oct_slo_result_dict[fname_type]['oct'] = logging_list[0]
                    oct_slo_result_dict[fname_type]['log'] = logging_list
                    
            else:
                
                # Analyse file
                output = analyse.analyse(path, 
                                save_directory, 
                                choroidalyzer, 
                                slosegmenter, 
                                avosegmenter,
                                fovsegmenter,
                                deepgpet,
                                param_dict)
                slo_analysis_output, oct_analysis_output = output

                # Store results for collating
                oct_slo_result_dict[fname_type]['metadata'] = oct_analysis_output[0]
                if slo_analysis_output is not None and analyse_slo_flag:
                    oct_slo_result_dict[fname_type]['slo'] = slo_analysis_output[1]
                else:
                    oct_slo_result_dict[fname_type]['slo'] = [pd.DataFrame()]
                oct_slo_result_dict[fname_type]['oct'] = oct_analysis_output[3]
                oct_slo_result_dict[fname_type]['log'] = oct_analysis_output[-1]

    # Collect all measurements together
    collate_data.collate_results(oct_slo_result_dict, save_directory, param_dict['analyse_choroid'], param_dict['analyse_slo'])

    # Complete !
    elapsed = time.time() - st
    print(f"Completed analysis in {elapsed:.03f} seconds.")


# Once called from terminal
if __name__ == "__main__":

    print("Checking configuration file for valid inputs...")

    # Load in configuration from file
    config_path = os.path.join(MODULE_PATH, 'config.txt')
    with Path(config_path).open('r') as f:
        lines = f.readlines()
    inputs = [l.strip() for l in lines if (":" in str(l)) and ("#" not in str(l))]
    params = {p.split(": ")[0]:p.split(": ")[1] for p in inputs}

    # Make sure inputs are correct format before constructing args dict
    for key, param in params.items():

        # Checks for directory
        if "directory" in key:
            check_param = param.replace(" ", "")
            if "analysis" in key:
                try:
                    assert os.path.exists(param), f"The specified path:\n{param}\ndoes not exist. Check spelling or location. Exiting analysis."
                except AssertionError as msg:
                    sys.exit(msg)
            continue

        # Check numerical value inputs and custom slab input
        param = param.replace(" ", "")
        if param == "":
            msg = f"No value entered for {key}. Please check config.txt. Exiting analysis"
            sys.exit(msg)

        # Error checking distance to measure features on H-line/V-line/Radial linescans 
        # within [100, 4000] microns either side of fovea
        elif key == 'linescan_roi_distance':
            try:
                param = int(param)
                roi_min, roi_max = 100, 4000
                if (param < roi_min) or (param > roi_max):
                    msg = print(f"Value {param} for parameter {key} must be in [{roi_min}, {roi_max}]. Falling back to default ROI distance of 1500 microns.")
                    param = 1500

            # Any unexpected error defaults this value to 1500 microns either side of fovea
            except:
                msg = print(f"Value {param} for parameter {key} is not valid. Falling back to default ROI of 1500 microns.")
                param = 1500
            params[key] = param

        # Error check measurement protocol for choroid, either according to image-axis 'vertical' or choroid-axis 'perpendicular'
        # Defaults to 'perpendicular' given error
        elif key == 'choroid_measure_type':
            if param not in ['vertical', 'perpendicular']:
                print(f"Method to measure choroid invalid. Must be 'perpendicular' or 'vertical'.\nSelecting 'perpendicular' by default.")
                params[key] = 'perpendicular'

        # Post-processing and error checking custom maps 
        elif key == "custom_maps":
            slabs = param.replace(" ", "").split(",")

            # If entering custom_maps as 0 as default on config.txt
            if slabs[0] == "0":
                params[key] = []
                continue

            # Flatten into separate layers and remove any involving the choroid
            # as custom_maps refers only to the retina
            layers_keys = np.array(list(KEY_LAYER_DICT.keys())[:-2])
            layers = np.unique(np.array([s.split("_") for s in slabs]).flatten())
            if ("CHORupper" in layers) or ("CHORlower" in layers):
                print(f"Custom thickness map inputs are only for individual retinal layers.")
                slabs = [s for s in slabs if np.all([l not in s for l in ["CHORupper", "CHORlower"]])]

            # Any custom-specified layers not in KEY_LAYER_DICT remove and log to user
            check_layers = np.array([l not in KEY_LAYER_DICT.keys() for l in layers])
            if np.any(check_layers):
                bad_layers = layers[np.where(check_layers)[0]]
                print(f"Specified layer(s) {bad_layers} do not exist.")
                print("Please check config file. Available layers (if they exist in .vol) are:")
                pprint.pprint(list(KEY_LAYER_DICT.keys()))
                print("Please format the custom layer as {lyr1}_{lyr2}, i.e. ILM_RPE, for automatic detection.")
                slabs = [s for s in slabs if np.all([l not in s for l in bad_layers])]

            # Re-order top-down such "BM_ILM" becomes "ILM_BM" to ease pipeline processing 
            final_slabs = []
            for s in slabs:
                l1,l2 = s.split("_")
                if np.where(l1 == layers_keys)[0][0] > np.where(l2 == layers_keys)[0][0]:
                    s = f"{l2}_{l1}"
                final_slabs.append(s)

            # Output to user which custom retinal thickness maps will be measures
            if len(final_slabs) > 0:
                print(f"Building custom thickness maps: {final_slabs}.\n")
                params[key] = final_slabs   
            else:
                print(f"No valid custom thickness maps detected. Carrying out analysis without custom maps.")
                params[key] = []

        # Ensure any binary flagging variables are specified either '0' or '1' and return if not.
        elif ("analyse" in key) or ("save" in key):
            try:
                assert param in ["0", "1"], f"{key} flag must be either 0 or 1, not {param}. Exiting analysis."
            except AssertionError as msg:
                sys.exit(msg)
        
        # All other config options should be an integer so quick check on these.  
        else:
            try:
                int(param)
            except:
                msg = print(f"Value {param} for parameter {key} is not valid. Please check config.txt, Exiting analysis.")
                sys.exit(msg)

    # Construct args dict and run
    args = {key:val 
            if (("directory" in key) or ('measure' in key) or ("custom" in key) or ("etdrs_distance" in key)) 
            else int(val) for (key,val) in params.items()}

    # run analysis
    run(args)
