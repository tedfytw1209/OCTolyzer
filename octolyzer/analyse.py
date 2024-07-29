import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = "\\".join(SCRIPT_PATH.split('\\')[:-1])
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import scipy
import utils
from PIL import Image, ImageOps
from pathlib import Path, PosixPath, WindowsPath

from octolyzer.measure.bscan.thickness_maps import grid
from octolyzer.segment.octseg import choroidalyzer_inference, deepgpet_inference
from octolyzer.measure.bscan import bscan_measurements, ppole_measurements
from octolyzer.measure.bscan.thickness_maps import map as map_module
from octolyzer import analyse_slo, utils, collate_data

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
                  "CHORupper": "Choroid - Sclera boundary",
                  "CHORlower": "Bruch's Membrane - Choroid boundary"}

def analyse(path, 
            save_path, 
            choroidalyzer=None, 
            slo_model=None, 
            avo_model=None, 
            fov_model=None,
            deepgpet=None,
            param_dict=None,
            verbose=True):
    """
    Inner function to analyse an individual metadata+image file.
    """
    # Initialise list of messages to save and output files
    logging_list = []
    oct_output = []

    # Default configuration if nothing specified
    if param_dict is None:
        # flags for choroid analysis, preprocessing bscans
        preprocess_data = 0
    
        # For saving out representative Bscan/SLO/segmentation masks
        save_ind_segmentations = 0
        save_ind_images = 1
    
        # thickness map parameters, default measure ETDRS but not square
        custom_maps = []
        all_maps = 0

        # By default we analyse choroid
        analyse_choroid = 1

        # By default we analyse the SLO image
        analyse_slo_flag = 0

        # By default we measure the posterior pole grid
        sq_grid_flag = 0

        # By default, we measure the choroid perpendicular
        chor_measure_type = 'perpendicular'

        # By default, our linescan ROI distance is 1500 microns either side of fovea (microns)
        macula_rum = 3000
        
    else:
        # flags for choroid analysis, preprocessing bscans
        preprocess_data = param_dict["preprocess_bscans"]
    
        # For saving out representative Bscan/SLO/segmentation masks
        save_ind_segmentations = param_dict["save_individual_segmentations"]
        save_ind_images = param_dict["save_individual_images"]

        # Custom retinal thickness maps
        custom_maps = param_dict["custom_maps"]
        all_maps = param_dict["analyse_all_maps"]

        # analysing choroid?
        analyse_choroid = param_dict['analyse_choroid']

        # square grid for Ppole
        sq_grid_flag = param_dict['analyse_square_grid']

        # analysing SLO?
        analyse_slo_flag = param_dict['analyse_slo']

        # User-specified measure type for choroid
        chor_measure_type = param_dict['choroid_measure_type']

        # User-specified ROI distance either side of fovea
        macula_rum = param_dict['linescan_roi_distance']

    # Default parameters for thickness maps: ETDRS grid and optional square grid
    etdrs_kwds = {"etdrs_microns":[1000,3000,6000]}
    square_kwds = {"N_grid":8, "grid_size":7000}
    map_flags = [1, sq_grid_flag]
    map_kwds = [etdrs_kwds, square_kwds]

    # By default we save individual results and collate segmentations
    collate_segmentations = 1
    
    # Default bscan/slo measurement parameters
    N_measures = "all" # Measuring all thicknesses across ROI to average over
    N_avgs = 0 # Robust thickness estimation, only relevant when N_measures is an integer
    chor_linescan_measure_type = chor_measure_type # Measuring type for choroidal metrics across OCT Linescans
    chor_ppole_measure_type = chor_measure_type # Measuring type for choroidal metrics across OCT Volumes
    ret_measure_type = 'vertical' # Measuring retina column-wise (via A-scans) according to most devices/literature
    
    # Double check existence of core save_path and create 
    fname_type = os.path.split(path)[1]
    fname = fname_type.split(".")[0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, fname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dirpath = os.path.split(save_path)[0]

    # segmentation directory
    if collate_segmentations:
        segmentation_directory = os.path.join(dirpath, "oct_segmentations")
        if not os.path.exists(segmentation_directory):
            os.mkdir(segmentation_directory)

    msg = f"\nANALYSING SLO+OCT OF {fname}.\n"
    logging_list.append(msg)
    if verbose:
        print(msg)
    output = utils.load_volfile(path, preprocess=preprocess_data*analyse_choroid, verbose=verbose,
                                custom_maps=custom_maps, logging=logging_list)
    bscan_data, metadata, (slo, slo_acq, slo_at_fovea), layer_pairwise, logging_list = output
    N_scans, M, N = bscan_data.shape
    oct_output.append(bscan_data)

    # Pixel spacing, SLO pixel scaling is assumed as isotropic
    scaleX, scaleY, scaleZ = metadata["bscan_scale_x"],metadata["bscan_scale_y"],metadata["bscan_scale_z"]
    bscan_scale = (scaleX, scaleY)
    slo_scale = metadata["slo_scale_xy"]

    # Analyse the SLO image
    scan_location = metadata["location"]
    if scan_location == "peripapillary":
        slo_location = "Optic disc"
    else:
        slo_location = "Macula"
    eye = metadata["eye"]

    # We need to analyse SLO if we have a peripapillary scan so we have the fovea and AV map
    # for optic disc assessment
    slo_metrics = False
    new_avodmap = False
    new_binary = False
    slo_recompute = new_avodmap + new_binary
    if analyse_slo_flag:
        slo_metrics = True

    # have_slo default True as .VOL has SLO included. In future support (.dcm)  
    # this may not be the case
    have_slo = True

    # OCT analysis is the priority, so skip SLO analysis if unexpected error occurs
    try:
        fname_output_path = os.path.join(save_path,f'{fname}_output.xlsx')
        if have_slo and (analyse_slo_flag or scan_location == 'peripapillary'):
            
            # Check if manual annotation exists - this is copied from sloctolyzer/main.py
            binary_nii_path = os.path.join(save_path, f"{fname}_slo_binary_map.nii.gz")
            avod_nii_path = os.path.join(save_path, f"{fname}_slo_avod_map.nii.gz")
            if os.path.exists(avod_nii_path):
                new_avodmap = True
            if os.path.exists(binary_nii_path):
                new_binary = True
            slo_recompute = new_avodmap + new_binary
                
            # Load in manual annotations if they exist, otherwise skip and start from scratch
            segmentation_dict = {}
            if new_avodmap or new_binary:     
                msg = "\nDetected manual annotation of"
                ind_df = pd.read_excel(fname_output_path, sheet_name="metadata")
            
                # Load in annotations if exists. If not, load in already saved segmentation
                if new_avodmap:
                    msg += " artery-vein-optic disc mask"
                    if new_binary:
                        msg += " and"
                    new_avodmap = utils.load_annotation(avod_nii_path)
                else:
                    avodmap = np.array(Image.open(os.path.join(save_path, f"{fname}_slo_avod_map.png")))
                    new_avodmap = np.concatenate([avodmap[...,np.newaxis]==191,
                                                avodmap[...,np.newaxis]==255,
                                                avodmap[...,np.newaxis]==127,
                                                avodmap[...,np.newaxis]>0], axis=-1)
                if new_binary:
                    msg += " binary vessel mask"
                    new_binary = utils.load_annotation(binary_nii_path, binary=True)
                else:
                    new_binary = np.array(ImageOps.grayscale(Image.open(os.path.join(save_path, f"{fname}_slo_binary_map.png"))))/255
                msg += ". Recomputing metrics..."
                print(msg)
                logging_list.append(msg)
                    
                # Collect new segmentations, and metadata 
                segmentation_dict['avod_map'] = new_avodmap.astype(int)
                segmentation_dict['binary_map'] = new_binary.astype(int)
                segmentation_dict['metadata'] = dict(ind_df.iloc[0])
            
            slo_analysis_output = analyse_slo.analyse(255*slo, save_path,
                                            slo_scale, slo_location, eye,
                                            slo_model, avo_model, fov_model,
                                            save_images=save_ind_segmentations, 
                                            compute_metrics=slo_metrics, verbose=verbose, 
                                            collate_segmentations=True, segmentation_dict=segmentation_dict)
            slo_meta_df, slo_measure_dfs, _, slo_segmentations, slo_logging_list = slo_analysis_output
            logging_list.extend(slo_logging_list)
            slo_avimout = slo_segmentations[-1]
            
        else:
            slo_analysis_output = None
    
    # Unexpected error in SLO analysis
    except Exception as e:
        user_fail = f"\nFailed to analyse SLO of {fname}."
        if verbose:
            print(user_fail)
        slo_log = utils.print_error(e, verbose)
        slo_logging_list = [user_fail] + slo_log
        logging_list.extend(slo_logging_list)
        have_slo = False
        slo_analysis_output = None

    # If recomputed SLO metrics with manual annotations, OCT metrics have already been computed
    # so load these in and return
    if slo_recompute:
        ind_df, _, oct_dfs, log = collate_data._load_files(save_path, logging_list=[])
        oct_analysis_output = ind_df, slo, bscan_data, oct_dfs, [], log

        # save new SLO measurements
        if slo_analysis_output is not None:
            slo_measure_dfs = slo_analysis_output[1]
            with pd.ExcelWriter(fname_output_path, engine = "openpyxl",  mode='a') as writer:
                workBook = writer.book
                for df in slo_measure_dfs:  
                    if len(df) > 0:
                        z = df.zone.iloc[0]
                        try:
                            workBook.remove(workBook[f'slo_measurements_{z}'])
                        except:
                            print("worksheet doesn't exist")
                        finally:
                            df.to_excel(writer, sheet_name=f'slo_measurements_{z}', index=False)
        return slo_analysis_output, oct_analysis_output


    # Alert to user we are analysing OCT from here on
    msg = f"\n\nANALYSING OCT of {fname}.\n"
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Forcing model instantiation if unspecified
    # Choroid segmentation models - macular B-scans
    if choroidalyzer is None or type(choroidalyzer) != choroidalyzer_inference.Choroidalyzer:
        msg = "Loading models..."
        logging_list.append(msg)
        if verbose:
            print(msg)
        choroidalyzer = choroidalyzer_inference.Choroidalyzer()
    # DeepGPET for peripapillary B-scans
    if deepgpet is None or type(deepgpet) != deepgpet_inference.DeepGPET:
        deepgpet = deepgpet_inference.DeepGPET()

    # Segment choroid
    # If macular-centred, use Choroidalyzer. If optic disc-centred, use deepGPET
    scan_type = metadata["bscan_type"]
    if analyse_choroid:
        msg = "Segmenting choroid and fovea..."
    else:
        msg = "Detecting fovea for grid/ROI alignment (through use of Choroidalyzer)..."
    logging_list.append(msg)
    if verbose:
        print(msg)
    if scan_location == "macular":
        rvfmasks, foveas = choroidalyzer.predict_list(bscan_data, soft_pred=True)
    elif scan_location == "peripapillary":
        rvfmasks = deepgpet.predict_list(bscan_data, soft_pred=True)

    # For execution time of the segmentation
    # return None
    
    # Resolve fovea detection. If at origin then threshold too high, apply filter function and warn user.
    if scan_location != "peripapillary":
        fovea_slice_num, fovea, fov_log = utils._get_fovea(rvfmasks, foveas, N_scans, scan_type, logging=[])
        logging_list.extend(fov_log)

    # Detect retinal layer keys
    pairwise_keys = list(layer_pairwise.keys())
    layer_keys = list(set(pd.DataFrame(pairwise_keys).reset_index(drop=True)[0].str.split("_", expand=True).values.flatten()))

    # If a single line scan, plot segmentations and measure thickness/area/CVI
    od_radius = None
    od_overlap = None
    od_warning = None
    od_centre = None
    missing_fovea = False
    if scan_location == "peripapillary":
        msg = f"\nSegmented B-scan visualisation saved out.\n"
        logging_list.append(msg)
        if verbose:
            print(msg)
        if analyse_choroid:
            traces = utils.get_trace(rvfmasks[0], 0.25, align=True)
            layer_pairwise["CHORupper_CHORlower"] = [np.array(traces)]
            layer_keys.append("CHORupper")
            layer_keys.append("CHORlower")

        layer_keys_copied = layer_keys.copy()
        if save_ind_segmentations:
            fig, (ax0,ax) = plt.subplots(2,1,figsize=(15,10))
            ax0.imshow(bscan_data[0], cmap='gray')
            ax.imshow(bscan_data[0], cmap="gray")
            for key, tr in layer_pairwise.items():
                for (k, t) in zip(key.split("_"), tr[0]):
                    if k in layer_keys_copied:
                        ax.plot(t[:,0],t[:,1])
                        layer_keys_copied.remove(k)
            ax.set_axis_off()
            ax0.set_axis_off()
            fig.tight_layout(pad = 0)
            fig.savefig(os.path.join(save_path, f"{fname}_octseg.png"), bbox_inches="tight")
            if collate_segmentations:
                fig.savefig(os.path.join(segmentation_directory, f"{fname}.png"), bbox_inches="tight")
            plt.close()

            # For a single B-scan, measure thickness and area of all layers, and CVI for choroid
        msg = f"""\nMeasuring thickness around the optic disc for retina and/or choroid.
Thickness measurements will be averaged following the standard peripapillary subgrids.
All measurements are made with respect to the image axis (vertical) as this is a circular B-scan (continuous at either end)."""
        logging_list.append(msg)
        if verbose:
            print(msg)

        # Extract metadata from SLO for disc-centred inputs to align peripapillary grid
        if have_slo:
            fovea_at_slo = slo_meta_df[["slo_fovea_x", "slo_fovea_y"]].values[0].astype(int)
            missing_fovea = slo_meta_df.missing_fovea.values[0].astype(bool)
            od_radius = slo_meta_df.optic_disc_radius_px.values[0].astype(int)
    
            # Determine A-scan along B-scan which is centred between the fovea and optic-disc.
            # We call this the temporal midpoint
            output = utils.align_peripapillary_data(metadata, fovea_at_slo, slo_acq, slo_avimout, 
                                                    fname, save_path, save_ind_segmentations)
            od_centre, offset_ratio, ascan_idx_temp0  = output
            od_overlap = np.round(100*offset_ratio, 3)
            del metadata['stxy_coord']
            msg = f"User-specified optic disc center is {od_overlap}% of the optic disc diameter."
            logging_list.append(msg)
            if verbose:
                print(msg)
    
            # Add warning to user if optic disc overlap is greater than 15% of the optic disc radius
            if od_overlap > 15:
                od_warning = True
                msg = f"WARNING: This overlap suggests the acquisition is off-centre from the optic disc. Please check scan/optic disc segmentation."
                logging_list.append(msg)
                if verbose:
                    print(msg)
            else:
                od_warning = False
                
        else:
            fovea_at_slo = [np.nan, np.nan]
            od_overlap = np.nan
            od_radius = np.nan
            msg = f"WARNING: Without SLO, peripapillary grid will be centred in the middle of the B-scan, and is likely off-centre"
            logging_list.append(msg)
            if verbose:
                print(msg)
            ascan_idx_temp0 = N//2

        # Measure thickness arrays per segmented layer
        measure_dict = {}
        for key, tr in layer_pairwise.items():

            # Measure thickness across entire B-scan
            peri_refpt = tr[0][0,N//2]
            thickness = map_module.measure_thickness(tr, scale=bscan_scale, 
                                              offset=0, fovea=peri_refpt, 
                                              measure_type="vertical", region_thresh=0,
                                              disable_progress=True)[0][0]
            
            # Pad thickness with zeros if segmentation doesn't extend to entire image
            stx, enx = tr[0][0,[0,-1],0]
            if thickness.shape[0] < N:
                msg = f"\nWARNING: Peripapillary segmentation for layer {key} missing {np.round(100*((stx+(N-enx))/N),2)}% pixels. Padding thickness array with 0s. Please check segmentation.\n"
                logging_list.append(msg)
                if verbose:
                    print(msg)
                thickness = np.pad(thickness, (max(0,stx-1), max(0,N-enx)))

            # Align the thickness vector
            align_idx = N//2 - ascan_idx_temp0
            if align_idx > 0:
                align_thickness = np.pad(thickness, (align_idx, 0), mode="reflect")[:N]
            else:
                align_thickness = np.pad(thickness, (0, -align_idx), mode="reflect")[align_idx:]

            # We create a moving average, a smoothed version of the raw aligned thickness values
            ma_idx = 32
            align_thickness_padded = np.pad(align_thickness, (ma_idx,ma_idx), mode="reflect")
            moving_avg = pd.Series(align_thickness_padded).rolling(window=ma_idx,
                                                                    center=True).mean().values[ma_idx:-ma_idx]
            
            # We fit a spline to the raw and smoothed thickness values, and define that over
            # [-180, 180] degree window
            N_line = align_thickness.shape[0]
            x_grid = np.linspace(-180., 180., N_line)
            spline_raw = scipy.interpolate.UnivariateSpline(x_grid, align_thickness)(x_grid)
            spline_ma = scipy.interpolate.UnivariateSpline(x_grid, moving_avg)(x_grid)
            spline_raw_coords = np.concatenate([[x_grid], [spline_raw]]).T
            spline_ma_coords = np.concatenate([[x_grid], [spline_ma]]).T

            # Organise thickness values into their circular subregions
            grid_cutoffs = np.array([0, 45, 90, 135, 225, 270, 315, 360]) - 180
            grids = ["nasal", "infero_nasal", "infero_temporal", 
                     "temporal", "supero_temporal", "supero_nasal", "nasal"]
            grid_measures_raw = {g+'_[um]':[] for g in grids}
            grid_measures_ma = {g+'_[um]':[] for g in grids}
            for g_str, g_idx_i, g_idx_j in zip(grids, grid_cutoffs[:-1]+180, grid_cutoffs[1:]+180):
                x_idx_i = int(N*(g_idx_i/360))
                x_idx_j = int(N*(g_idx_j/360))
                grid_measures_raw[g_str+'_[um]'].extend(list(spline_raw_coords[x_idx_i:x_idx_j, 1]))
                grid_measures_ma[g_str+'_[um]'].extend(list(spline_ma_coords[x_idx_i:x_idx_j, 1]))
            
            # Average across entire grid
            grid_measures_raw["All"+'_[um]'] = spline_raw_coords[:,1].mean()
            grid_measures_ma["All"+'_[um]'] = spline_ma_coords[:,1].mean()

            # Average in subgrid of temporal zone, orientated to fovea
            grid_measures_raw["PMB"+'_[um]'] = grid_measures_raw["temporal_[um]"][30:60]
            grid_measures_ma["PMB"+'_[um]'] = grid_measures_ma["temporal_[um]"][30:60]

            # Measure the average thickness per circular subgrid
            grid_means_raw = {key:int(np.mean(value)) for key, value in grid_measures_raw.items()}
            grid_means_ma = {key:int(np.mean(value)) for key, value in grid_measures_ma.items()}

            # Nasal-temporal ratio, catch exception of zero division if segmentation doesn't cover temporal region
            try:
                grid_means_raw["N/T"] = grid_means_raw["nasal_[um]"]/grid_means_raw["temporal_[um]"]
                grid_means_ma["N/T"] = grid_means_ma["nasal_[um]"]/grid_means_ma["temporal_[um]"]
            except:
                grid_means_raw["N/T"] = np.nan
                grid_means_ma["N/T"] = np.nan
            
            
            # Save to dict
            measure_dict[key] = grid_means_ma

            # Plot the thickness curve, and if SLO show the peripapillary grid measurements overlaid
            if save_ind_images and have_slo:
                grid.plot_peripapillary_grid(slo, slo_acq, metadata, grid_means_ma, fovea_at_slo, 
                                            spline_raw_coords, spline_ma_coords, key, fname+f'_{key}', save_path)
            else:
                grid.plot_thickness_profile(spline_raw_coords, spline_ma_coords, key, fname+f'_{key}', save_path)

        # Organise traces to be saved out
        all_ytr = {}
        layer_keys_copied = layer_keys.copy()
        for key, trace in layer_pairwise.items():
            for (k, t) in zip(key.split("_"), trace[0]):
                if k in layer_keys_copied:
                    (xtr, ytr) = t[:,0], t[:,1]
                    xst, xen = xtr[[0,-1]]
                    ytr_pad = np.pad(ytr, ((max(xst-1,0), N-xen)), mode="constant")
                    all_ytr[k] = {i:ytr_pad[i] for i in range(N)}
                    layer_keys_copied.remove(k)
        seg_df = utils.nested_dict_to_df(all_ytr).reset_index()
        seg_df = seg_df.rename({"index":"layer"}, inplace=False, axis=1)
        seg_df = {f"{scan_type}": seg_df}

    elif scan_type != "Ppole":
        msg = f"\nSegmented B-scan visualisation saved out.\n"
        logging_list.append(msg)
        if verbose:
            print(msg)
        if analyse_choroid:
            # Extract region mask and remove any vessel segmented pixels from outside segmented choroid
            traces = utils.get_trace(rvfmasks[0][0], 0.5, align=True)
            region_mask = utils.rebuild_mask(traces, img_shape=(M, N))
            vessel_mask = region_mask.astype(float) * rvfmasks[0][1].astype(float)
            vessel_cmap = np.concatenate([vessel_mask[...,np.newaxis]] 
                                         + 2*[np.zeros_like(vessel_mask)[...,np.newaxis]] 
                                         + [vessel_mask[...,np.newaxis] > 0.01], axis=-1)
            layer_pairwise["CHORupper_CHORlower"] = [np.array(traces)]
            layer_keys.append("CHORupper")
            layer_keys.append("CHORlower")

        layer_keys_copied = layer_keys.copy()
        if save_ind_segmentations:
            fig, (ax0,ax) = plt.subplots(1,2,figsize=(12,6))
            ax0.imshow(bscan_data[0], cmap='gray')
            ax.imshow(bscan_data[0], cmap="gray")
            for key, tr in layer_pairwise.items():
                for (k, t) in zip(key.split("_"), tr[0]):
                    if k in layer_keys_copied:
                        ax.plot(t[:,0],t[:,1])
                        layer_keys_copied.remove(k)
            if analyse_choroid:
                ax.imshow(vessel_cmap, alpha=0.5)
            if scan_type != "AV-line":
                ax.scatter(foveas[0][0], foveas[0][1], s=50, marker="X", edgecolor=(0,0,0), color="r", linewidth=1)
            ax.set_axis_off()
            ax0.set_axis_off()
            fig.tight_layout(pad = 0)
            fig.savefig(os.path.join(save_path, f"{fname}_octseg.png"), bbox_inches="tight")
            if collate_segmentations:
                fig.savefig(os.path.join(segmentation_directory, f"{fname}.png"), bbox_inches="tight")
            plt.close()

        # Analysis isn't entirely supported yet for AV-line scans, so just save out
        # Bscan, SLO and the segmented, do not measure.
        if scan_type != "AV-line":

            # For a single B-scan, measure thickness and area of all layers, and CVI for choroid
            msg = f"""Measuring average and subfoveal thickness, area, and vessel area/vascular index (for choroid only).
Region of interest is fovea-centred using a distance of {macula_rum}microns temporal/nasal to fovea.
All retinal measurements are made vertically, i.e. with respect to the image axis (vertical).
All choroidal measurements are made {chor_measure_type}."""
            logging_list.append(msg)
            if verbose:
                print(msg)
            measure_dict = {}
            for key, tr in layer_pairwise.items():
                vess_mask = None
                meas_type = ret_measure_type
                if "CHOR" in key:
                    vess_mask = vessel_mask
                    meas_type = chor_linescan_measure_type

                output = bscan_measurements.compute_measurement(tr[0], vess_mask=vess_mask, fovea=foveas[0], scale=bscan_scale, 
                                                                macula_rum=macula_rum, N_measures=N_measures, N_avgs=N_avgs,
                                                                measure_type=meas_type, img_shape=(M,N),
                                                                verbose=True, force_measurement=True)
                measure_dict[key] = {"subfoveal_thickness_[um]":output[0], "thickness_[um]":output[1], "area_[mm2]":output[2]}
                if "CHOR" in key:
                    measure_dict[key]["vascular_index"] = output[-2]
                    measure_dict[key]["vessel_area_[mm2]"] = output[-1]
        else:
            msg = f"""Scan location intersects arteries/veins and is not fovea-centred.
Measurements of thickness, area, etc. are not supported (yet).
Instead, B-scan and SLO images are automatically saved out."""
            logging_list.append(msg)
            if verbose:
                print(msg)
            measure_dict = {}
            save_ind_images = 1
        
        # Measurement metadata
        horizontal = [False, True][scan_type=="H-line"]
        if scan_type in ["H-line", "V-line"]:
            metadata["bscan_fovea_x"] = foveas[0][0]
            metadata["bscan_fovea_y"] = foveas[0][1]
            acq_angle, fovea_at_slo = map_module.detect_angle(slo_at_fovea, fovea=foveas[0], horizontal=horizontal, inpt=".vol")
            metadata["slo_fovea_x"] = fovea_at_slo[0]
            metadata["slo_fovea_y"] = fovea_at_slo[1]
            metadata["acquisition_angle_degrees"] = acq_angle
            metadata["linescan_area_ROI_microns"] = macula_rum
            metadata["choroid_measure_type"] = chor_measure_type
        else:
            metadata["bscan_fovea"] = None
            metadata["slo_fovea"] = None
            metadata["acquisition_angle"] = "Unknown"

        # Organise traces to be saved out
        all_ytr = {}
        layer_keys_copied = layer_keys.copy()
        for key, trace in layer_pairwise.items():
            for (k, t) in zip(key.split("_"), trace[0]):
                if k in layer_keys_copied:
                    (xtr, ytr) = t[:,0], t[:,1]
                    xst, xen = xtr[[0,-1]]
                    ytr_pad = np.pad(ytr, ((max(xst-1,0), N-xen)), mode="constant")
                    all_ytr[k] = {i:ytr_pad[i] for i in range(N)}
                    layer_keys_copied.remove(k)
        seg_df = utils.nested_dict_to_df(all_ytr).reset_index()
        seg_df = seg_df.rename({"index":"layer"}, inplace=False, axis=1)
        seg_df = {f"{scan_type}": seg_df}

    # Otherwise, generate and save out thickness/vessel maps, plus compute ETDRS volume
    else:
        msg = f"""\nGenerating thickness and volume maps following ETDRS (0.5mm,1.5mm,3mm radial concentric grids).
All retinal measurements are made vertically, i.e. with respect to the image axis (vertical).
All choroidal measurements are made {chor_measure_type}.
NOTE:Subregion volumes will not be computed for CVI map."""
        logging_list.append(msg)
        if verbose:
            print(msg)

        # Extract parameters for generating maps, rmove any vessel pixels outside choroid region for vmasks
        rmasks = np.array([utils.rebuild_mask(utils.get_trace(rvf_i[0], 0.5, align=False), img_shape=(M,N)) for rvf_i in rvfmasks])
        vmasks = np.array([rmask*rvf_i[1] for (rmask, rvf_i) in zip(rmasks, rvfmasks)])
        eye = metadata["eye"]

        # if retina fully segmentd, then we can also extract other custom_maps.
        # Otheriwse, construct default choroid and retina.
        if analyse_choroid:
            ppole_keys = ["choroid", "choroid_vessel", 'ILM_BM']
            ppole_units = ['[um]', '[um2]', '[um]']
            ppole_segs = [rmasks, rmasks, layer_pairwise['ILM_BM']]
        else:
            ppole_keys = ['ILM_BM']
            ppole_units = ['[um]']
            ppole_segs = [layer_pairwise['ILM_BM']]

        if len(layer_pairwise) > 1:
            if all_maps:
                for key_pair in pairwise_keys:
                    if key_pair not in custom_maps:
                        ppole_keys.append(key_pair)
                        ppole_units.append('[um]')
                        ppole_segs.append(layer_pairwise[key_pair])
            if len(custom_maps) > 0:
                for key_pair in custom_maps:
                    if key_pair not in ppole_keys:
                        ppole_keys.append(key_pair)
                        ppole_units.append('[um]')
                        ppole_segs.append(layer_pairwise[key_pair])

        # rename summary layers
        keys_to_names = ['ILM_BM', 'ILM_ELM', 'ELM_BM']
        names_to_keys = ['retina', 'inner_retina', 'outer_retina']
        ppole_keys = np.array(ppole_keys).astype('<U14')
        for k2n, n2k in zip(keys_to_names, names_to_keys):
            ppole_keys[ppole_keys==k2n] = n2k
        ppole_keys = list(ppole_keys)

        # Generate maps and measure in ETDRS grid 
        grid_type = ["etdrs", "square"]
        map_dict = {}
        measure_dict = {}
        volmeasure_dict = {}
        if collate_segmentations:
            ctmap_args = {}
            ctmap_args['core'] = [slo, fname, segmentation_directory]
        for (m_flag, m_type) in zip(map_flags, grid_type):
            if m_flag:
                measure_dict[m_type] = {}
                volmeasure_dict[m_type] = {}
        
        # save out thickness maps and visualisations in single folder to clean up directory
        map_save_path = os.path.join(save_path,'thickness_maps')
        if not os.path.exists(map_save_path):
            os.mkdir(map_save_path)

        # return metadata, ppole_segs, fovea, vmasks, ppole_keys

        # Loop over segmented layers and generate user-specified maps
        for key, seg in zip(ppole_keys, ppole_segs):

            # Compute maps
            msg = f"    {key}_map"
            logging_list.append(msg)
            if verbose:
                print(msg)
            ves_chorsegs = None
            measure_type = "vertical"
            if "choroid" in key:
                measure_type = ret_measure_type
                if key == "choroid_vessel":
                    ves_chorsegs = vmasks
                    measure_type = chor_ppole_measure_type
            map_output = map_module.construct_map(slo, slo_at_fovea, seg,
                                            fovea, fovea_slice_num, 
                                            bscan_scale, scaleZ,
                                            max_N=N, log_list=[],
                                            ves_chorsegs=ves_chorsegs,
                                            measure_type=measure_type)

            # Measure grids on the maps and save out in dedicated folder 
            for i,(m_flag, m_kwd) in enumerate(zip(map_flags, map_kwds)):
                m_type = grid_type[i]
                if m_flag:
                    if key == "choroid_vessel":
                        slo_output, macular_map, (angle, fovea_at_slo), cvi_map, map_messages = map_output
                        cvi_key = "choroid_CVI"
                        ppole_units.append('')
                        ppole_keys.append(cvi_key)
                        fname_key = fname+f"_{cvi_key}_{m_type}_map"

                        dtype = np.float64
                        grid_measure_output = grid.measure_grid(cvi_map, fovea_at_slo, 
                                                                scaleX, eye, rotate=angle, 
                                                                measure_type=m_type, grid_kwds=m_kwd,
                                                                interp=True, plot=save_ind_segmentations, 
                                                                slo=slo_output, dtype=dtype,
                                                                fname=fname_key, save_path=map_save_path)
                        grid_output, gridvol_output, grid_messages = grid_measure_output

                        logging_list.extend(grid_messages)                                
                        measure_dict[m_type][cvi_key] = grid_output
                        volmeasure_dict[m_type][cvi_key] = gridvol_output
                        map_dict[cvi_key] = pd.DataFrame(cvi_map)

                        if m_type=='etdrs' and collate_segmentations:
                            ctmap_args[cvi_key] = [cvi_map, fovea_at_slo, 
                                                  scaleX, eye, angle, dtype,
                                                  grid_output, gridvol_output]
                    
                    else:
                        slo_output, macular_map, (angle, fovea_at_slo), map_messages = map_output
                    logging_list.extend(map_messages)

                    dtype = np.uint64
                    unit = 'thickness' if m_type != 'choroid_vessel' else 'area'
                    grid_measure_output = grid.measure_grid(macular_map, fovea_at_slo, 
                                                            scaleX, eye, rotate=angle, 
                                                            measure_type=m_type, grid_kwds=m_kwd,
                                                            interp=True, plot=save_ind_segmentations, 
                                                            slo=slo_output, dtype=dtype,
                                                            fname=fname+f"_{key}_{m_type}_{unit}_map", 
                                                            save_path=map_save_path)
                    grid_output, gridvol_output, grid_messages = grid_measure_output

                    logging_list.extend(grid_messages)                                                                 
                    measure_dict[m_type][key] = grid_output
                    volmeasure_dict[m_type][key] = gridvol_output

                map_dict[key] = pd.DataFrame(macular_map)
                if m_type=='etdrs' and key in ['retina','choroid'] and collate_segmentations:
                    ctmap_args[key] = [macular_map, fovea_at_slo, 
                                      scaleX, eye, angle, dtype,
                                      grid_output, gridvol_output]
            # break
                
        # Measurement metadata
        metadata["bscan_fovea_x"] = fovea[0]
        metadata["bscan_fovea_y"] = fovea[1]
        metadata["slo_fovea_x"] = fovea_at_slo[0]
        metadata["slo_fovea_y"] = fovea_at_slo[1]
        metadata["acquisition_angle_degrees"] = angle
        metadata["choroid_measure_type"] = chor_measure_type

        # Exit analysis for debugging for ppole scan
        # return metadata, ppole_segs, fovea, vmasks, ppole_keys

        # Plot core maps into single plot and save out
        if collate_segmentations and map_flags[0]==1:
            fig = grid.plot_multiple_grids(ctmap_args)
            fig.savefig(os.path.join(save_path, fname+'.png'), bbox_inches="tight", transparent=False)
            fig.savefig(os.path.join(segmentation_directory, fname+'.png'), bbox_inches="tight", transparent=False)
        plt.close()

        # Macular measurements fixed, so not added to metadata
        # if "square" in measure_dict.keys():
        #     measure_dict["square_grid_size"] = square_kwds["N_grid"]
        #     measure_dict["etdrs_grid_width"] = square_kwds["grid_size"]
        # if "etdrs" in measure_dict.keys():
        #     metadata["etdrs_distance"] = ",".join(np.array(etdrs_kwds["etdrs_microns"]).astype(str))
        
        # Add choroid Ppole traces to retinal segmentations
        if analyse_choroid:
            rtraces = [utils.get_trace(rvf_i[0],0.5,False) for rvf_i in rvfmasks]
            layer_pairwise["CHORupper_CHORlower"] = rtraces
            layer_keys.append("CHORupper")
            layer_keys.append("CHORlower")

        # Organise traces to be saved out - overcomplicated as I am working
        # with pairwise segmentation traces, not individual ones. 
        seg_df = {}
        layer_keys_copied = layer_keys.copy()
        for key, trace_xy_all in layer_pairwise.items():
            for k_idx, k in enumerate(key.split("_")):
                if k in layer_keys_copied:
                    all_ytr = {}
                    for s_idx, trace in enumerate(trace_xy_all):
                        t = trace[k_idx]
                        (xtr, ytr) = t[:,0], t[:,1]
                        try:
                            xst, xen = xtr[[0,-1]]
                            ytr_pad = np.pad(ytr, ((max(xst-1,0), N-xen)), mode="constant")
                            all_ytr[s_idx] = {i:ytr_pad[i] for i in range(N)}
                        except Exception as e:
                            message = f"\nAn exception of type {type(e).__name__} occurred. Error description:\n{e.args[0]}"
                            user_fail = f"Failed to store segmentations for B-scan {s_idx+1}/{N_scans} for layer {k}. Saving as NAs"
                            log_save = [message, user_fail]
                            logging_list.extend(log_save)
                            if verbose:
                                print(message)
                                print(user_fail)
                            all_ytr[s_idx] = {i:np.nan for i in range(N)}
     
                    layer_keys_copied.remove(k)
                    df = utils.nested_dict_to_df(all_ytr).reset_index()
                    df = df.rename({"index":"scan_number"}, inplace=False, axis=1)
                    seg_df[k] = df

        # Add the slo-superimposed maps to seg_df to save out
        # seg_df = {**map_dict, **seg_df}
        # To reduce file size of output xlsx, save maps out as .npy files.
        if save_ind_segmentations:
            for key, macular_map in map_dict.items():
                unit = ''
                if key != 'choroid_CVI':
                    unit = 'thickness' if m_type != 'choroid_vessel' else 'area'
                np.save(os.path.join(map_save_path,f"{fname}_{key}_{unit}_map.npy"), macular_map)

    # Add any disc-centred metadata
    metadata["missing_fovea"] = missing_fovea
    metadata["slo_fovea_x"] = fovea_at_slo[0]
    metadata["slo_fovea_y"] = fovea_at_slo[1]
    if od_centre is not None:
        metadata["optic_disc_overlap_index_%"] = od_overlap
        metadata['optic_disc_overlap_warning'] = od_warning
        metadata['optic_disc_x'] = int(od_centre[0])
        metadata['optic_disc_y'] = int(od_centre[1])
        metadata['optic_disc_radius_px'] = od_radius
        metadata["choroid_measure_type"] = 'vertical'

    # Add units to end of metadata
    metadata["thickness_units"] = "microns"
    metadata["choroid_vascular_index_units"] = 'dimensionless'
    metadata["choroid_vessel_density_units"] = "micron2"
    metadata["area_units"] = "mm2"
    metadata["volume_units"] = "mm3"

    # Exit analysis for debugging for line scan
    # return metadata, layer_pairwise, fovea, vessel_mask, layer_keys

    # If saving out bscan and slo image. If ppole, only saying out bscan at fovea
    # This is automatically done for AV-line scans.
    if save_ind_images:
        cv2.imwrite(os.path.join(save_path,f"{fname}_slo.png"), 
                                (255*slo).astype(np.uint8))
        if scan_location != 'peripapillary':
            cv2.imwrite(os.path.join(save_path,f"{fname}_slo_acquisition_lines.png"), 
                                    (255*slo_acq).astype(np.uint8))
        if N_scans == 1:
            cv2.imwrite(os.path.join(save_path,f"{fname}_bscan.png"), 
                        (255*bscan_data[0]).astype(np.uint8))
        else:
            fovea_slice_num = N_scans//2 + 1
            cv2.imwrite(os.path.join(save_path,f"{fname}_bscan_fovea.png"), 
                        (255*bscan_data[fovea_slice_num]).astype(np.uint8))

    # Save out raw probability vessel segmentation maps if analysing choroid and analysing peripapillary scan
    if scan_location != 'peripapillary':
        vmasks = np.array([(mask[0]>0.5)*mask[1] for mask in rvfmasks]).reshape(bscan_data.shape)
        if save_ind_segmentations and analyse_choroid:
            if N_scans == 1 and scan_type != "Ppole":
                cv2.imwrite(os.path.join(save_path, f"{fname}_chorvessel_mask.png"), (255*vessel_mask).astype(int))
            else:
                np.save(os.path.join(map_save_path, f"{fname}_chorvessel_maps.npy"), vmasks)

    # Organise measurements on Bscans into dataframes
    meta_df = pd.DataFrame(metadata, index=[0])
    if scan_type == "Ppole":
        ppole_key_unit_df = pd.DataFrame({'map_name':ppole_keys, 'units':ppole_units}).drop_duplicates()
        ppole_vol_unit_df = ppole_key_unit_df.copy()
        ppole_vol_unit_df['units'] = '[mm3]'

        # Extract only retinal layers
        retina_layers = np.array(list(KEY_LAYER_DICT.keys())[:-2])
        pairwise_keys = [f"{k1}_{k2}" for (k1,k2) in zip(retina_layers[:-1], retina_layers[1:])]
        all_maps = ppole_key_unit_df.map_name.values

        # Order rows of results dataframes anatomically
        if analyse_choroid:
            choroid_maps = ['choroid', 'choroid_CVI', 'choroid_vessel']
        else:
            choroid_maps = []
        retina_sum_maps = []
        retina_custom_maps = []
        retina_layer_maps = []
        for map_name in all_maps:
            if 'retina' in map_name:
                retina_sum_maps.append(map_name)
            elif 'choroid' not in map_name:
                if map_name in pairwise_keys:
                    retina_layer_maps.append(map_name)
                else:
                    retina_custom_maps.append(map_name)
        ordered_maps = retina_sum_maps+retina_layer_maps+retina_custom_maps+choroid_maps
        
        measure_dfs = []
        measure_grids = []
        volmeasure_dfs = []
        for grid_type in ["etdrs", "square"]:
            if grid_type in measure_dict.keys():
                measure_grids.append(grid_type)

                # Unpack dict of dicts
                df = measure_dict[grid_type]
                df = utils.nested_dict_to_df(df).reset_index()
                df = df.rename({"index":"map_name"}, inplace=False, axis=1)
                df = df.merge(ppole_key_unit_df, on='map_name', how='inner')
                # add unit column and shift
                cols = list(df.columns)
                cols.insert(1, cols.pop(cols.index('units')))
                df = df.loc[:, cols]
                # Order rows anatomically
                df['map_name'] = pd.CategoricalIndex(df['map_name'], ordered=True, categories=ordered_maps)
                df = df.sort_values('map_name').reset_index(drop=True)
                measure_dfs.append(df.drop_duplicates())

                # Same for volume dataframes
                voldf = volmeasure_dict[grid_type]
                voldf = utils.nested_dict_to_df(voldf).reset_index()
                voldf = voldf.rename({"index":"map_name"}, inplace=False, axis=1)
                voldf = voldf.merge(ppole_vol_unit_df, on='map_name', how='inner')
                cols = list(voldf.columns)
                cols.insert(1, cols.pop(cols.index('units')))
                voldf = voldf.loc[:, cols]
                voldf['map_name'] = pd.CategoricalIndex(voldf['map_name'], ordered=True, categories=ordered_maps)
                voldf = voldf.sort_values('map_name').reset_index(drop=True)
                volmeasure_dfs.append(voldf.drop_duplicates())
                
    elif scan_type != "AV-line":
        measure_df = utils.nested_dict_to_df(measure_dict).reset_index()
        measure_df = measure_df.rename({"index":"layer"}, inplace=False, axis=1)

        # rename whole/inner/outer retinal layers
        keys_to_names = ['ILM_BM', 'ILM_ELM', 'ELM_BM']
        names_to_keys = ['retina', 'inner_retina', 'outer_retina']
        for k2n, n2k in zip(keys_to_names, names_to_keys):
            measure_df.replace(k2n, n2k, inplace=True)

        # order map layer names anatomically
        all_pairwise_layers = list(layer_pairwise.keys())
        all_pairwise_layers = names_to_keys + all_pairwise_layers
        ordered_maps = []
        for map_name in all_pairwise_layers:
            if map_name in list(measure_df.layer.values):
                ordered_maps.append(map_name)
        measure_df['layer'] = pd.CategoricalIndex(measure_df['layer'], ordered=True, categories=ordered_maps)
        measure_df = measure_df.sort_values('layer').reset_index(drop=True)
        measure_dfs = [measure_df]

    # Layer keys, ordered anaomtically
    ordered_keys = np.array(list(KEY_LAYER_DICT))
    key_df = pd.DataFrame({"key":layer_keys,"layer":[KEY_LAYER_DICT[key] for key in layer_keys],
                        "layer_number":[np.where(key == ordered_keys)[0][0] for key in layer_keys]})
    key_df = key_df.sort_values("layer_number")
    del key_df["layer_number"]

    with pd.ExcelWriter(os.path.join(save_path, f'{fname}_output.xlsx')) as writer:
        # write metadata
        meta_df.to_excel(writer, sheet_name='metadata', index=False)

        # Save out metadata key and descriptions
        metakeydf = utils.metakey_df
        metakeydf = metakeydf[metakeydf.column.isin(list(meta_df.columns))]
        metakeydf.to_excel(writer, sheet_name='metadata_keys', index=False)

        # write OCT results, either map measurements (for PPole only) or single b-scan measurements
        if scan_type == "Ppole":
            for measure_df, volmeasure_df, grid_type in zip(measure_dfs, volmeasure_dfs, measure_grids):
                measure_df.to_excel(writer, sheet_name=f'{grid_type}_measurements', index=False)
                volmeasure_df.to_excel(writer, sheet_name=f'{grid_type}_volume_measurements', index=False)    
        elif scan_type != "AV-line":
            for measure_df in measure_dfs:
                measure_df.to_excel(writer, sheet_name="oct_measurements", index=False)

        # write SLO measurements
        if slo_analysis_output is not None and analyse_slo:
            for df in slo_measure_dfs:
                if len(df) > 0:
                    z = df.zone.iloc[0]
                    df.to_excel(writer, sheet_name=f'slo_measurements_{z}', index=False)

        # write out segmentations
        for key,df in seg_df.items():
            if key != key.lower():
                name = f"segmentations_{key}"
            else:
                name = f"maps_{key}"
            df.to_excel(writer, sheet_name=name, index=False)

        # write out layer keys
        key_df.to_excel(writer, sheet_name="layer_keys", index=False)

    msg = f"\nSaved out metadata, measurements and segmentations."
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Organise outputs from analysis script
    oct_measures = measure_dfs.copy()
    oct_segmentations_maps = [seg_df]
    if scan_location != 'peripapillary':
        oct_segmentations_maps.append(vmasks)
    if scan_type == "Ppole":
        oct_segmentations_maps.append(map_dict)
        for df in volmeasure_dfs:
            oct_measures.append(df)
        if sq_grid_flag:
            oct_measures = [oct_measures[0],oct_measures[2],oct_measures[1],oct_measures[3]]
    oct_analysis_output = [meta_df, slo, bscan_data] + [oct_measures] + oct_segmentations_maps + [logging_list]

    # final log
    msg = f"\nCompleted analysis of {fname}.\n"
    logging_list.append(msg)
    if verbose:
        print(msg)
    
    # Save out log
    with open(os.path.join(save_path, f"{fname}_log.txt"), "w") as f:
        for line in logging_list:
            f.write(line+"\n")
        
    return slo_analysis_output, oct_analysis_output