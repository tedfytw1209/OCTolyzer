import logging
import numpy as np
import scipy.integrate as integrate

from octolyzer.measure.bscan.utils import (extract_bounds, curve_length, curve_location, 
                                            get_trace, nearest_coord, construct_line, 
                                            detect_orthogonal_pts, interp_trace)
from octolyzer.measure.bscan.bscan_measurements import compute_area_enclosed


def compute_crosssectionarea(traces, 
                             fovea: [tuple, np.ndarray] = None,
                             scale: tuple[int, int] = (11.49,3.87),
                             macula_rum: int = 3000,
                             offset: int = 15,
                             measure_type: str = "perpendicular",
                             return_lines: bool = False):
    '''
    Given the location of the fovea (regardless of it being superior/inferior to the fovea) and a radius
    from the fovea to measure the area to, compute area approximately parallel to choroid (dictated by local
    tangent at fovea along RPE-Choroid trace).
    '''
    # Instantiate analysis class (for constructing lines and detecting fovea)
    # Extract traces and scales
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]

    # If macula_rum is None, then compute total available area, dictated by the ends of the traces
    # This is used for computing RNFL area
    if (macula_rum is None) and (fovea is None):
        rpechor_st_en, chorscl_st_en = top_chor[[0,-1]], bot_chor[[0,-1]]
        area_bnds_arr = np.concatenate([rpechor_st_en, chorscl_st_en], axis=0)

    # if macula_rum and fovea are specified, then area is determined on macula scans
    else:
        # locate nearest coordinat on RPE-C boundary to selected foveal coordinate
        # and construct tangent line to this cooardinate
        rpechor_refpt, offset_pts = nearest_coord(top_chor, fovea, offset=offset)
        ref_idx = rpechor_refpt[0] - rpechor_stx
        curve_indexes = curve_location(top_chor, distance=macula_rum, ref_idx=ref_idx, scale=scale)
        if curve_indexes is None:
            return None
        reference_pts = np.asarray([top_chor[idx] for idx in curve_indexes])

        # Compute reference points along Choroid-Scleral boundary, given the reference points
        # along the RPE-Choroid boundary
        if measure_type == "perpendicular":
            chorscl_refpts, reference_pts, _, endpoint_errors = detect_orthogonal_pts(reference_pts, traces, offset)
            reference_pts[~endpoint_errors] = np.nan
            chorscl_refpts[~endpoint_errors] = np.nan
        elif measure_type == "vertical":
            st_Bx = bot_chor[0,0]
            chorscl_refpts = bot_chor[reference_pts[:,0]-st_Bx]
        area_bnds_arr = np.concatenate([reference_pts, chorscl_refpts], axis=0)

    # Compute choroid area
    choroid_mm_area, plot_info = compute_area_enclosed(traces, area_bnds_arr, scale, plot=True)

    if return_lines:
        additional_output = [area_bnds_arr, plot_info]
        return choroid_mm_area, *additional_output
    else:
        return choroid_mm_area


def compute_volume(masks, 
                   fovea: np.ndarray = None,
                   scale: tuple[float, float] = (11.49,3.87),
                   bscan_delta: float = None,
                   macula_rum: int = 3000,
                   measure_type: str = "perpendicular"):
    """
    Given a stack of masks for a given macula Ppole scan, compute the choroid volume and the subregional volumes.
    
    NOTE: Because of a lack of sample size for each quadrant, the sum of subregional volumes do not equal the total
    volume. This needs addressed and worked on.
    
    INPUTS:
    ----------------
        mask (np.array) : Array of binary masks. Can input traces instead

        fovea (np.array) : xy-space coordinate of the fovea. If None, then assumed to be evaluating either a volume
            image (without a fovea) or it is simply unknown.
            
        scale (3-tuple) : x-y-micron scale dictated by the device.

        bscan_delta (float) : Micron distance between successive B-scans in the. stack.

        macula_rum (int) : Radius of macular volume to compute in microns

        measure_type : Whether to measure locally perpendicular to the upper boundary ("perpendicular") or measure
        columnwise, i.e. per A-scan ("vertical").

    RETURNS:
    -----------------
        choroid_volume (np.array) : Float measured as cubic millimetres.

        choroid_subr (float) : List of volume measured for the four quadrants (superior-inferior-nasal-temporal).
    """
    # Metric constants. delta_zum is distance between consecutive B-scans 
    # (240 microns for a 31-slice volume stack, 120 microns for 61-slice, etc.)
    N_scans = len(masks)
    fovea_slice_N = N_scans // 2 + 1
    delta_zum = bscan_delta or [120,240][N_scans==31]

    # If masks is an array, we assume stack of binary masks
    if isinstance(masks, np.ndarray):
        traces = []
        for i in range(N_scans):
            traces.append(get_trace(masks[i], None, align=True))

    # If a list, then is already list of trace boundaries
    elif isinstance(masks, list):
        traces = masks.copy()
    else:
        logging.warning(f"Input not astype np.array or list.")
        return 0

    # If fovea is known
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea
    # If not, take middle column of ground truth choroid mask.
    else:
        fovea_trace = traces[fovea_slice_N]
        fovea_mask = masks[fovea_slice_N]
        x_N = int(0.5 * (fovea_trace[0].shape[0] + fovea_trace[1].shape[0]))
        x_st = int(0.5 * (fovea_trace[0][0, 0] + fovea_trace[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = fovea_mask.shape[0] // 2
        ref_pt = np.array([x_half, y_half])

    # Compute cross-sectional areas across macular scans
    chor_areas = []
    chor_dict_areas = dict()
    dum_from_fovea = []
    all_traced = True
    for i in range(-macula_rum // delta_zum + 1, macula_rum // delta_zum + 1):

        s_i = fovea_slice_N + i
        dum_from_fovea.append(i * delta_zum)
        macula_i_rum = np.sqrt(macula_rum ** 2 - (i * delta_zum) ** 2)
        s_trace = traces[s_i]

        # Try computing area
        try:
            choroid_area = compute_crosssectionarea(traces=s_trace, fovea_coord=ref_pt,
                                                    macula_rum=macula_i_rum, scale=scale, 
                                                    measure_type=measure_type)
        except:
            logging.warning(f"Could not make measurement {macula_rum} either side of reference point. Returning 0.")
            return 0

        if choroid_area is None:
            logging.warning(f"Could not make measurement {macula_i_rum} either side of reference point for image {s_i}. Returning 0.")
            return 0
            
        chor_areas.append(choroid_area)
        chor_dict_areas[s_i] = choroid_area
    chor_mm2_areas = np.asarray(chor_areas)

    # Approximate volume
    # Micron distance across macula between slices, coverted to mm
    dmm_from_fovea = 1e-3 * np.asarray(dum_from_fovea)
    slice_z = 1e-3 * delta_zum
    chor_mm3_volume = integrate.simpson(chor_mm2_areas, dx=slice_z, even='avg', axis=-1)

    return chor_mm3_volume, chor_dict_areas


def compute_volume_subregions(masks, 
                              fovea: np.ndarray = None,
                              scale: tuple[float, float] = (11.49,3.87),
                              bscan_delta: float = None,
                              macula_rum: int = 3000,
                              measure_type: str = "perpendicular",
                              offset: int = 15,
                              eye: str = "OD"):
    """
    Given a stack of masks for a given macula Ppole scan, compute the choroid volume and the subregional volumes.
    
    INPUTS:
    ----------------
        mask (np.array) : Array of binary masks. Can input traces instead

        fovea (np.array) : xy-space coordinate of the fovea. If None, then assumed to be evaluating either a volume
            image (without a fovea) or it is simply unknown.

        scale (3-tuple) : x-y-micron scale dictated by the device.

        bscan_delta (float) : Micron distance between successive B-scans in the. stack.

        measure_type : Whether to measure locally perpendicular to the upper boundary ("perpendicular") or measure
        columnwise, i.e. per A-scan ("vertical").

        macula_rum (int) : Radius from fovea to define macular region.

        eye (str) : Type of eye (so as to swap temporal/nasal labelling). Should be automated in future.

    RETURNS:
    -----------------
        choroid_volume (np.array) : Float measured as cubic millimetres.

        choroid_subr (float) : List of volume measured for the four quadrants (superior-inferior-nasal-temporal).
    """
    # Region labelling depending on if volume is for right eye or left eye
    if eye == "OD":
        regions = ["Superior", "Nasal", "Inferior", "Temporal"] 
    elif eye == "OS":
        regions = ["Superior", "Temporal", "Inferior", "Nasal"]

    # Metric constants. delta_zum is distance between consecutive B-scans (240 microns for a 
    # 31-slice volume stack, 120 microns for 61-slice, etc.)
    N_scans = len(masks)
    fovea_slice_N = N_scans // 2
    delta_zum = bscan_delta or [120,240][N_scans==31]

    # Offset the radial distance if we don't have enough scans to sample choroid area to reduce measurement error
    # between choroid volume and sum of subregional volumes
    if N_scans == 31:
        if macula_rum > 500:
            macula_rum += 20
        else:
            macula_rum += 100

    # If masks is an array, we assume stack of binary masks
    if isinstance(masks, np.ndarray):
        traces = []
        for i in range(N_scans):
            traces.append(extract_bounds(masks[i]))

    # If a list, then is already list of trace boundaries
    elif isinstance(masks, list):
        traces = masks.copy()
    else:
        logging.warning(f"Input not astype np.array or list.")
        return 0

    # If fovea is known
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea
    # If not, take middle column of ground truth choroid mask.
    else:
        fovea_trace = traces[fovea_slice_N]
        fovea_mask = masks[fovea_slice_N]
        x_N = int(0.5 * (fovea_trace[0].shape[0] + fovea_trace[1].shape[0]))
        x_st = int(0.5 * (fovea_trace[0][0, 0] + fovea_trace[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = fovea_mask.shape[0] // 2
        ref_pt = np.array([x_half, y_half])

    # We loop over all slices which are contained within the volume specified by macula_rum
    subr_areas = {region:[] for region in regions}
    subr_dict_areas = {region:{} for region in regions}
    for i in range(-macula_rum // delta_zum+1, macula_rum // delta_zum+1):

        # Work out index in scan Z-stack we are processing and how many microns to measures either side
        # of fovea. Extract trace
        s_i = fovea_slice_N + i
        macula_i_rum = np.sqrt(macula_rum ** 2 - (i * delta_zum) ** 2)
        s_trace = traces[s_i]
        top_chor, bot_chor = s_trace
        rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]

        # locate nearest coordinat on RPE-C boundary to selected foveal coordinate
        # and construct tangent line to this cooardinate
        rpechor_refpt, offset_pts = nearest_coord(top_chor, ref_pt, offset=offset)
        ref_idx = rpechor_refpt[0] - rpechor_stx
        curve_indexes = curve_location(top_chor, distance=macula_i_rum, ref_idx=ref_idx, scale=scale)
        if curve_indexes is None:
            logging.warning(f"Could not make measurement {macula_rum} either side of reference point. Returning 0.")
            return 0

        # For non-foveal slice, work out indexes along RPE-C boundary where we split the area into their
        # respective subregions. For foveal slice, we only have 2 subregions, for area to contribute to 
        # temporal and nasal regions.
        if s_i != fovea_slice_N:

            # Work out distance in microns from fovea which defines where two subregions of the 
            # macular volume meet
            subr_dum = np.abs(i) * (delta_zum)

            # If the chord line distance (macula_i_rum) is greater than where the temporal/nasal reference
            # lines intersect the chord line itself, we must split area into central, nasal and temporal
            # subregions. Otherwise, area only contributes centrally (i.e. to superior/inferior)
            if macula_i_rum >= subr_dum: 
                curve_sub_idxs = curve_location(top_chor, distance=subr_dum, ref_idx=ref_idx, scale=scale)
                rpe_subR = np.asarray([top_chor[idx] for idx in [curve_indexes[0], curve_sub_idxs[0]]])
                rpe_subC = np.asarray([top_chor[idx] for idx in curve_sub_idxs])
                rpe_subL = np.asarray([top_chor[idx] for idx in [curve_sub_idxs[1], curve_indexes[1]]])
                rpe_pts = [rpe_subR, rpe_subC, rpe_subL]

                # Defines the subregion index to add areas to. 
                # 3 - Temporal, 2 - Inferior, 1 - Nasal, 0 - Superior
                subregion_idxs = [3, 2, 1] if i < 0 else [3, 0, 1]
            else:
                rpe_subC = np.asarray([top_chor[idx] for idx in curve_indexes])
                rpe_pts =[rpe_subC]
                subregion_idxs = [2] if i < 0 else [0]
  
        else:
            
            rpe_subR = np.asarray([top_chor[idx] for idx in [curve_indexes[0], ref_idx]])
            rpe_subL = np.asarray([top_chor[idx] for idx in [ref_idx, curve_indexes[1]]])
            rpe_pts = [rpe_subR, rpe_subL]
            subregion_idxs = [3, 1]

        # Detect the points along the choroid-scleral boundary where we define the smallest irregular
        # quadrilateral around the enclosed choroid area.
        chor_pts = []
        rpe_pts_new = []
        for i, rpe_ref in enumerate(rpe_pts):
            if measure_type == "perpendicular":
                chor_i, rpe_i, _, endpoint_errors = detect_orthogonal_pts(rpe_ref, traces, offset)
                rpe_i[~endpoint_errors] = np.nan
                chor_i[~endpoint_errors] = np.nan
            elif measure_type == "vertical":
                st_Bx = bot_chor[0,0]
                rpe_i = rpe_ref.copy()
                chor_i = bot_chor[rpe_ref[:,0]-st_Bx]
            chor_pts.append(chor_i)
            rpe_pts_new.append(rpe_i)
        rpe_pts = rpe_pts_new

        # Calculate the choroid area given an enclosed area using rpe_pts and chor_pts and add to one 
        # of the subregion lists
        for j, (rpe_ref, chor_ref) in enumerate(zip(rpe_pts, chor_pts)):
            area_bnds_arr = np.concatenate([rpe_ref, chor_ref], axis=0)
            area = compute_area_enclosed(s_trace, area_bnds_arr, scale, plot=False)
            subr_areas[regions[subregion_idxs[j]]].append(area) #[s_i] = area
            subr_dict_areas[regions[subregion_idxs[j]]][s_i] = area

    # With subregional areas, we can go ahead and compute volumes 
    slice_z = 1e-3 * delta_zum
    subr_volumes = {region:[] for region in regions}
    for key in subr_areas.keys():
       subr_volumes[key] = integrate.simpson(subr_areas[key], dx=slice_z, even='avg', axis=-1)

    return subr_volumes, subr_dict_areas


def compute_all_volumes(masks, 
                        fovea: np.ndarray = None,
                        scale: tuple[float, float] = (11.49,3.87),
                        bscan_delta: float = None,
                        macula_lims: list[int,int,int] = [500, 1500, 3000],
                        measure_type: str = "perpendicular",
                        offset: int = 15,
                        eye: str = "OD"):
    pass


def compute_etdrs_volume(subr_vols: dict, 
                         eye: str ="OD"):
    '''
    Given subregional volumes, compute volume across the standardised ETDRS grid

    Input is assumed to be a dictionary which at the first level corresponds to
    the radial distance from fovea, and second level corresponds to the regional
    locations across the macula
    '''
    low, mid, high = list(subr_vols.keys())
    if eye == "OD":
        etdrs_locs = ["Superior", "Nasal", "Inferior", "Temporal"]
    elif eye == "OS":
        etdrs_locs = ["Superior", "Nasal", "Inferior", "Temporal"]
    
    choroid_etdr_vol = {}
    etdrs_grid = {low:"central", mid:"inner", high:"outer"}
    etdrs_subrgrid = ["central"] + [" ".join([grid, loc]) for grid in list(etdrs_grid.values())[1:] for loc in etdrs_locs]
    for loc in etdrs_subrgrid:
        if loc == "central":
            choroid_etdr_vol[loc] = sum(list(subr_vols[low].values()))
        else:
            in_out, reg_loc = loc.split(" ")
            
            if in_out=="inner":
                choroid_etdr_vol[loc] = subr_vols[mid][reg_loc] - subr_vols[low][reg_loc]
            elif in_out == "outer":
                choroid_etdr_vol[loc] = subr_vols[high][reg_loc] - subr_vols[mid][reg_loc]

    return choroid_etdr_vol