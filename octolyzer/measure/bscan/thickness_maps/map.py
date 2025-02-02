import numpy as np
import os
import cv2
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.ndimage import gaussian_filter, median_filter

from skimage.transform import rotate, resize
from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
from skimage import measure

from octolyzer import utils
from octolyzer.measure.bscan import utils as bscan_utils



def detect_angle(slo_acq, 
                 slo_acq_pad,
                 fovea_slice_num=30,
                 oct_N=768,
                 fovea=None, 
                 N_scans=61, 
                 horizontal=True, 
                 plot=False, 
                 logging_list = []):
    """
    Detects the angle at which the macula volume scan is oriented relative to the imaging x-axis, using 
    the SLO image at the fovea. This angle is useful for rotating the thickness maps to align with the capture region 
    on the SLO image during registration.

    Parameters:
    -----------
    slo_acq : numpy.ndarray
        A 3D array representing the SLO image with all OCT B-scan acquisition locations overlaid, with
        dimensions (N, N, C).

    slo_acq_pad : numpy.ndarray
        A 2D array representing the starting padding values in the x- and y-axis for slo_acq

    fovea_slice_num : int, default=30
        Location in the Ppole OCT stack where the fovea-centred OCT B-scan is.
    
    oct_N : int, optional, default=768
        Lateral pixel resolution in the OCT image.

    fovea : numpy.ndarray, default=None
        A 2D array representing the (xy)-coordinates of the fovea on the fovea-centred OCT B-scan. If None,
        this will assume the centre of the acquisition location is the coordinate of the fovea on the SLO.

    N_scans : int, optional, default=61
        The number of slices in the OCT stack, which is used to determine which channels to use to identify the
        fovea-centred acquisiton location.

    horizontal : bool, optional, default=True
        A boolean indicating whether the fovea-centred B-scan is horizontal (True) or vertical (False). 
        This affects the coordinate system and the final angle calculation (only relevant if scan is V-line).

    plot : bool, optional, default=False
        A boolean that, if set to True, will plot the SLO image, the fovea location, and the fitted line used 
        to calculate the angle.

    logging_list : list, optional, default=[]
        List to store warning logs.
        
    Returns:
    --------
    angle_degrees : float
        The angle in degrees of the macula scan from the x-axis. If the B-scan is V-line, the angle 
        is assumed as 90 degrees.

    fovea_in_slo : numpy.ndarray
        A 2D array representing the fovea coordinates in the SLO image.

    acq_centre : numpy.ndarray
        A 2D array representing the centre of the OCT acquisition.

    logging_list : list,
        List of warning logs.

    Notes:
    ------
    - The function uses the RANSAC regressor to fit a line to the fovea-centred B-scan's acquisition location on the SLO.
      The slope of this line is then used to compute the angle of the scan from the x-axis.
    - The angle is calculated in radians, then converted to degrees. If the scan is vertical (V-line), the angle is 
      set to 90 degrees.

    Example:
    --------
    output = detect_angle(slo_acq, slo_acq_pad, fovea_slice_num, fovea, N_scans=31, horizontal=True, plot=True)
    angle_degrees, fovea_in_slo, acq_centre, logging_list = output
    """
    # Image shape and number of pixels to define ROI to locate line and corresponding angle
    _, N, C = slo_acq.shape
    acq_slice_num = N_scans//2

    # If fovea was not located, then set it as None and use centre of acquisition ROI 
    if fovea.sum() == 0:
        msg = 'Cannot locate fovea on OCT B-scan, defaulting fovea on SLO to the centre of OCT acquisition on SLO.'
        if verbose:
            print(msg)
        logging_list.append(msg)
        fovea = None

    # Identify the colour which identifies the acquisition line(s) of interest (fovea-centred and acquisition-centred)
    if N_scans == 1:
        acq_palette = [np.array([0,1,0])]
    else:
        acq_palette = np.array(plt.get_cmap("nipy_spectral")(np.linspace(0.1, 0.9, N_scans)))[...,:-1]

    # Use start and end acquisition locations to define acquisition ROI mask to define acquisition centre.
    where_botline, where_topline = [np.argwhere(skeletonize(np.all(slo_acq == acq_palette[i], axis=2))) for i in [0,N_scans-1]]
    if horizontal:
        where_botline, where_topline = where_botline[:,[1,0]], where_topline[:,[1,0]]
    top_st, top_en = where_botline[where_botline[:,0].argmin()], where_botline[where_botline[:,0].argmax()]
    bot_st, bot_en = where_topline[where_topline[:,0].argmin()], where_topline[where_topline[:,0].argmax()]
    vertices = [top_st, bot_st, bot_en, top_en]
    polygon = np.array(vertices)[:,[1,0]]
    mask = polygon2mask(slo_acq.shape, polygon).astype(float)
    mask_centre = measure.regionprops(mask.astype(int))[0].centroid
    acq_centre = np.array([mask_centre[1], mask_centre[0]]).astype(int)
    
    # Detect acquisition location for fovea and for ROI
    where_fovline = np.argwhere(skeletonize(np.all(slo_acq == acq_palette[fovea_slice_num], axis=2)))
    if horizontal:
        where_fovline = where_fovline[:,[1,0]]
    where_fovline = where_fovline[where_fovline[:,0].argsort()]  
    fov_line = RANSACRegressor().fit(where_fovline[:,0].reshape(-1,1), where_fovline[:,1])

    # Detect fovea on the SLO by stepping along the line of acquisition, interpolated
    # to be discretised by the OCT B-scan lateral pixel resolution, according to the 
    # x-value of the fovea on the OCT B-scan
    if fovea is not None:
        rangex = np.linspace(*where_fovline[[0,-1]][:,0], oct_N).astype(int)
        rangey = fov_line.predict(rangex.reshape(-1,1)).astype(int)
        loa = np.array([rangex, rangey]).T
        fovea_in_slo = loa[fovea[0]]
    else:
        fovea_in_slo = acq_centre.copy()

    # Work out angle of elevation from x-axis
    gradient = fov_line.estimator_.coef_[0]
    angle = np.arctan(gradient)
    angle_degrees = angle*180 / np.pi
    if not horizontal:
        angle_degrees = 90.0

    # If V-line scan, then coordinates are swapped as acquisition line is vertical, 
    # not horizontal
    if not horizontal:
        fovea_in_slo = fovea_in_slo[[1,0]]

    # If plotting
    if plot:
        x_pts = (np.arange(N)).reshape(-1, 1)
        yfov_pts = fov_line.predict(x_pts)
        
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(slo_acq)
        ax.scatter(fovea_in_slo[0], fovea_in_slo[1], c="r", s=100, marker='X', 
                   edgecolors=(0,0,0), label='OCT B-scan fovea', zorder=3)
        ax.scatter(acq_centre[0], acq_centre[1], c="b", s=100, marker='X', 
                   edgecolors=(0,0,0), label='OCT acquisition centre', zorder=3)
        if horizontal:
            ax.plot(x_pts, yfov_pts, c='r', linestyle='--', linewidth=1)
        else:
            ax.plot(yfov_pts, x_pts, c='r', linestyle='--', linewidth=1)
        ax.legend(fontsize=16)
        ax.set_axis_off()

    # Final coordinates require removing padding from slo_acq
    fovea_in_slo -= slo_acq_pad
    acq_centre -= slo_acq_pad
    
    return angle_degrees, fovea_in_slo, acq_centre, logging_list



def measure_thickness(segs, acq_centre, scale, offset=15, oct_N=768, slo_N=768,
                      measure_type="perpendicular", region_thresh=1, 
                      disable_progress=True, logging=[]):
    """
    Measure layer thickness across all posterior pole segmentation slices (B-scans) by calculating 
    the distance between the upper and lower layer boundaries, using the specified measurement method 
    (perpendicular or vertical) for each slice.

    Parameters:
    -----------
    segs : list of numpy.ndarray
        List of layer segmentations for each B-scan. Each element represents a segmentation of 
        the layer (top and bottom boundaries) of a B-scan.

    acq_centre : tuple
        (x,y)-coordinates of the centre of the OCT acquisition ROI, used to align the thickness 
        map with the SLO image.

    scale : tuple
        The scaling factor in microns-per-pixel for the x and y axes, used to convert pixel distances into microns. 
        Format: (micron_per_pixel_x, micron_per_pixel_y).

    offset : int, optional, default=15
        The number of pixels around the reference point used for defining the tangential line to measure 
        thickness. Adjusts how much local changes affect the tangent line.

    oct_N : int, optional, default=768
        Lateral pixel resolution in the OCT image, used for aligning resolution with SLO.

    slo_N : int, optional, default=768
        Lateral pixel resolution in the SLO image, used for aligning resolution with OCT.

    measure_type : {"perpendicular", "vertical"}, optional, default="perpendicular"
        The method used for measuring choroidal measurements:
        - "perpendicular": Measures the thickness perpendicular to the upper boundary (RPE-Choroid).
        - "vertical": Measures the vertical distance between the upper and lower boundaries (per A-scan).

    region_thresh : float, optional, default=1
        Threshold used to post-process layer segmentations.

    disable_progress : bool, optional, default=True
        If True, disables the tqdm progress bar during processing.

    logging : list, optional, default=[]
        List to log any warnings or errors encountered during the execution.

    Returns:
    --------
    ct_data : list of numpy.ndarray
        List of thickness measurements (in microns) for each B-scan, where each entry 
        corresponds to a thickness measurement across the entire scan.

    ct_acqs : list of int
        List of x-coordinates for each B-scan, indicating where the centre of the OCT acquisition
        lies in the segmentation to align with the SLO.

    ct_topStx : list of float
        List of starting x-coordinates for the upper boundary, one for each B-scan.

    logging : list of str
        Updated list of log messages, including warnings or errors encountered during the process.

    Notes:
    ------
    - This function assumes that the segmentation includes both the upper and lower boundaries.
    - The `measure_type` can be either "perpendicular" or "vertical," depending on 
      the layer (choroid or retina).
    - The function handles cases where a B-scan does not have a valid segmentation by logging a warning 
      and skipping that scan, creating an empty area in the map.
    - The output data (`ct_data`, `ct_acqs`, `ct_topStx`) are aligned to the centre of the OCT acquisition 
      for accurate downstream spatial analysis.

    Example:
    --------
    ct_data, ct_acqs, ct_topStx, logging = measure_thickness(segs, acq_centre=(384, 384), scale=(11.48, 3.87))
    """
    N_scans = len(segs)

    ct_data = []
    ct_acqs = []
    ct_topStx = []
    acq_x = int((oct_N/slo_N)*acq_centre[0])
    for idx, bmap in enumerate(tqdm(segs, disable=disable_progress, total=N_scans)):

        # Smart crop, unless ndim is 3, then these are retinal layer segmentations
        if bmap.ndim == 3:
            traces = bmap.copy()
        else:
            try:
                traces = utils.get_trace(bmap, threshold=0.5, align=True)
            except:
                traces = (np.empty((0,2)), np.empty((0,2)))
        top_lyr, bot_lyr = traces
    
        # Catch exception if a B-scan doesn't have any segmentations,
        N_t = top_lyr.shape[0]
        N_b = bot_lyr.shape[0]
        if (N_t == 0) or (N_b == 0):
            fail_msg = f"WARNING: B-scan {idx+1}/{N_scans} does not have a valid trace for this layer. This slice will be empty in the map"
            print(fail_msg)
            logging.append(fail_msg)
            ct_topStx.append(0)
            ct_data.append(-1*np.ones((oct_N)))
            ct_acqs.append(acq_x)
            continue

        # Catch any other unexpected problem
        try: 
            # Select every coordinate along upper boundary to compute thickness at
            st_Tx = top_lyr[0,0]
            ct_topStx.append(st_Tx)
            

            # If measuring perpendicularly to upper boundary, or vertically
            if measure_type == "perpendicular":
                toplyr_pts = top_lyr if offset is None else top_lyr[offset:-offset]
                botlyr_pts, toplyr_pts, _ = bscan_utils.detect_orthogonal_coords(toplyr_pts, 
                                                                                 traces, 
                                                                                 offset=offset, 
                                                                                 tol=2)
            elif measure_type == "vertical":
                toplyr_pts = top_lyr.copy()
                st_Bx = bot_lyr[0,0]
                botlyr_pts = bot_lyr[toplyr_pts[:,0]-st_Bx]
        
            # Combine upper and lower boundary reference points and remove padding
            boundary_pts = np.swapaxes(np.concatenate([toplyr_pts[...,np.newaxis], 
                                                    botlyr_pts[...,np.newaxis]], axis=-1), 1, 2)

            # Compute choroid thickness at each boundary point using scale
            micron_pixel_x, micron_pixel_y = scale
            delta_xy = np.abs(np.diff(boundary_pts, axis=boundary_pts.ndim-2)) * np.array([micron_pixel_x, micron_pixel_y])
            ct_i = np.sqrt(np.square(delta_xy).sum(axis=-1)).mean(axis=1)
            ct_data.append(ct_i)

            # This should only fail if the segmentation doesn't reach as far as the 
            # horizontal pixel location of the acquisiton centre
            try:
                rpe_acqx = np.argwhere(toplyr_pts[:,0] == acq_x)[0][0]

            # Catch this rare exception
            except:
                # Check if segmentation is mostly in left-portion/right-portion,
                # using the horizontal coordinate of the acquisition centre 
                if boundary_pts[:,0,0].mean() < 2*acq_x:
                    rpe_acqx = acq_x - boundary_pts[0][0,0]
                else:
                    rpe_acqx = - boundary_pts[0][0,0]
            ct_acqs.append(rpe_acqx)

        except Exception as e:
            msg_error = f"\nUnknown exception of type {type(e).__name__} occurred. Error description:\n{e.args[0]}"
            msg_fail = f"Failure to measure thickness for B-scan {idx+1}/{N_scans} for current layer. Returning as 0s"
            print(msg_error)
            print(msg_fail)
            logging.extend([msg_error, msg_fail])
            ct_topStx.append(0)
            ct_data.append(-1*np.ones((oct_N)))
            ct_acqs.append(acq_x)
        
    return ct_data, ct_acqs, ct_topStx, logging



def measure_vessels(ves_chorsegs, reg_chorsegs, acq_centre, scale, offset=15, oct_N=768, slo_N=768,
                    measure_type="perpendicular", region_thresh=1, disable_progress=True, logging=[]):
    """
    Measure choroidal vessel area and choroidal vascular index (CVI) across all posterior pole segmentation slices 
    (B-scans), by calculating the vessel area and CVI at each slice using the specified measurement method 
    (perpendicular or vertical) for each B-scan.

    Parameters:
    -----------
    ves_chorsegs : list of numpy.ndarray
        List of binary choroid vessel segmentations for each B-scan. Each element represents a segmentation of 
        the choroidal vessel layer in a B-scan.

    reg_chorsegs : list of numpy.ndarray
        List of binary region choroid segmentations for each B-scan. Each element represents a segmentation of the 
        choroidal region in a B-scan (used for measuring the vessel area and CVI).

    acq_centre : tuple
        (x,y)-coordinates of the centre of the OCT acquisition ROI, used to align the thickness 
        map with the SLO image.

    scale : tuple
        The scaling factor in microns-per-pixel for the x and y axes, used to convert pixel distances into microns. 
        Format: (micron_per_pixel_x, micron_per_pixel_y).

    offset : int, optional, default=15
        The number of pixels around the reference point used for defining the tangential line to measure 
        thickness. Adjusts how much local changes affect the tangent line.

    oct_N : int, optional, default=768
        Lateral pixel resolution in the OCT image, used for aligning resolution with SLO.

    slo_N : int, optional, default=768
        Lateral pixel resolution in the SLO image, used for aligning resolution with OCT.

    measure_type : {"perpendicular", "vertical"}, optional, default="perpendicular"
        The method used for measuring choroidal measurements:
        - "perpendicular": Measures the thickness perpendicular to the upper boundary (RPE-Choroid).
        - "vertical": Measures the vertical distance between the upper and lower boundaries (per A-scan).

    region_thresh : float, optional, default=1
        Threshold used to post-process layer segmentations.

    disable_progress : bool, optional, default=True
        If True, disables the tqdm progress bar during processing.

    logging : list, optional, default=[]
        List to log any warnings or errors encountered during the execution.

    Returns:
    --------
    cv_data : list of numpy.ndarray
        List of choroidal vessel area measurements (in square microns) for each B-scan, 
        where each entry corresponds to the vessel area across the entire scan.

    cvi_data : list of numpy.ndarray
        List of choroidal vessel index (CVI) measurements for each B-scan, where each entry corresponds 
        to the CVI across the entire scan.

    cv_fovs : list of int
        List of x-coordinates for each B-scan, indicating where the centre of the OCT acquisition
        lies in the segmentation to align with the SLO.

    cv_topStx : list of float
        List of starting x-coordinates for the upper choroid boundary (RPE-Choroid), one for each B-scan.

    logging : list of str
        Updated list of log messages, including warnings or errors encountered during the process.

    Notes:
    ------
    - This function assumes that the segmentation includes both the upper (RPE-Choroid) and lower 
      (Choroid-Sclera) choroidal boundaries, as well as the choroidal vessel layer.
    - The `measure_type` can be either "perpendicular" (preferred) or "vertical," depending on 
      the type of measurement needed for choroidal vessel area and CVI.
    - The function handles cases where a B-scan does not have a valid segmentation by logging a warning 
      and skipping that scan.
    - The output data (`cv_data`, `cvi_data`, `cv_fovs`, `cv_topStx`) are aligned to the OCT acquisition centre 
      for accurate spatial analysis.

    Example:
    --------
    cv_data, cvi_data, cv_fovs, cv_topStx, logging = measure_vessels(ves_chorsegs, reg_chorsegs, acq_centre=(350, 250), 
                                                                    scale=(1.0, 1.0))
    """
    N_scans = len(ves_chorsegs)
    cv_data = []
    cvi_data = []
    cv_fovs = []
    cv_topStx = []
    acq_x = int((oct_N/slo_N)*acq_centre[0])
    for idx, (v_binmap, reg_bmap) in enumerate(tqdm(zip(ves_chorsegs, reg_chorsegs), total=N_scans, disable=disable_progress)):

        # Smart crop
        # Error handling if B-scan clipped so region mask is invalid, making vessel mask invalid
        try:
            traces = utils.get_trace(reg_bmap, threshold=0.5, align=True)
            binmap = v_binmap * reg_bmap
        except:
            traces = (np.empty((0,2)), np.empty((0,2)))
            binmap = np.zeros_like(reg_bmap)
        top_chor, bot_chor = traces
    
        
        # Catch exception if a B-scan doesn't have any segmentations,
        N_t = top_chor.shape[0]
        N_b = bot_chor.shape[0]
        if (N_t == 0) or (N_b == 0):
            fail_msg = f"WARNING: B-scan {idx+1}/{N_scans} does not have a valid trace for this layer. This slice will be empty in the map"
            print(fail_msg)
            logging.append(fail_msg)
            cv_topStx.append(0)
            cv_data.append(-1*np.ones((oct_N)))
            cvi_data.append(-1*np.ones((oct_N)))
            cv_fovs.append(acq_x)
            continue

        # Catch any other unexpected problem
        try: 
            # Select every coordinate along upper boundary to compute cvi at
            st_Tx = top_chor[0,0]
            cv_topStx.append(st_Tx)

            # If measuring perpendicularly to upper boundary, or vertically
            if measure_type == "perpendicular":
                rpechor_pts = top_chor if offset is None else top_chor[offset:-offset]
                vessel_pixels = np.zeros(rpechor_pts.shape[0])
                chorscl_pts, rpechor_pts, perps = bscan_utils.detect_orthogonal_coords(rpechor_pts, 
                                                                                       traces, 
                                                                                       offset=offset, 
                                                                                       tol=2)
                max_M = binmap.shape[0]
                perps[:,0] = perps[:,0].clip(0, oct_N-1)
                perps[:,1] = perps[:,1].clip(0, oct_N-1)
                vessel_pixels = binmap[perps[:,1], perps[:,0]].sum(axis=-1)
                region_pixels = reg_bmap[perps[:,1], perps[:,0]].sum(axis=-1)
                cvi_pixels = vessel_pixels / region_pixels
            elif measure_type == "vertical":
                rpechor_pts = top_chor.copy()
                vessel_pixels = binmap[:,rpechor_pts[:,0]].sum(axis=0)
                region_pixels = reg_bmap[:,rpechor_pts[:,0]].sum(axis=0)
                cvi_pixels = vessel_pixels / region_pixels

            # Compute how much vessel area was picked up 
            micron_pixel_x, micron_pixel_y = scale
            pixel_micron_area = micron_pixel_x * micron_pixel_y
            vessel_area = vessel_pixels * pixel_micron_area
            cvi_data.append(cvi_pixels)
            cv_data.append(vessel_area)

            # This should only fail if the segmentation doesn't reach as far as the 
            # horizontal pixel location of the OCT acquisition centre
            try:
                rpe_acqx = np.argwhere(rpechor_pts[:,0] == acq_x)[0][0]

            # Catch this rare exception
            except:
                # Check if segmentation is mostly in left-portion/right-portion,
                # using the horizontal coordinate of the OCT acquisition centre 
                if rpechor_pts[:,0].mean() < 2*acq_x:
                    rpe_acqx = acq_x - rpechor_pts[0,0]
                else:
                    rpe_acqx = - rpechor_pts[0,0]
            cv_fovs.append(rpe_acqx)

        except Exception as e:
            msg_error = f"\nUnknown exception of type {type(e).__name__} occurred. Error description:\n{e.args[0]}"
            msg_fail = f"Failure to measure thickness for B-scan {idx+1}/{N_scans} for current layer. Returning as 0s"
            print(msg_error)
            print(msg_fail)
            logging.extend([msg_error, msg_fail])
            cv_topStx.append(0)
            cv_data.append(-1*np.ones((oct_N)))
            cvi_data.append(-1*np.ones((oct_N)))
            cv_fovs.append(acq_x)
        
    return cv_data, cvi_data, cv_fovs, cv_topStx, logging



def build_chth_map(ct_data, ct_acqs, ct_topStx, acq_centre, N_stack, slo_Vs, 
                   slo_N=768, line_distance=10, type="bilinear", verbose=False):
    """
    Build a thickness/density map to scale with the SLO image.

    Parameters:
    -----------
    ct_data : list of numpy.ndarray
        A list containing the thickness/density data for each B-scan slice. Each element represents the 
        thickness values along each B-scan.

    ct_acqs : list of int
        List of x-coordinates for each B-scan, indicating where the centre of the OCT acquisition
        lies in the segmentation to align with the SLO.

    ct_topStx : list of float
        A list of starting x-coordinates for the upper boundary for each B-scan.

    acq_centre : tuple
        (x,y)-coordinates of the centre of the OCT acquisition ROI, used to align the thickness 
        map with the SLO image.

    N_stack : int
        The total number of B-scan slices in the OCT volume.

    slo_Vs : tuple
        Vertical information on the OCT volume capture ROI on the SLO image. Format: (slo_V, slo_V_t, slo_V_b) 
        where:
        - slo_V : slo_V_t + slo_V_b, i.e., the total number of pixel rows of the OCT capture region on the SLO image
        - slo_V_t : top vertical pixel position (row) of the OCT capture region on the SLO image
        - slo_V_b : bottom vertical pixel position (row) of the OCT capture region on the SLO image

    slo_N : int, optional, default=768
        Lateral resolution of the SLO image.

    line_distance : int, optional, default=10
        The distance between successive OCT B-scans in pixels, used for smoothing the thickness/density map. 
        Affects the Gaussian smoothing kernel size.

    type : str, optional, default="bilinear"
        The interpolation method used for scaling the choroidal thickness map. Options include "bilinear" 
        and others supported by `torch.nn.functional.interpolate`.

    verbose : bool, optional, default=False
        If True, displays intermediate plots of the thickness/density map during processing.

    Returns:
    --------
    ct_padded : numpy.ndarray
        The scaled and smoothed thickness/density map, padded and aligned to the macular ROI on the SLO image.

    ct_mask_padded : numpy.ndarray
        The binary mask indicating where the thickness/density map was measured, aligned and padded to the SLO image.

    Notes:
    ------
    - This function interpolates and smooths the thickness/density map to align it with the SLO image, 
      scaling it vertically and horizontally as needed.
    - The function uses a Gaussian filter to smooth the map, and padding is applied to ensure proper alignment.
    - The `ct_map` is padded using constant values (-1) at locations where no measurements exist, 
      and smoothing is applied to prevent interpolation artefacts.
    - The function optionally logs and visualizes the intermediate steps when `verbose=True`.

    Example:
    --------
    ct_padded, ct_mask_padded = build_chth_map(ct_data, ct_acqs, ct_topStx, acq_centre=(384, 384), 
                                               N_stack=61, slo_Vs=(600, 290, 310), slo_N=768)
    """  
    # Build map of choroid thicknesses per slice
    slo_V, slo_V_t, slo_V_b = slo_Vs
    ct_lengths = [ct.shape[0] for ct in ct_data]
    max_shape = max(ct_lengths)
    max_ct = np.argmax(ct_lengths)
    acq_x, acq_y = acq_centre

    # In order to prevent any interpolation artefacts, create a smooth version of the thickness/density
    # map which is just a padded original, duplicate edge values
    ct_map = np.zeros((N_stack, slo_N))
    ct_smooth = np.zeros((N_stack, slo_N))
    for i, (ct_i, ct_a, ct_l, stX) in enumerate(zip(ct_data, ct_acqs, ct_lengths, ct_topStx)):
        pad_l = acq_x - ct_a 
        pad_r = slo_N - (pad_l + ct_l)
        ct_map[i] = np.pad(ct_i, (pad_l, pad_r), mode="constant", constant_values=-1)
        ct_smooth[i] = np.pad(ct_i, (pad_l, pad_r), mode="edge")
    ct_mask = (ct_map > -1).astype(float)
    
    if verbose:
        fig, ax = plt.subplots(1,1,figsize=(9,7))
        hmax = sns.heatmap(ct_map[::-1],
                    cmap = "rainbow",
                    alpha = 0.5,
                    zorder = 2,
                    ax = ax)

    # Vertical interpolation to scale to ChTh map to the SLO
    # Interpolate both the binary map of where the original choroid map was defined
    # and the smoothed version, and then set everywhere not originally measured at to 0
    ct_maskT = torch.tensor(ct_mask).unsqueeze(0).unsqueeze(0)
    ct_mask_scaled = F.interpolate(ct_maskT, size=(slo_V, slo_N), mode="nearest").squeeze(0).squeeze(0).numpy().astype(bool)
    ct_mapT = torch.tensor(ct_smooth).unsqueeze(0).unsqueeze(0)
    ct_scaled = F.interpolate(ct_mapT, size=(slo_V, slo_N), mode=type).squeeze(0).squeeze(0).numpy()

    # Smooth using a isotropic Gaussian window with standard eviation dependent on the distance
    # between B-scans
    ct_scaled = gaussian_filter(ct_scaled, sigma=2*np.sqrt(line_distance))
    ct_scaled[~ct_mask_scaled] = -1
    if verbose:
        fig, ax = plt.subplots(1,1,figsize=(9,7))
        hmax = sns.heatmap(ct_scaled[::-1],
                    cmap = "rainbow",
                    alpha = 0.5,
                    zorder = 2,
                    ax = ax)

    # Pad vertically to align with macular ROI on SLO
    ct_M, ct_N = ct_scaled.shape
    pad_M = (max(0, acq_y-slo_V_t),
             slo_N - (slo_V+max(0, acq_y-slo_V_t)))

    ct_mask_padded = np.pad(ct_mask_scaled, 
                            ((pad_M[1], pad_M[0]),(0, 0)), 
                            mode="constant", 
                            constant_values=False)[::-1]
    ct_padded = np.pad(ct_scaled, 
                       ((pad_M[1], pad_M[0]),(0,0)))[::-1]

    if verbose:
        fig, ax = plt.subplots(1,1,figsize=(9,7))
        hmax = sns.heatmap(ct_padded,
                    cmap = "rainbow",
                    alpha = 0.5,
                    zorder = 2,
                    ax = ax)
        ax.scatter(acq_x, acq_y, s=100, marker='X', edgecolors=(0,0,0), c='b', zorder=4)
    
    return ct_padded, ct_mask_padded



def trim_map(depth_map, mask, length=2):
    """
    Trims extreme values from a given map by applying a mask. This function is typically used when
    `rotate_slo=False`, i.e., when rotating the map onto the SLO. It trims the edges where rotation interpolation 
    may lead to extreme or unwanted values.

    Parameters:
    -----------
    depth_map : numpy.ndarray
        The thickness/density map (e.g., a choroid thickness heatmap) to be trimmed. This is the array 
        that will be modified based on the mask to remove extreme values at the boundaries.

    mask : numpy.ndarray
        A binary mask (same shape as `map`) used to define the region of interest. Values outside the 
        mask will be trimmed, and the map will be updated accordingly.

    length : int, optional, default=2
        The number of boundary iterations to apply when trimming the map. This defines how many times 
        the function will loop over the boundary to further refine the mask and trim extreme values.

    Returns:
    --------
    new_map : numpy.ndarray
        The modified map with trimmed extreme values based on the provided mask.

    Notes:
    ------
    - The function creates a copy of the input `map` to avoid modifying the original map.
    - The `mask` is converted to an 8-bit unsigned integer type and processed using cv2 to find and trim 
      boundaries. The boundary points of the mask are progressively reset to 0, effectively trimming 
      the map at those locations.
    - Extreme map values outside the valid mask region are set to `-1`.
    
    Example:
    --------
    trimmed_map = trim_map(original_map, boundary_mask, length=3)
    # This will trim the extreme values from `original_map` based on `boundary_mask` with 3 iterations 
    # over the boundary and return the modified `trimmed_map`.
    """
    new_map = depth_map.copy()
    mask = 255*(mask.astype(np.uint8))
    for i in range(length):
        threshold = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
        boundary = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].squeeze()
        mask[boundary[:,1], boundary[:,0]] = 0
    new_map[~mask.astype(bool)] = -1
    return new_map




def construct_map(slo, slo_acq, slo_acq_pad, 
                  reg_chorsegs, fovea, fovea_slice_num, 
                  scale=(11.49,3.87), bscan_delta=120, slo_N=768, oct_N=768, 
                  measure_type="vertical", rotate_slo=False,
                  ves_chorsegs=None, log_list = []):
    """
    Wrapper function to build a paired SLO and thickness/density map.

    Parameters:
    -----------
    slo : numpy.ndarray
        Clean SLO image without any acquisition locations or annotations.
    
    slo_acq : numpy.ndarray
        SLO scan with all B-scan acquisition locations superimposed.

    slo_acq_pad : numpy.ndarray
        A 2D array representing the starting padding values in the x- and y-axis for slo_acq

    reg_chorsegs : list of numpy.ndarray
        List of region segmentations used to generate thickness/density map.

    fovea : tuple of int
        Pixel (x,y)-coordinate of the fovea in the fovea-centered OCT B-scan.

    fovea_slice_num : int
        The slice number that corresponds to the fovea-centered OCT B-scan in the stack of B-scans.

    scale : tuple of float, optional, default=(11.49, 3.87)
        Horizontal and vertical scaling factors of the B-scan in microns-per-pixel.

    bscan_delta : int, optional, default=120
        Micron distance between parallel macular B-scans in the volume.

    slo_N : int, optional, default=768
        Lateral pixel resolution of the SLO image, used to align the resolution of the SLO and B-scan together.

    oct_N : int, optional, default=768
        Lateral pixel resolution of the OCT B-scan.

    measure_type : str, optional, default="vertical"
        Defines the method of measuring thickness/density: either "vertical" (measuring column-wise)
        or "perpendicular" (measuring perpendicular to the upper boundary, preferred for choroid).

    rotate_slo : bool, optional, default=False
        Whether to rotate the SLO image to match the thickness/density map, or rotate the map to match the SLO.

    ves_chorsegs : list of numpy.ndarray, optional, default=None
        List of choroid vessel segmentations. If provided, vessel density map is computed.

    log_list : list, optional, default=[]
        A list to store logs.

    Returns:
    --------
    slo_output : numpy.ndarray
        The output SLO image, possibly rotated to match the thickness/density map.

    chmap_output : numpy.ndarray
        The thickness/density map aligned with the SLO image and OCT capture region.

    angle_and_fovea : tuple
        The angle of rotation and the fovea coordinates in the SLO image.

    log_list : list
        Updated list of logs and any exceptions that occurred.

    cvi_map_output : numpy.ndarray, optional
        If vessel segmentations are provided, the output CVI map aligned with the SLO image.

    Notes:
    ------
    - This function creates a paired SLO and thickness/density map, optionally with a vessel density map.
    - It interpolates the thickness/density map from OCT data to match the SLO image resolution.
    - Choroid vessel segmentations are optional, but if provided, choroid vessel density is included in the output.
    - The function adjusts the image resolution to match the alignment of the SLO and OCT B-scan images.

    Example:
    --------
    slo_output, chmap_output, (angle, fovea_in_slo), log_list = construct_map(slo, slo_at_fovea, reg_chorsegs, 
                                                                             fovea, fovea_slice_num, scale=(11.49, 3.87))
    """
    # Core parameters
    N_stack = len(reg_chorsegs)
    scale_X, scale_Y = scale
    N, M = (slo_N, slo_N)
    verbose = False
    
    # Scan number at the centre of the OCT acquisition
    acq_slice_num = N_stack//2
    
    # vertical distance to interpolate and sample ChTh heatmap from
    line_distance = bscan_delta/scale_X
    delta_b, delta_t = np.array([acq_slice_num, N_stack-acq_slice_num-1])
    slo_V_t = int(delta_t*line_distance)
    slo_V_b = int(delta_b*line_distance)
    slo_V = slo_V_t+slo_V_b
    slo_Vs = (slo_V, slo_V_t, slo_V_b)
    
    # Detect angle of rotation for SLO to map to ChTh map
    # slo_at_fovea: SLO scan with fovea B-scan acquisition location superimposed
    # fovea_in_slo: pixel coordinates of the fovea on the SLO image, important to measuring thickness map.
    # acq_centre: centre of OCT acquisition, important for centring thickness map
    # slo: clean SLO image without any acquisition locations
    output = detect_angle(slo_acq, slo_acq_pad, fovea_slice_num, oct_N, fovea, N_stack, logging_list=log_list, plot=verbose)
    angle, fovea_in_slo, acq_centre, log_list = output
    
    if rotate_slo:
        slo_acq_output = rotate(slo_acq, angle=angle, center=acq_centre, resize=False)
        slo_output = rotate(slo, angle=angle, center=acq_centre, resize=False)
    else:
        slo_acq_output = slo_acq.copy()
        slo_output = slo.copy()
    
    # Measure choroid thickneess for every slice
    if ves_chorsegs is None:
        chor_data, chor_fovs, chor_stxs, log = measure_thickness(reg_chorsegs, 
                                                                 fovea_in_slo, 
                                                                 scale, 
                                                                 oct_N=oct_N,
                                                                 slo_N=slo_N,
                                                                 logging=[],
                                                                 measure_type=measure_type)
        fname_type = "_region"
    
    # Measure choroid vessel density along every slice
    else:
        chor_data, chor_cvi_data, chor_fovs, chor_stxs, log = measure_vessels(ves_chorsegs, 
                                                                              reg_chorsegs, 
                                                                              fovea_in_slo, 
                                                                              scale, 
                                                                              logging=[],
                                                                              oct_N=oct_N,
                                                                              slo_N=slo_N,
                                                                              measure_type=measure_type)
        fname_type = "_vessel"
    log_list.extend(log)

    # Interpolate map to a heatmap aligned with SLO resolution
    ch_map, ch_mask = build_chth_map(chor_data, chor_fovs, chor_stxs, fovea_in_slo, 
                                     N_stack, slo_Vs, slo_N=slo_N, line_distance=line_distance, verbose=verbose)
    if ves_chorsegs is not None:
        cvi_map, _ = build_chth_map(chor_cvi_data, chor_fovs, chor_stxs, fovea_in_slo, 
                                     N_stack, slo_Vs, slo_N=slo_N, line_distance=line_distance, verbose=verbose)
    
    # If rotating the depth map, not the SLO
    if not rotate_slo:
        chmap_output = rotate(ch_map, angle=-angle, center=acq_centre, resize=False, 
                              preserve_range=True, cval=-1, mode="constant")
        chmask_output = rotate(ch_mask.astype(bool), angle=-angle, center=acq_centre, resize=False, 
                               cval=False, mode="constant", order=0)
        chmap_output = trim_map(chmap_output, chmask_output)
        if ves_chorsegs is not None:
            cvimap_output = rotate(cvi_map, angle=-angle, center=acq_centre, resize=False, 
                              preserve_range=True, cval=-1, mode="constant")
            cvimap_output = trim_map(cvimap_output, chmask_output)
    
    # If rotating the SLO
    else:
        chmap_output = ch_map.copy()
        chmask_output = ch_mask.copy()
        chmap_output[~chmask_output] = -1
        if ves_chorsegs is not None:
            cvimap_output = cvi_map.copy()
            cvimap_output[~chmask_output] = -1
    
    if ves_chorsegs is None:
        return slo_output, chmap_output, (angle, fovea_in_slo, acq_centre), log_list
    else:
        return slo_output, chmap_output, (angle, fovea_in_slo, acq_centre), cvimap_output, log_list


def plot_slo_map(slo, depth_map, fname="", save_path="", transparent=False, cbar=True, clip=None):
    """
    Plots a thickness/density map (e.g., a heatmap) of the retina/choroid overlaid on an SLO image. 
    The function visualises the relationship between a thickness map and the underlying SLO image, 
    optionally saving the figure to a file.

    Parameters:
    -----------
    slo : numpy.ndarray or None
        The SLO image (grayscale) that serves as the background for the heatmap. 
        If `None`, only the heatmap will be plotted.

    depth_map : numpy.ndarray
        The thickness/density map to be overlaid on the SLO image. This map could represent any relevant 
        measurement or feature, such as a heatmap of choroid thickness.

    fname : str, optional, default=""
        The filename for saving the plot. If an empty string is provided, the figure will not be saved.

    save_path : str, optional, default=""
        The directory path where the plot will be saved. If `fname` is provided and `save_path` is 
        an empty string, the file will be saved in the current working directory.

    transparent : bool, optional, default=False
        Whether the saved figure should have a transparent background. Relevant only if `fname` is 
        provided and the figure is being saved.

    cbar : bool, optional, default=True
        Whether to display a color bar alongside the heatmap. If `False`, the color bar will not be shown.

    clip : float, optional, default=None
        The upper limit for the heatmap's color scale. If `None`, the maximum value will be set to the 
        99.5th percentile of the values in the map (excluding -1). If a specific value is provided, 
        that value will be used to clip the heatmap color scale. Good for removing outliers in visualisation
        from interpolation artefacts during map generation.

    Returns:
    --------
    None
        The function does not return anything, but generates and optionally saves the plot.

    Notes:
    ------
    - The function uses a heatmap to display the `map` values, applying a `rainbow` colormap with 
      transparency (alpha=0.5) over the SLO image.
    - If `clip` is provided, the heatmap's color scale will be clipped to that value. Otherwise, it 
      will be based on the 99.5th percentile of the `map` values.
    - If `slo` is provided, the SLO image will be displayed beneath the heatmap, with proper alignment.

    Example:
    --------
    plot_slo_map(slo_image, choroid_map, fname="choroid_map.png", save_path="/path/to/save", transparent=True)
    # This will generate a plot of `choroid_map` overlaid on `slo_image` and save it as "choroid_map.png".
    """
    # if clipping heatmap
    mask = depth_map < 0
    if clip is None:
        vmax = np.quantile(depth_map[depth_map != -1], q=0.995)
    else:
        vmax = clip
    
    if cbar:
        figsize=(9,7)
    else:
        figsize=(9,9)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    hmax = sns.heatmap(depth_map,
                cmap = "rainbow",
                alpha = 0.5,
                vmax = vmax,
                zorder = 2,
                mask=mask,
                ax = ax,
                cbar=cbar)
    if slo is not None:
        hmax.imshow(slo, cmap="gray",
                aspect = hmax.get_aspect(),
                extent = hmax.get_xlim() + hmax.get_ylim(),
                zorder = 1)
    ax.set_axis_off()
    if fname != "": 
       fig.savefig(os.path.join(save_path, fname), pad_inches=0,
                   bbox_inches="tight", transparent=transparent)
       plt.close()