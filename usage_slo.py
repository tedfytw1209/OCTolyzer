import argparse
from octolyzer import main_dicom

def main():
    parser = argparse.ArgumentParser(description="Run main_dicom with custom parameters.")

    parser.add_argument("--analysis_csv", default=r"/blue/ruogu.fang/tienyuchang/OCT_EDA/Paired_OCT_Data_ADCON_samples_part2.csv", help="Path to the analysis CSV file")
    parser.add_argument("--output_directory", default=r"/blue/ruogu.fang/tienyuchang/SLO_Output", help="Directory to save the output")
    parser.add_argument("--robust_run", type=int, default=1)
    parser.add_argument("--save_individual_segmentations", type=int, default=0)
    parser.add_argument("--save_individual_images", type=int, default=0)
    parser.add_argument("--preprocess_bscans", type=int, default=0)
    parser.add_argument("--analyse_choroid", type=int, default=0)
    parser.add_argument("--analyse_slo", type=int, default=1)
    parser.add_argument("--custom_maps", nargs="*", default=[])
    parser.add_argument("--analyse_all_maps", type=int, default=0)
    parser.add_argument("--analyse_square_grid", type=int, default=0)
    parser.add_argument("--choroid_measure_type", type=str, default="vertical")
    parser.add_argument("--linescan_roi_distance", type=int, default=1500)

    args = parser.parse_args()

    main_dicom.run(vars(args))

if __name__ == "__main__":
    main()