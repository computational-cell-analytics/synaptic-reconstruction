import os
import sys
from glob import glob

import pandas as pd

sys.path.append("processing")


def main():
    from parse_table import get_data_root

    data_root = get_data_root()
    val_table = os.path.join(data_root, "Electron-Microscopy-Susi", "Validierungs-Tabelle-v3.xlsx")
    val_table = pd.read_excel(val_table)

    n_check = 0
    for _, row in val_table.iterrows():
        orientation = row["Ribbon-Orientierung"].lower()
        if orientation == "pillar?":
            orientation = "pillar"
        folder = os.path.join(
            data_root, "Electron-Microscopy-Susi", "Analyse",
            row.Bedingung, f"Mouse {row.Maus}", orientation,
            str(row["OwnCloud-Unterordner"])
        )
        assert os.path.exists(folder), folder
        if row.Korrektur != "j":
            continue

        correction_folder = os.path.join(folder, "Korrektur")
        if not os.path.exists(correction_folder):
            correction_folder = os.path.join(folder, "korrektur")
        correction_files = glob(os.path.join(correction_folder, "*.tif"))

        if len(correction_files) == 0:
            print("Could not find expected corrections for:")
            print(folder)
            continue

        corrected_analysis = os.path.join(correction_folder, "measurements.xlsx")
        if not os.path.exists(corrected_analysis):
            print("Could not find correction file for:")
            print(corrected_analysis)
            continue

        n_check += 1

    print("Checked", n_check, "/", (val_table["Korrektur"].values == "j").sum(), "tomograms")


if __name__ == "__main__":
    main()
