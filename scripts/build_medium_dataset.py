import csv
from collections import Counter
from pathlib import Path

import numpy as np

from astrospectra.preprocessing import preprocess_spectrum
from astrospectra.spectrum import load_sdss_spectrum


MANIFEST_FILE = Path(
    "data/manifests/medium_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/medium_dataset.npz"
)

NUMBER_OF_POINTS = 2048

CLASS_NAMES = (
    "STAR",
    "GALAXY",
    "QSO",
)

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


def read_manifest() -> list[dict[str, str]]:
    """Read spectrum information from the medium dataset manifest."""

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}. "
            "Run download_medium_dataset.py first."
        )

    with MANIFEST_FILE.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(
            "The manifest contains no spectra."
        )

    return rows


def main() -> None:
    """Build a fixed-length medium machine-learning dataset."""

    manifest_rows = read_manifest()

    feature_rows = []
    mask_rows = []
    labels = []
    redshifts = []
    spectrum_ids = []
    file_paths = []

    total_spectra = len(manifest_rows)

    for number, row in enumerate(
        manifest_rows,
        start=1,
    ):
        object_class = (
            row["object_class"]
            .strip()
            .upper()
        )

        file_path = Path(
            row["file_path"]
        )

        if object_class not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown object class: {object_class}"
            )

        print(
            f"Processing {number}/{total_spectra}: "
            f"{object_class} — {file_path.name}"
        )

        try:
            spectrum = load_sdss_spectrum(
                file_path
            )

            processed = preprocess_spectrum(
                spectrum,
                number_of_points=NUMBER_OF_POINTS,
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to process spectrum: {file_path}"
            ) from error

        feature_rows.append(
            processed.flux.astype(
                np.float32
            )
        )

        mask_rows.append(
            processed.valid_mask.astype(
                bool
            )
        )

        labels.append(
            CLASS_TO_INDEX[object_class]
        )

        redshifts.append(
            float(row["redshift"])
        )

        spectrum_ids.append(
            str(row["specobjid"])
        )

        file_paths.append(
            str(file_path)
        )

    features = np.stack(
        feature_rows
    )

    masks = np.stack(
        mask_rows
    )

    targets = np.asarray(
        labels,
        dtype=np.int64,
    )

    redshifts_array = np.asarray(
        redshifts,
        dtype=np.float32,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=features,
        masks=masks,
        y=targets,
        class_names=np.asarray(
            CLASS_NAMES,
            dtype="U",
        ),
        redshifts=redshifts_array,
        spectrum_ids=np.asarray(
            spectrum_ids,
            dtype="U",
        ),
        file_paths=np.asarray(
            file_paths,
            dtype="U",
        ),
    )

    class_counts = Counter(
        row["object_class"]
        for row in manifest_rows
    )

    print("\nProcessed medium dataset complete")
    print("---------------------------------")
    print(f"Feature shape: {features.shape}")
    print(f"Mask shape: {masks.shape}")
    print(f"Target shape: {targets.shape}")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name}: "
            f"{class_counts[class_name]}"
        )

    print(f"Saved dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()