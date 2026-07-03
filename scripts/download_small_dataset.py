import csv
from pathlib import Path

from astroquery.sdss import SDSS


DATA_RELEASE = 17
SAMPLES_PER_CLASS = 10
CANDIDATES_PER_CLASS = 25

OBJECT_CLASSES = (
    "STAR",
    "GALAXY",
    "QSO",
)

OUTPUT_DIRECTORY = Path("data/raw/dataset")
MANIFEST_FILE = OUTPUT_DIRECTORY / "manifest.csv"


def query_candidates(
    object_class: str,
):
    """Find clean SDSS spectra belonging to one object class."""

    query = f"""
        SELECT TOP {CANDIDATES_PER_CLASS}
            specObjID AS specobjid,
            plate,
            mjd,
            fiberID AS fiberid,
            class AS object_class,
            z
        FROM SpecObj
        WHERE class = '{object_class}'
          AND zWarning = 0
        ORDER BY plate, mjd, fiberID
    """

    return SDSS.query_sql(
        query,
        data_release=DATA_RELEASE,
        timeout=120,
    )


def download_class(
    object_class: str,
) -> list[dict[str, object]]:
    """Download a small number of spectra for one class."""

    print(f"\nSearching for {object_class} spectra...")

    candidates = query_candidates(object_class)

    if candidates is None or len(candidates) == 0:
        raise RuntimeError(
            f"No candidates were found for {object_class}."
        )

    class_directory = (
        OUTPUT_DIRECTORY / object_class
    )

    class_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []
    successful_downloads = 0

    for row in candidates:
        if successful_downloads >= SAMPLES_PER_CLASS:
            break

        plate = int(row["plate"])
        mjd = int(row["mjd"])
        fiberid = int(row["fiberid"])
        specobjid = int(row["specobjid"])
        redshift = float(row["z"])

        filename = (
            f"spec-{plate:04d}-"
            f"{mjd}-"
            f"{fiberid:04d}.fits"
        )

        output_file = class_directory / filename

        print(
            f"Downloading {object_class} "
            f"{successful_downloads + 1}/"
            f"{SAMPLES_PER_CLASS}: {filename}"
        )

        try:
            if not output_file.exists():
                spectra = SDSS.get_spectra(
                    plate=plate,
                    mjd=mjd,
                    fiberID=fiberid,
                    data_release=DATA_RELEASE,
                    timeout=120,
                    show_progress=False,
                )

                if spectra is None or len(spectra) == 0:
                    print("  Download returned no spectrum.")
                    continue

                spectrum = spectra[0]

                try:
                    spectrum.writeto(
                        output_file,
                        overwrite=True,
                    )
                finally:
                    spectrum.close()

            successful_downloads += 1

            manifest_rows.append(
                {
                    "specobjid": specobjid,
                    "object_class": object_class,
                    "redshift": redshift,
                    "plate": plate,
                    "mjd": mjd,
                    "fiberid": fiberid,
                    "file_path": str(output_file),
                }
            )

        except Exception as error:
            print(f"  Download failed: {error}")

    if successful_downloads < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Only downloaded {successful_downloads} "
            f"{object_class} spectra."
        )

    return manifest_rows


def main() -> None:
    """Download a balanced demonstration dataset."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for object_class in OBJECT_CLASSES:
        class_rows = download_class(object_class)
        all_rows.extend(class_rows)

    with MANIFEST_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "specobjid",
            "object_class",
            "redshift",
            "plate",
            "mjd",
            "fiberid",
            "file_path",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\nDataset download complete")
    print("-------------------------")
    print(f"Total spectra: {len(all_rows)}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()