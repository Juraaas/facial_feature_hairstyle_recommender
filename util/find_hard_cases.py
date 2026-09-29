import os
import cv2
import sys
import csv
import time
from tqdm import tqdm
from src.hair_classifier import classify_hair
from src.hair_segmentation import segment_face
sys.path.insert(0, "backend")

DATASET_DIR = "dataset/celeba/celeba_hq_256"
OUTPUT_DIR = "dataset/hair_dataset"

HARD_CASES_FILE = os.path.join(OUTPUT_DIR, "hard_cases.txt")
DEBUG_CSV_FILE = os.path.join(OUTPUT_DIR, "hair_debug.csv")

CHECKPOINT_EVERY = 500
HARD_CASE_THRESHOLD = 0.70

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(
    f for f in os.listdir(DATASET_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

print(f"Found {len(files)} images.", flush=True)

processed_files = set()

if os.path.exists(DEBUG_CSV_FILE):
    print(f"Found existing checkpoint: {DEBUG_CSV_FILE}", flush=True)

    with open(DEBUG_CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("filename"):
                processed_files.add(row["filename"])

    print(
        f"Already processed: {len(processed_files)} images.",
        flush=True
    )


files_to_process = [
    f for f in files
    if f not in processed_files
]

print(
    f"Remaining: {len(files_to_process)} images.",
    flush=True
)

hard_cases = set()

if os.path.exists(HARD_CASES_FILE):
    with open(HARD_CASES_FILE, "r", encoding="utf-8") as f:
        hard_cases = {
            line.strip()
            for line in f
            if line.strip()
        }

print(
    f"Existing hard cases: {len(hard_cases)}",
    flush=True
)


csv_exists = os.path.exists(DEBUG_CSV_FILE)

csv_file = open(
    DEBUG_CSV_FILE,
    "a",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

if not csv_exists:
    csv_writer.writerow([
        "filename",
        "status",

        "hair_type",
        "hair_conf",

        "hairline",
        "hairline_conf",

        "coverage",

        "read_time",
        "segmentation_time",
        "classification_time",
        "total_time",

        "is_hard_case",
    ])

    csv_file.flush()


def save_checkpoint():
    with open(
        HARD_CASES_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for fname in sorted(hard_cases):
            f.write(fname + "\n")

    csv_file.flush()

    print(
        f"\n[CHECKPOINT] "
        f"hard_cases={len(hard_cases)} | "
        f"saved={HARD_CASES_FILE}",
        flush=True
    )

start_time = time.perf_counter()

total_read_time = 0.0
total_segmentation_time = 0.0
total_classification_time = 0.0
total_processing_time = 0.0

errors = 0


with tqdm(
    files_to_process,
    desc="Processing",
    unit="img"
) as progress:

    for index, fname in enumerate(progress, start=1):

        total_start = time.perf_counter()

        img_path = os.path.join(DATASET_DIR, fname)


        t0 = time.perf_counter()

        img = cv2.imread(img_path)

        read_time = time.perf_counter() - t0

        total_read_time += read_time

        if img is None:

            errors += 1

            csv_writer.writerow([
                fname,
                "read_error",

                "",
                "",
                "",
                "",
                "",

                read_time,
                "",
                "",
                "",

                False,
            ])

            csv_file.flush()

            continue


        t0 = time.perf_counter()

        try:
            hair_mask, _ = segment_face(img)

        except Exception as e:

            errors += 1

            segmentation_time = time.perf_counter() - t0

            csv_writer.writerow([
                fname,
                f"segmentation_error: {type(e).__name__}",

                "",
                "",
                "",
                "",
                "",

                read_time,
                segmentation_time,
                "",
                segmentation_time + read_time,

                False,
            ])

            csv_file.flush()

            continue

        segmentation_time = time.perf_counter() - t0

        total_segmentation_time += segmentation_time


        t0 = time.perf_counter()

        try:
            result = classify_hair(
                img,
                hair_mask
            )

        except Exception as e:

            errors += 1

            classification_time = time.perf_counter() - t0

            csv_writer.writerow([
                fname,
                f"classification_error: {type(e).__name__}",

                "",
                "",
                "",
                "",
                "",

                read_time,
                segmentation_time,
                classification_time,
                read_time + segmentation_time + classification_time,

                False,
            ])

            csv_file.flush()

            continue

        classification_time = time.perf_counter() - t0

        total_classification_time += classification_time

        hair_conf = float(result.get("hair_conf", 0.0))
        hairline_conf = float(result.get("hairline_conf", 0.0))
        coverage = float(result.get("coverage", 0.0))

        hair_type = result.get("hair_type")
        hairline = result.get("hairline")

        is_hard_case = hair_conf < HARD_CASE_THRESHOLD

        if is_hard_case:
            hard_cases.add(fname)

        total_time = time.perf_counter() - total_start

        total_processing_time += total_time

        csv_writer.writerow([
            fname,
            "ok",

            hair_type,
            hair_conf,

            hairline,
            hairline_conf,

            coverage,

            round(read_time, 4),
            round(segmentation_time, 4),
            round(classification_time, 4),
            round(total_time, 4),

            is_hard_case,
        ])

        processed = index
        elapsed = time.perf_counter() - start_time

        avg_time = (
            total_processing_time / processed
            if processed > 0
            else 0
        )

        images_per_sec = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = len(files_to_process) - processed

        eta_seconds = remaining * avg_time

        progress.set_postfix(
            hard=len(hard_cases),
            sec=f"{avg_time:.3f}",
            eta=f"{eta_seconds / 60:.1f}m"
        )

        if index % CHECKPOINT_EVERY == 0:
            save_checkpoint()

save_checkpoint()

csv_file.close()

elapsed_total = time.perf_counter() - start_time

processed_count = len(files_to_process)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Processed this run: {processed_count}")
print(f"Hard cases:         {len(hard_cases)}")
print(f"Errors:             {errors}")

if processed_count > 0:

    print(
        f"Average time/image: "
        f"{total_processing_time / processed_count:.3f}s"
    )

    print(
        f"Images/sec: "
        f"{processed_count / elapsed_total:.2f}"
    )

print(
    f"Total time: "
    f"{elapsed_total / 60:.2f} min"
)

if processed_count > 0:

    print(
        f"\nAverage stage times:"
    )

    print(
        f"  read:          "
        f"{total_read_time / processed_count:.4f}s"
    )

    print(
        f"  segmentation:  "
        f"{total_segmentation_time / processed_count:.4f}s"
    )

    print(
        f"  classification: "
        f"{total_classification_time / processed_count:.4f}s"
    )

print("\nOutput:")
print(f"  Hard cases: {HARD_CASES_FILE}")
print(f"  Debug CSV:  {DEBUG_CSV_FILE}")