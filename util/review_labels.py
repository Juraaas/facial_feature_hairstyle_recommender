import cv2
import os
import sys
import pandas as pd

LABELS_CSV = "dataset/hair_dataset/labels.csv"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"

HAIR_TYPES = ["straight", "wavy", "curly", "coily", "unknown"]
HAIRLINES  = ["normal", "receding", "uneven", "unknown"]

df = pd.read_csv(LABELS_CSV)

if len(sys.argv) == 3:
    col = sys.argv[1]
    value = sys.argv[2]
    subset = df[df[col] == value].copy()
    print(f"Reviewing {len(subset)} images labeled as {col}={value}")
else:
    subset = df.copy()
    print(f"Reviewing all {len(subset)} images")

changes = 0

for idx, row in subset.iterrows():
    path = os.path.join(TRAIN_IMAGES, row["filename"])
    if not os.path.exists(path):
        continue
    img = cv2.imread(path)
    display = cv2.resize(img, (512, 512))

    label_text = f"hair_type={row['hair_type']} hairline={row['hairline']}"
    cv2.putText(display, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, row["filename"], (10, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Review Labels", display)

    print(f"\n{row['filename']}")
    print(f"current: hair_type={row['hair_type']} hairline={row['hairline']}")
    print("SPACE=OK s/w/c/o/x=hair_type n/r/e/u=hairline q=quit")

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        continue

    elif key == ord('s'): df.at[idx, "hair_type"] = "straight"; changes += 1
    elif key == ord('w'): df.at[idx, "hair_type"] = "wavy";     changes += 1
    elif key == ord('c'): df.at[idx, "hair_type"] = "curly";    changes += 1
    elif key == ord('o'): df.at[idx, "hair_type"] = "coily";    changes += 1
    elif key == ord('x'): df.at[idx, "hair_type"] = "unknown";  changes += 1

    elif key == ord('n'): df.at[idx, "hairline"] = "normal";    changes += 1
    elif key == ord('r'): df.at[idx, "hairline"] = "receding";  changes += 1
    elif key == ord('e'): df.at[idx, "hairline"] = "uneven";    changes += 1
    elif key == ord('u'): df.at[idx, "hairline"] = "unknown";   changes += 1

    if changes % 10 == 0 and changes > 0:
        df.to_csv(LABELS_CSV, index=False)
        print(f"  Auto-saved ({changes} changes)")

cv2.destroyAllWindows()
df.to_csv(LABELS_CSV, index=False)
print(f"\nDone. {changes} changes saved to {LABELS_CSV}")