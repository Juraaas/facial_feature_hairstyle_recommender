import cv2
import sys 
import os
import pandas as pd

HAIR_TYPES = ["straight", "wavy", "curly", "coily", "unknown"]
HAIRLINES = ["normal", "receding", "uneven", "unknown"]

def label_images(images_dir, output_csv, start_from=0):
    files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(('.png', '.jpg'))
    ])

    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        labeled = set(df_existing["filename"].tolist())
        records = df_existing.to_dict("records")
        print(f"Loaded {len(labeled)} existing labels")
    else:
        labeled = set()
        records = []

    files_todo = [f for f in files if f not in labeled]
    print(f"Remaining: {len(files_todo)} images")

    for i, fname in enumerate(files_todo[start_from:], start_from):
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        
        display = cv2.resize(img, (512, 512))
        cv2.imshow("Hair Labeler", display)

        print(f"\n[{i+1}/{len(files_todo)}] {fname}")
        print("Hair type: s=straight w=wavy c=curly o=coily u=unknown")
        print("Hairline: n=normal r=receding e=uneven u=unknown")
        print("q = quit and save")

        hair_type = None
        hairline = None

        while hair_type is None:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'): hair_type = "straight"
            elif key == ord('w'): hair_type = "wavy"
            elif key == ord('c'): hair_type = "curly"
            elif key == ord('o'): hair_type = "coily"
            elif key == ord('u'): hair_type = "unknown"
            elif key == ord('q'):
                pd.DataFrame(records).to_csv(output_csv, index=False)
                print(f"Saved {len(records)} labels to {output_csv}")
                cv2.destroyAllWindows()
                return
            
        print(f"hair_type = {hair_type}")
        print("Now hairline: n=normal r=receding e=uneven u=unknown")

        while hairline is None:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'): hairline = "normal"
            elif key == ord('r'): hairline = "receding"
            elif key == ord('e'): hairline = "uneven"
            elif key == ord('u'): hair_type = "unknown"
            elif key == ord('q'):
                pd.DataFrame(records).to_csv(output_csv, index=False)
                print(f"Saved {len(records)} labels to {output_csv}")
                cv2.destroyAllWindows()
                return
            
        print(f"hairline = {hairline}")

        records.append({
            "filename": fname,
            "hair_type": hair_type,
            "hairline": hairline,
        })

        if len(records) % 50 == 0:
            pd.DataFrame(records).to_csv(output_csv, index=False)
            print(f"Auto-saved {len(records)} lables")
        
    cv2.destroyAllWindows()
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"\nDone. Saved {len(records)} labels to {output_csv}")

if __name__ == "__main__":
    label_images(
        images_dir= "dataset/hair_dataset/images",
        output_csv= "dataset/hair_dataset/labels.csv",
        start_from= int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )

