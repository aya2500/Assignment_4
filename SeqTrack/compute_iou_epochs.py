import os
import csv

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0


def read_boxes(file_path):
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    boxes = []
    for line in lines:
        parts = [float(x) for x in line.replace(',', ' ').split()]
        if len(parts) >= 4:
            boxes.append(parts[:4])
    return boxes


def compute_sequence_iou(gt_file, pred_file):
    gt_boxes = read_boxes(gt_file)
    pred_boxes = read_boxes(pred_file)
    n = min(len(gt_boxes), len(pred_boxes))
    if n == 0:
        return 0
    ious = [compute_iou(gt_boxes[i], pred_boxes[i]) for i in range(n)]
    return sum(ious) / len(ious)


# === Configuration ===
sequence_list = [
    # 'book-3',
    #'book-10',
    #'book-11',
    # 'book-19',
    # 'coin-3',
    # 'coin-6',
    # 'coin-7',
    'coin-18',
]

root_results = r"D:\Assignment_3\SeqTrack\test\tracking_results\seqtrack"
dataset_root = r"D:\Assignment_3\SeqTrack\data\lasot\coin"
output_csv = "epoch_iou_results.csv"

# === Process ===
epoch_results = []

for folder in sorted(os.listdir(root_results)):
    epoch_dir = os.path.join(root_results, folder, "lasot")
    if not os.path.isdir(epoch_dir):
        continue

    seq_ious = []
    for seq in sequence_list:
        gt_file = os.path.join(dataset_root, seq, "groundtruth.txt")
        pred_file = os.path.join(epoch_dir, f"{seq}.txt")
        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            print(f"⚠️ Missing files for {seq} in {folder}")
            continue

        mean_iou = compute_sequence_iou(gt_file, pred_file)
        seq_ious.append(mean_iou)
        print(f"{folder} | {seq}: IoU = {mean_iou:.4f}")

    if seq_ious:
        avg_iou = sum(seq_ious) / len(seq_ious)
        epoch_results.append((folder, avg_iou))
        print(f"✅ {folder}: Mean IoU = {avg_iou:.4f}\n")

# === Print summary ===
print("\n=================== SUMMARY ===================")
for folder, iou in epoch_results:
    print(f"{folder:<25}  IoU = {iou:.4f}")

# === Save to CSV ===
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch Folder", "Mean IoU"])
    writer.writerows(epoch_results)

print(f"\n💾 Results saved to: {output_csv}")
