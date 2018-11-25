
import json

with open("train.txt", encoding="utf-8") as f:
    train = json.load(f)
with open("dev.txt", encoding="utf-8") as f:
    dev = json.load(f)

label_set = set()

def save_data(filename, data):
    with open(filename, "w") as f:
        for k, v in data.items():
            f.write("\t".join([k, v["label"], v["query"]]) + "\n")
            label_set.add(v["label"])

save_data("train_raw.txt", train)
save_data("dev_raw.txt", dev)

with open("label_ids.txt", "w") as f:
    label_set = sorted(label_set)
    for i, label in enumerate(label_set):
        f.write(label + "\t" + str(i+1) + "\n")
