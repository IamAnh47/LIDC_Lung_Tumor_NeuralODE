import json

records = []

for i in range(1, 451):
    record_id = f"LIDC-IDRI-{i:04d}"
    records.append(record_id)

with open("processed_records.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4)

print("Đã tạo processed_records.json")
