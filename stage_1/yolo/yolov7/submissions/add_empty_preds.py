import csv


def make_one(directory: str):
    # Open the first CSV file
    with open("submission_1.csv", "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Open the second CSV file
    with open("submission_2.csv", "r") as f:
        reader = csv.reader(f)
        # Append the rows from the second file to the list
        rows += list(reader)

    # Open the third CSV file
    with open("submission_3.csv", "r") as f:
        reader = csv.reader(f)
        # Append the rows from the third file to the list
        rows += list(reader)

    for i, row in enumerate(rows):
        row[0] = i
        row[-1] = float(row[-1]) * 100
    # Write the combined rows to a new CSV file
    with open("combined.csv", "w", newline="") as f:

        writer = csv.writer(f)
        header = [
            "detection_id",
            "image_name",
            "image_width",
            "image_height",
            "object_class_id",
            "object_class_name",
            "bbox_left",
            "bbox_top",
            "bbox_right",
            "bbox_bottom",
            "confidence",
        ]
        writer.writerow(header)
        writer.writerows(rows)


def check_for_missing_numbers(csv_file):
    # Open the file in read-only mode
    predicted_images = set()
    missing_images = set(list(range(1, 218)))
    max_index = 0
    with open(csv_file, "r") as file:
        for line in list(file)[1:]:
            fields = line.split(",")
            index = int(fields[0])
            image = int(fields[1].split(".")[0])
            missing_images.discard(image)
            max_index = max(index, max_index)

    with open(csv_file, "a+") as file:
        file_writer = csv.writer(file, delimiter=",")
        for image in missing_images:
            print(f"add picture {image}.jpg with no detections.")
            max_index += 1
            row = {
                "detection_id": max_index,
                "image_name": f"{str(image)}.jpg",
                "image_width": 1280,
                "image_height": 720,
                "object_class_id": 2050,
                "object_class_name": "str",
            }
            file_writer.writerow(row.values())


combined = make_one(None)
check_for_missing_numbers("combined.csv")
