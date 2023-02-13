import fiftyone as fo

# The directory containing the dataset to import
dataset_dir = "data/test_destination/"

# The type of the dataset being imported
dataset_type = fo.types.YOLOv5Dataset  # for example

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    )
print(dataset)


# The directory to which to write the exported dataset
export_dir = "data/test_dataset"

# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "ground_truth"  # for example
dataset_type = fo.types.COCODetectionDataset  # for example

# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=dataset_type,
    label_field=label_field,
)