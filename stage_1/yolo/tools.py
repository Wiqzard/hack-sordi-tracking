def falsy_path(directory: str) -> bool:
    return bool(
        (
            directory.startswith(".")
            or directory.endswith("json")
            or directory.endswith("zip")
        )
    )


def check_area2(box, area_threshold):
    x1, y1, x2, y2 = (
        box[0],
        box[1],
        box[2],
        box[3],
    )
    return (x2 - x1) * (y2 - y1) > area_threshold


models = {1: [1, 2, 3, 4, 5], 2: [6, 15, 16, 17, 9, 8], 3: [7, 10, 11, 12, 13, 14]}


from utils.tools import dotdict

test_args = dotdict()
test_args.area_threshold_min = 2000
test_args.area_threshold_max = 800000
test_args.partion_assets = 3
