import numpy as np
import cv2


def shift_frame(frame: np.ndarray, shift: int) -> np.ndarray:
    """
    Shifts the frame to the left or right by the specified number of pixels.

    :param frame: np.ndarray : The frame to shift.
    :param shift: int : The number of pixels to shift the frame.
    :return: np.ndarray : The shifted frame.
    """
    height, width = frame.shape[:2]
    black_img = np.zeros((height, width, 3), np.uint8)

    if shift > 0:
        img_left = frame[:, shift:]
        black_img[:, : width - shift] = img_left
    elif shift < 0:
        img_right = frame[:, : width + shift]
        black_img[:, -width - shift :] = img_right

    return black_img


# create generator that yields the shifted frame from original frame
def generate_shifted_frames(frame: np.ndarray, shift: int) -> np.ndarray:
    for i in range(1, shift):
        yield shift_frame(frame, i)  # shift frame to the right


# Load the image
img = cv2.imread("dataset/1.jpg")

# Get the width and height of the image
height, width = img.shape[:2]

for i in range(1, width):
    # Create a black image with the same size as the original
    black_img = np.zeros((height, width, 3), np.uint8)

    # Shift the original image to the left by i pixels
    img_left = img[:, i:]

    # Replace the right side of the black image with the shifted original image
    black_img[:, : width - i] = img_left

    # Save the image
    cv2.imwrite(f"dataset/test_shift/image_left_{i}.jpg", black_img)
