import random
import math
import cv2
import numpy as np
from PIL import Image

from scipy.ndimage import binary_erosion, binary_dilation
import skimage
from skimage import measure, draw

import dlib
from facenet_pytorch.models.mtcnn import MTCNN

mtcnn_detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("libs/shape_predictor_68_face_landmarks.dat")


def face_cutout(
    image,
    original=None,
    landmarks=None,
    mask=None,
    probability=0.5,
    cutout_fill=0,
    threshold=0.3,
):
    choice = random.choice([0, 1])
    if choice == 0:
        image = sensory_cutout(image, original, landmarks, mask, probability, cutout_fill, threshold)
    elif choice == 1:
        image = convex_hull_cutout(image, original, mask, probability, cutout_fill, threshold)

    return image


def sensory_cutout(
    image,
    original=None,
    landmarks=None,
    mask=None,
    probability=0.5,
    cutout_fill=0,
    threshold=0.3,
):
    """[Face Cutout using sensory points. The points in this implementation are predicted by MTCNN.]

    Args:
        image ([numpy.ndarray]): [Input face image for cutout]
        mask ([numpy.ndarray], optional): [Difference mask for fake image]. Defaults to None.
        landmarks ([numpy.ndarray], optional):
            [numpy array containing 5 face landmarks detected by MTCNN NOT DLib]. Defaults to None.
        probability (float, optional): [probability of cutout]. Defaults to 0.5.
        cutout_fill (int, optional): [cutout fill value]. Defaults to 0. [-1 = Random Fill]
        threshold (float, optional): [threshold for fake images]. Defaults to 0.3.

    Returns:
        [numpy.ndarray]: [Augmented image]
    """
    if random.random() > probability:
        return image


    # If MTCNN landmarks are not provided
    if landmarks is None:
        if original is not None:
            frame_img = Image.fromarray(original[..., ::-1])
        else:
            frame_img = Image.fromarray(image[..., ::-1])
        batch_boxes, conf, landmarks = mtcnn_detector.detect(frame_img, landmarks=True)
        if landmarks is not None and len(landmarks) == 5:
            landmarks = np.around(landmarks[0]).astype(np.int16)
        else:
            return image # If no landmark can be detected, return unaugmented image

    choice = random.choice([0, 1, 2])
    if choice == 0:
        image = remove_eyes(image, landmarks, mask, cutout_fill, threshold)
    elif choice == 1:
        image = remove_mouth(image, landmarks, mask, cutout_fill, threshold)
    elif choice == 2:
        image = remove_nose(image, landmarks, mask, cutout_fill, threshold)

    return image


def remove_eyes(image, landmarks, mask=None, cutout_fill=0, threshold=0.3):
    if landmarks is None:
        return image

    (x1, y1), (x2, y2) = landmarks[:2]
    line = cv2.line(
        np.zeros_like(image[..., 0]), (x1, y1), (x2, y2), color=(1), thickness=2
    )
    w = _distance((x1, y1), (x2, y2))

    return _remove(image, mask, w, line, cutout_fill, threshold)


def remove_mouth(image, landmarks, mask=None, cutout_fill=0, threshold=0.3):
    if landmarks is None:
        return image

    (x1, y1), (x2, y2) = landmarks[-2:]
    line = cv2.line(
        np.zeros_like(image[..., 0]), (x1, y1), (x2, y2), color=(1), thickness=2
    )
    w = _distance((x1, y1), (x2, y2))

    return _remove(image, mask, w, line, cutout_fill, threshold)


def remove_nose(image, landmarks, mask=None, cutout_fill=0, threshold=0.3):
    if landmarks is None:
        return image

    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(
        np.zeros_like(image[..., 0]), (x3, y3), (x4, y4), color=(1), thickness=2
    )
    w = _distance((x1, y1), (x2, y2))

    return _remove(image, mask, w, line, cutout_fill, threshold)


def _remove(image, mask, w, line, cutout_fill, threshold):
    image = image.copy()
    if mask is not None:
        mask_ones = np.count_nonzero(mask == 1)
    for i in range(3, 7):  # Try multiple times to get max overlap below threshold
        line_ = binary_dilation(line, iterations=int(w // i))
        if mask is not None:
            cut = np.bitwise_and(line_, mask)
            cut_ones = np.count_nonzero(cut == 1)
            if (cut_ones / mask_ones) > threshold:
                continue
        if cutout_fill == -1:
            image[line_, :] = np.random.randint(0, 255, image[line_, :].shape)
        else:
            image[line_, :] = cutout_fill
        break
    return image


def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def convex_hull_cutout(
    image,
    original=None,
    mask=None,
    probability=0.5,
    cutout_fill=0,
    threshold=0.3,
    detector=dlib_detector,
    predictor=dlib_predictor,
):
    """[Face Cutout using face outline points. Uses 3 methods
        1. Selects only a fixed number of consecutive points and creates the polygon
        2. Randomly select any number of points and cutout the enclosing polygon
        3. Divides face to 4 segments using centroid and selects the minimum overlapping section]

    Args:
        image ([numpy.ndarray]): [Input image for cutout]
        detector ([Dlib face detector]): [description]
        predictor ([Dlib landmark predictor]): [Dlib landmark predictor after loading weights]
        mask ([numpy.ndarray], optional): [Binary mask for fake face]. Defaults to None.
        probability (float, optional): [Cutout probability]. Defaults to 0.5.
        cutout_fill (int, optional): [Cutout Fill]. Defaults to 0. [-1 = Random Fill]
        threshold (float, optional): [Overlay threshold for rho]. Defaults to 0.3.

    Returns:
        [numpy.ndarray]: [Augmented Image]
    """
    if random.random() > probability:
        return image


    image_ = image.copy()

    rects = detector(original if original is not None else image)
    if len(rects) == 0:
        return image
    sp = predictor(original if original is not None else image, rects[0])

    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    outline = landmarks[[*range(17), *range(26, 16, -1)]]

    polygon = None
    choice = random.choice([0, 1, 2])
    if choice == 0:
        points = random.randint(8, 15)
        mx = -1
        poly_ = None
        for i in range(0, len(outline), 2):
            vertices = outline[i : i + points]
            if len(vertices) < points:
                break
            Y, X = draw.polygon(vertices[:, 1], vertices[:, 0])
            polygon = np.zeros(image.shape[:2], dtype=np.uint8)
            polygon[Y, X] = 1
            if mask is not None:
                cut = np.bitwise_and(polygon, mask)
                cut_ones = np.count_nonzero(cut == 1)
                mask_ones = np.count_nonzero(mask == 1)
                if (cut_ones / mask_ones) > threshold:
                    continue
                elif _poly_area(vertices[:, 0], vertices[:, 1]) > mx:
                    mx = _poly_area(vertices[:, 0], vertices[:, 1])
                    poly_ = polygon

        polygon = poly_
    elif choice == 1:
        for i in range(15, 8, -1):
            vertices = outline[np.random.randint(outline.shape[0], size=i), :]
            Y, X = draw.polygon(vertices[:, 1], vertices[:, 0])
            polygon = np.zeros(image.shape[:2], dtype=np.uint8)
            polygon[Y, X] = 1
            if mask is not None:
                cut = np.bitwise_and(polygon, mask)
                cut_ones = np.count_nonzero(cut == 1)
                mask_ones = np.count_nonzero(mask == 1)
                if (cut_ones / mask_ones) > threshold:
                    continue
            break
    elif choice == 2:
        Y, X = draw.polygon(outline[:, 1], outline[:, 0])
        polygon = np.zeros(image.shape[:2], dtype=np.uint8)
        polygon[Y, X] = 1
        polygon = _centroid_cut(polygon, mask, threshold)

    if polygon is not None:
        if cutout_fill == -1:
            image_[polygon > 0] = np.random.randint(0, 255, image_[polygon > 0].shape)
        else:
            image_[polygon > 0] = cutout_fill
    return image_


def _centroid_cut(polygon, mask, threshold=0.3):
    y, x = measure.centroid(polygon)
    y = int(y)
    x = int(x)
    parts = []
    _p = polygon.copy()
    _p[:y, :] = 0
    parts.append(_p)
    _p = polygon.copy()
    _p[y:, :] = 0
    parts.append(_p)
    _p = polygon.copy()
    _p[:, :x] = 0
    parts.append(_p)
    _p = polygon.copy()
    _p[:, x:] = 0
    parts.append(_p)

    flag = 0
    for i in range(4):
        if mask is not None:
            flag = 1
            cut = np.bitwise_and(parts[i], mask)
            cut_ones = np.count_nonzero(cut == 1)
            mask_ones = np.count_nonzero(mask == 1)
            if (cut_ones / mask_ones) > threshold:
                continue
        if flag == 0:
            return parts[random.randint(0, 3)]
        else:
            return parts[i]
    return None


def _poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
