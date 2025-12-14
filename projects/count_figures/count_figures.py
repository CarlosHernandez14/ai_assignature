import cv2 as cv
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# HSV color thresholds. Each color maps to one or more (lower, upper) HSV ranges
COLOR_HSV_RANGES = {
    "blue": [
        ((90, 60, 40), (140, 255, 255)),
    ],
    "green": [
        ((35, 40, 40), (85, 255, 255)),
    ],
    "yellow": [
        ((20, 60, 40), (35, 255, 255)),
    ],
    "red": [
        ((0, 60, 40), (10, 255, 255)),
        ((170, 60, 40), (180, 255, 255)),
    ],
}


@dataclass
class Component:
    """Represents a connected component in the binary image."""
    area: int
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


def build_color_mask(hsv_img: np.ndarray, color_name: str) -> np.ndarray:
    """
    Create a binary mask (0/255) by thresholding hsv_img for the specified color.
    Supports colors that require multiple ranges (e.g., red).
    """
    color_key = color_name.lower()
    ranges = COLOR_HSV_RANGES.get(color_key)
    if not ranges:
        raise ValueError(f"Unsupported color '{color_name}'. Available: {list(COLOR_HSV_RANGES.keys())}")

    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        lower = np.array(low, dtype=np.uint8)
        upper = np.array(high, dtype=np.uint8)
        part = cv.inRange(hsv_img, lower, upper)  # 255 where within range, else 0
        mask = cv.bitwise_or(mask, part)

    return mask


def clean_mask(mask_255: np.ndarray, kernel_size: Tuple[int, int] = (7, 7),
               open_iters: int = 1, close_iters: int = 1) -> np.ndarray:
    """
    Apply morphological open and close to remove small noise and close tiny gaps.
    Returns a cleaned mask (0/255).
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    cleaned = mask_255.copy()
    if open_iters > 0:
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel, iterations=open_iters)
    if close_iters > 0:
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=close_iters)
    return cleaned


def binarize_by_color(bgr_img: np.ndarray, color_name: str) -> np.ndarray:
    """
    Convert BGR image to HSV, threshold by color, clean mask, and return a binary grid (0/1).
    Foreground (selected color) pixels are 1; background pixels are 0.
    """
    hsv = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    raw_mask = build_color_mask(hsv, color_name)
    cleaned_mask = clean_mask(raw_mask)
    binary = (cleaned_mask > 0).astype(np.uint8)  # 0/1
    return binary


def count_components_bfs(binary_grid: np.ndarray, connectivity: int = 4) -> Tuple[int, List[Component]]:
    """
    Count connected components on a binary grid using BFS flood-fill.

    - binary_grid: np.ndarray of shape (H, W), with values {0,1}
    - connectivity: 4 or 8 neighbors for adjacency

    Returns:
      (num_components, components_list)
    """
    h, w = binary_grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: List[Component] = []

    if connectivity == 8:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(h):
        for col in range(w):
            # Start a BFS when we find an unvisited foreground pixel
            if binary_grid[row, col] == 1 and not visited[row, col]:
                queue = deque()
                queue.append((row, col))
                visited[row, col] = True

                area = 0
                x_min = x_max = col
                y_min = y_max = row

                while queue:
                    cy, cx = queue.popleft()
                    area += 1

                    # Update bounding box
                    if cx < x_min: x_min = cx
                    if cx > x_max: x_max = cx
                    if cy < y_min: y_min = cy
                    if cy > y_max: y_max = cy

                    # Explore neighbors
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if not visited[ny, nx] and binary_grid[ny, nx] == 1:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                components.append(Component(area=area, bbox=(x_min, y_min, x_max, y_max)))

    return len(components), components


def main():
    image_path = "/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/count_figures/img_figures.jpg" 
    selected_color = "green"  
    connectivity = 4   # BFS connectivity: 4 or 8

    # Load input image
    bgr = cv.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image at: {image_path}")

    # Color binarization
    binary = binarize_by_color(bgr, selected_color)

    # APPLY BFS
    num, comps = count_components_bfs(binary, connectivity=connectivity)

    # Report results
    print(f"Selected color: {selected_color}")
    print(f"Detected figures (components): {num}")
    for i, c in enumerate(comps, 1):
        x_min, y_min, x_max, y_max = c.bbox
        print(f"  - Component {i}: area={c.area}, bbox=(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})")

    # Optional: simple visualization of the mask if matplotlib is available
    try:
        plt.figure(figsize=(6, 6))
        plt.imshow((binary * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        plt.title(f"Binarized mask for color '{selected_color}'")
        plt.axis("off")
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
