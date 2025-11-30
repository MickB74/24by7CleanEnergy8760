import pandas as pd
from PIL import Image
import numpy as np
from collections import Counter

def analyze_colors(image_path):
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    
    # Reshape to list of pixels
    pixels = data.reshape(-1, 4)
    
    # Filter out transparent pixels
    visible_pixels = [tuple(p) for p in pixels if p[3] > 0]
    
    # Count colors
    counts = Counter(visible_pixels)
    
    print(f"Top 10 colors in {image_path}:")
    for color, count in counts.most_common(10):
        print(f"Color: {color}, Count: {count}")

if __name__ == "__main__":
    analyze_colors("logo.png")
