from PIL import Image
import numpy as np

def create_dark_logo(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # Extract channels
    r, g, b, a = data.T
    
    # Identify the dark blue text color
    # Based on analysis: (40, 84, 119) is the core color.
    # Let's define a range for "dark blue/slate"
    # R: 0-100, G: 40-120, B: 80-160
    
    # Condition for text pixels (Dark Blue)
    text_mask = (r < 100) & (g < 120) & (b > 60) & (a > 0)
    
    # Change text pixels to White (255, 255, 255)
    data[..., 0][text_mask.T] = 255 # R
    data[..., 1][text_mask.T] = 255 # G
    data[..., 2][text_mask.T] = 255 # B
    
    # Save
    im2 = Image.fromarray(data)
    im2.save(output_path)
    print(f"Saved dark mode logo to {output_path}")

if __name__ == "__main__":
    create_dark_logo("logo.png", "logo_dark.png")
