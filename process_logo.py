from PIL import Image
import numpy as np

def convert_white_to_black(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # Define white threshold (e.g., > 240 for R, G, B)
    r, g, b, a = data.T
    white_areas = (r > 240) & (g > 240) & (b > 240)
    
    # Set white areas to transparent
    data[..., 3][white_areas.T] = 0
    
    im2 = Image.fromarray(data)
    im2.save(output_path)
    print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    input_path = "/Users/michaelbarry/.gemini/antigravity/brain/da4703d3-1c6f-4eaf-98dd-5bd376151fc2/uploaded_image_1764346040138.png"
    output_path = "logo.png"
    convert_white_to_black(input_path, output_path)
