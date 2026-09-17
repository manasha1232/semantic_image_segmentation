import os
import cv2
import numpy as np

def generate_synthetic_portrait(output_path="input/sample_portrait.jpg", width=800, height=600):
    """Generates a synthetic portrait image with subject on textured background."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Background: Complex outdoor scene (Trees & Sky)
    bg = np.ones((height, width, 3), dtype=np.uint8) * 60
    bg[0:int(height*0.4), :] = (220, 180, 100) # Sky
    bg[int(height*0.4):, :] = (40, 120, 40)   # Green foliage
    
    # Draw Subject Silhouette (Person) in center foreground
    cx, cy = width // 2, height // 2 + 50
    head_r = 65
    
    # Head
    cv2.circle(bg, (cx, cy - 130), head_r, (180, 210, 240), -1)
    cv2.circle(bg, (cx, cy - 130), head_r, (100, 120, 150), 2)
    # Hair
    cv2.ellipse(bg, (cx, cy - 150), ( head_r + 5, 45), 0, 180, 360, (30, 30, 30), -1)
    # Eyes & Smile
    cv2.circle(bg, (cx - 22, cy - 140), 7, (30, 30, 30), -1)
    cv2.circle(bg, (cx + 22, cy - 140), 7, (30, 30, 30), -1)
    cv2.ellipse(bg, (cx, cy - 110), (22, 14), 0, 0, 180, (30, 30, 30), 3)
    
    # Shoulders & Body
    body_pts = np.array([
        [cx - 150, height],
        [cx - 100, cy - 50],
        [cx + 100, cy - 50],
        [cx + 150, height]
    ], dtype=np.int32)
    cv2.fillPoly(bg, [body_pts], (220, 70, 50)) # Red jacket
    
    # Title Tag
    cv2.putText(bg, "SEMANTIC SEGMENTATION PORTRAIT TEST BENCH", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
                
    cv2.imwrite(output_path, bg)
    print(f"[OK] Synthetic portrait saved to '{output_path}'")
    return output_path

def generate_all_demo_scenes():
    """Generates synthetic portrait and scene images."""
    generate_synthetic_portrait("input/sample_portrait.jpg")

if __name__ == "__main__":
    generate_all_demo_scenes()
