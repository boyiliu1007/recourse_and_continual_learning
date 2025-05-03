from PIL import Image
import os

# png folder
folder_path = 'Pure Recourse Experiment/Results/' 

# Generate list of filenames based on naming pattern
image_files = []
i = 0
while True:
    filename = os.path.join(folder_path, f'recourse_round_{i}.png')
    if os.path.exists(filename) and i <= 10:
        image_files.append(filename)
        i += 1
    else:
        break

# Check if images were found
if not image_files:
    print("No images found with the pattern 'recourse_round_{i}.png'")
else:
    # Open images
    images = [Image.open(img) for img in image_files]

    # Save as GIF
    output_path = os.path.join(folder_path, 'output.gif')
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # duration between frames in milliseconds
        loop=1         
    )
    print(f"GIF saved as {output_path}")