import glob
import os
import re
from PIL import Image
from IPython.display import display # To display the image in the notebook
import torch # Assuming torch is used elsewhere, good to have imports

# --- Helper Function (using the regex for the '_seed' format) ---
def get_epoch_from_filename(filename):
    """
    Extracts the epoch number from filenames like
    'samples/mnist_run_0_epoch_0010_seed42.png'.
    """
    # Regex expects _epoch_ followed by digits, then _seed followed by digits, then .png
    match = re.search(r'_epoch_(\d+)_seed\d+\.png$', filename) # Use the seed-aware regex

    if match:
        return int(match.group(1))
    print(f"Warning: Could not parse epoch number from {filename} using pattern '_epoch_(\\d+)_seed\\d+\\.png$'.")
    return -1

# --- Updated Main Function for GIF Creation ---
def create_gif_from_images_notebook(
    directory,            # Directory containing all samples
    run_identifier,       # String to identify the run (e.g., "run_0")
    output_filename,      # Path for the output GIF
    duration_ms=200,      # Duration per frame
    loop=0                # Loop count (0=infinite)
    ):
    """
    Finds images for a specific run in a directory, sorts them by epoch,
    creates an animated GIF, and optionally displays it.

    Args:
        directory (str): Path to the directory containing sample images.
        run_identifier (str): Identifier string for the specific run
                              (e.g., "run_0", "run_1").
        output_filename (str): Path to save the output GIF file.
        duration_ms (int): Duration (in milliseconds) for each frame. Default is 200.
        loop (int): Number of times the GIF should loop. 0 means loop indefinitely. Default is 0.

    Returns:
        str: The path to the saved GIF file if successful, otherwise None.
    """
    # Construct the glob pattern to find files for the specific run
    # Assumes format like "prefix_RUN-IDENTIFIER_epoch_NUM_seedNUM.png"
    image_pattern = os.path.join(directory, f"mnist_{run_identifier}_epoch_*.png") # Adjust prefix if needed

    print(f"Searching for images matching: {image_pattern}")
    filenames = glob.glob(image_pattern)

    if not filenames:
        print(f"Error: No images found matching the pattern for run '{run_identifier}': {image_pattern}")
        return None

    # Sort filenames based on the epoch number using the appropriate regex
    sorted_filenames = sorted(filenames, key=lambda f: get_epoch_from_filename(f))
    valid_filenames = [f for f in sorted_filenames if get_epoch_from_filename(f) >= 0] # Filter errors

    if not valid_filenames:
        print(f"Error: Found files for run '{run_identifier}', but couldn't parse epoch numbers correctly.")
        return None

    print(f"Found {len(valid_filenames)} valid images for run '{run_identifier}'.")
    if len(valid_filenames) > 0:
         print(f"Epoch range: {get_epoch_from_filename(valid_filenames[0])} to {get_epoch_from_filename(valid_filenames[-1])}")


    # Open images using Pillow
    images = []
    for filename in valid_filenames:
        try:
            img = Image.open(filename)
            images.append(img)
        except Exception as e:
            print(f"Warning: Skipping file {filename} due to error opening it: {e}")
            continue

    if not images:
         print("Error: No images could be successfully opened for this run.")
         return None

    # Save as an animated GIF
    try:
        print(f"Saving animated GIF to: {output_filename}")
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=loop
        )
        print("GIF created successfully!")
        return output_filename
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return None

# --- Example Usage in a Jupyter Notebook Cell ---

# Define parameters for the specific run you want to process
sample_directory = "/nvme/notebooks/samples" # Directory containing all runs
run_id_to_process = "run_2" # Specify which run
output_gif_path = f"mnist_{run_id_to_process}_progress.gif" # Output filename based on run
frame_duration = 300

# Call the function
saved_gif_path = create_gif_from_images_notebook(
    directory=sample_directory,
    run_identifier=run_id_to_process,
    output_filename=output_gif_path,
    duration_ms=frame_duration
)

# Optionally display the created GIF
if saved_gif_path:
    print(f"\nDisplaying created GIF: {saved_gif_path}")
    display(Image.open(saved_gif_path))