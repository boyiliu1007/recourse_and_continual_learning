import os
import matplotlib.pyplot as plt

# Get the directory of the current script file
current_file_path = __file__
current_directory = os.path.dirname(current_file_path)

# Get the current file name without the extension
current_file_name = os.path.basename(current_file_path)
current_file_name_without_extension = os.path.splitext(current_file_name)[0]

# Print the current file name without the extension
print(f"Current file name without extension: {current_file_name_without_extension}")

# # Specify the path for the new folder
# DIRECTORY = os.path.join(os.path.dirname(__file__), "output")

# # Create the folder if it does not exist
# try:
#     os.makedirs(DIRECTORY, exist_ok=True)
#     print(f"Folder '{DIRECTORY}' is ready.")
# except Exception as e:
#     print(f"An error occurred: {e}")

# plt.figure()
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# plt.title('Sample Plot')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# # Save the plot into the folder
# try:
#     plt.savefig(os.path.join(DIRECTORY, "sample_plot.png"))
# except Exception as e:
#     print(f"An error occurred while saving the plot: {e}")

# pass in directory and 
