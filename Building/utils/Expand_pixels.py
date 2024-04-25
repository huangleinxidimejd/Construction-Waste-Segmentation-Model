from PIL import Image
import os


def expand_image(image_path, target_width, target_height):
    with Image.open(image_path) as img:
        # Get the current size of the image
        current_width, current_height = img.size

        # Calculate the ratio to scale the image to the target size
        width_ratio = target_width / current_width
        height_ratio = target_height / current_height

        # Use the minimum ratio to maintain the original aspect ratio of the image
        scale_ratio = min(width_ratio, height_ratio)

        # Calculate the new size of the image after expansion
        new_width = int(current_width * scale_ratio)
        new_height = int(current_height * scale_ratio)

        # Resize the image using the calculated size
        expanded_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Create a new blank image with the target size
        target_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))

        # Calculate the position to paste the expanded image in the center of the blank image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Paste the expanded image onto the blank image
        target_img.paste(expanded_img, (paste_x, paste_y))

        return target_img


# Path to the directory containing the images
image_directory = 'C:/Users/dell/Desktop/新建文件夹 (2)'

# Path to the directory where expanded images will be saved
output_directory = 'C:/Users/dell/Desktop/新建文件夹 (2)/1'

# Target size for the expanded images
target_width = 500
target_height = 500

# Get a list of all image files in the directory
image_files = [file for file in os.listdir(image_directory) if file.endswith('.jpg') or file.endswith('.png')]

# Loop through each image and expand it to the target size
for image in image_files:
    image_path = os.path.join(image_directory, image)
    expanded_image = expand_image(image_path, target_width, target_height)
    output_path = os.path.join(output_directory, image)
    expanded_image.save(output_path)
