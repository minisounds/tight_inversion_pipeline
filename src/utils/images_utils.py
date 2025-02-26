from PIL import Image, ImageDraw, ImageFont
import os
import torch
import glob
import matplotlib.pyplot as plt

def read_images_in_path(path, size = (512,512)):
    image_paths = []
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(path, filename)
            image_paths.append(image_path)
    image_paths = sorted(image_paths)
    return [Image.open(image_path).convert("RGB").resize(size) for image_path in image_paths]

def concatenate_images(image_lists, return_list = False):
    num_rows = len(image_lists[0])
    num_columns = len(image_lists)
    image_width = image_lists[0][0].width
    image_height = image_lists[0][0].height

    grid_width = num_columns * image_width
    grid_height = num_rows * image_height if not return_list else image_height
    if not return_list:
        grid_image = [Image.new('RGB', (grid_width, grid_height))]
    else:
        grid_image = [Image.new('RGB', (grid_width, grid_height)) for i in range(num_rows)]

    for i in range(num_rows):
        row_index = i if return_list else 0
        for j in range(num_columns):
            image = image_lists[j][i]
            x_offset = j * image_width
            y_offset = i * image_height if not return_list else 0
            grid_image[row_index].paste(image, (x_offset, y_offset))

    return grid_image if return_list else grid_image[0]

def concatenate_images_single(image_lists):
    num_columns = len(image_lists)
    image_width = image_lists[0].width
    image_height = image_lists[0].height

    grid_width = num_columns * image_width
    grid_height = image_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for j in range(num_columns):
        image = image_lists[j]
        x_offset = j * image_width
        y_offset = 0
        grid_image.paste(image, (x_offset, y_offset))

    return grid_image

def get_captions_for_images(images, device):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    )  # doctest: +IGNORE_RESULT

    res = []
    
    for image in images:
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        res.append(generated_text)

    del processor
    del model
    
    return res

def find_and_plot_images(directory, output_file, recursive=True, figsize=(15, 15), image_formats=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")):
    """
    Finds all images in the specified directory (optionally recursively) 
    and saves them in a single figure with their filenames.

    Parameters:
        directory (str): Path to the directory.
        output_file (str): Path to save the resulting figure (e.g., 'output.png').
        recursive (bool): Whether to search directories recursively.
        figsize (tuple): Size of the resulting figure.
        image_formats (tuple): Image file formats to look for.

    Returns:
        None
    """
    # Gather all image file paths
    pattern = "**/" if recursive else ""
    images = []
    for fmt in image_formats:
        images.extend(glob.glob(os.path.join(directory, pattern + fmt), recursive=recursive))

    images = [image for image in images if "noise.jpg" not in image and "result" not in image]  # Filter out noise and result images
    # move "original" to the front, followed by "reconstruction" and then the rest
    images = sorted(
        images,
        key=lambda x: (not x.endswith("original.jpg"), not x.endswith("reconstruction.jpg"), x)
    )
    
    if not images:
        print("No images found!")
        return

    # Create a figure
    num_images = len(images)
    cols = num_images  # Max 5 images per row
    rows = (num_images + cols - 1) // cols  # Calculate number of rows
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten() if num_images > 1 else [axs]  # Flatten axes for single image case

    for i, image_path in enumerate(images):
        # Open and plot image
        img = Image.open(image_path)
        axs[i].imshow(img)
        axs[i].axis('off')  # Remove axes
        axs[i].set_title(os.path.basename(image_path), fontsize=8)  # Add filename

    # Hide any remaining empty axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=200)  # Save the figure to the file
    plt.close(fig)  # Close the figure to free up memory
    print(f"Figure saved to {output_file}")


def add_label_to_image(image, label):
    """
    Adds a label to the lower-right corner of an image.

    Args:
        image (PIL.Image): Image to add the label to.
        label (str): Text to add as a label.

    Returns:
        PIL.Image: Image with the added label.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)


    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define font and size
    font_size = int(min(image.size) * 0.05)  # Adjust font size based on image dimensions
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Replace with a font path if needed
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not found

    # Measure text size using textbbox
    text_bbox = draw.textbbox((0, 0), label, font=font)  # (left, top, right, bottom)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position the text in the lower-right corner with some padding
    padding = 10
    position = (image.width - text_width - padding, image.height - text_height - padding)

    # Add a semi-transparent background for the label
    draw.rectangle(
        [
            (position[0] - padding, position[1] - padding),
            (position[0] + text_width + padding, position[1] + text_height + padding)
        ],
        fill=(0, 0, 0, 150)  # Black with transparency
    )

    # Draw the label
    draw.text(position, label, fill="white", font=font)

    return image

def crop_center_square_and_resize(img, size, output_path=None):
    """
    Crops the center of an image to make it square.
    
    Args:
        img (PIL.Image): Image to crop.
        output_path (str, optional): Path to save the cropped image. If None, the cropped image is not saved.
    
    Returns:
        Image: The cropped square image.
    """
    width, height = img.size
    # Determine the shorter side
    side_length = min(width, height)
    # Calculate the cropping box
    left = (width - side_length) // 2
    top = (height - side_length) // 2
    right = left + side_length
    bottom = top + side_length
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    # Resize the image
    cropped_img = cropped_img.resize(size)
    
    # Save the cropped image if output path is specified
    if output_path:
        cropped_img.save(output_path)
    
    return cropped_img
