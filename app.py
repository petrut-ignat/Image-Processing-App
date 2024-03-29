# importing the necessary packages
import PySimpleGUI as sg
from PIL import Image, ImageFilter
from PIL.Image import fromarray
from io import BytesIO
import numpy as np
import time
import math
import sys
from numpy import mean, zeros_like, array


def grayscale(image):

    image = array(image.convert('RGB'))

    for i in range(0, image.shape[0], 1):
        for j in range(0, image.shape[1], 1):
            image[i][j][:] = int(0.3 * image[i][j][0] + 0.59 * image[i][j][1] + 0.11 * image[i][j][2])
            
    return Image.fromarray(image.astype('uint8'))

def detect_edges(image):

    image = array(image.convert('RGB'))
    height, width = image.shape[:2]

    new_image = zeros_like(image)
    new_image[:][:] = [255, 255, 255]

    for i in range(1, width-1):
        for j in range(1, height-1):
            old_pix = image[j][i]
            left_pix = image[j][i-1]
            bottom_pix = image[j+1][i]
            right_pix = image[j][i+1]
            top_pix = image[j-1][i]

            if abs(mean(old_pix[:]) - mean(left_pix[:])) > 10 or \
                abs(mean(old_pix[:]) - mean(bottom_pix[:])) > 10 or \
                abs(mean(old_pix[:]) - mean(right_pix[:])) > 10 or \
                abs(mean(old_pix[:]) - mean(top_pix[:])) > 10:
                new_image[j][i] = [0, 0, 0]

    return fromarray(new_image.astype('uint8'))



def invert_colors(image):
# de prezentat diferenta de timp intre metodele de mai jos

    # width, height = image.size

    # inverted_image = Image.new('RGB', (width, height))

    # for y in range(height):
    #     for x in range(width):
    #         pixel = image.getpixel((x, y))
    #         inverted_pixel = tuple(255 - color for color in pixel)
    #         inverted_image.putpixel((x, y), inverted_pixel)

    # return inverted_image

    image = np.array(image.convert('RGB'))

    inverted_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            inverted_image[i][j][:] = 255 - image[i][j][:]

    return Image.fromarray(inverted_image.astype('uint8'))


def mirror_image(image):

    # width, height = image.size

    # mirrored_image = Image.new('RGB', (width, height))

    # for y in range(height):
    #     for x in range(width):
    #         pixel = image.getpixel((width - x - 1, y))
    #         mirrored_image.putpixel((x, y), pixel)

    # return mirrored_image

    image = np.array(image.convert('RGB'))

    mirror_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            mirror_image[i][j][:] = image[i][image.shape[1] - j - 1][:]

    return Image.fromarray(mirror_image.astype('uint8'))


def flip_image_vertically(image):

    # width, height = image.size

    # flipped_image = Image.new('RGB', (width, height))

    # for y in range(height):
    #     for x in range(width):
    #         pixel = image.getpixel((x, height - y - 1))
    #         flipped_image.putpixel((x, y), pixel)

    # return flipped_image
    
    image = np.array(image.convert('RGB'))

    flipped_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flipped_image[i][j][:] = image[image.shape[0] - i - 1][j][:]

    return Image.fromarray(flipped_image.astype('uint8'))


def sharpen_image(image, amount):

    sharpened_image = image.filter(ImageFilter.Kernel((3,3), (0,-1,0,-1,5,-1,0,-1,0), 1, 0))
    sharpened_image = Image.blend(image, sharpened_image, amount)
    
    return sharpened_image.convert('RGB')


def gaussian_kernel_1d(sigma):
    # Compute the 1D Gaussian kernel of size 6*sigma+1
    size = int(6*sigma + 1)
    kernel = np.zeros(size)
    for i in range(size):
        x = i - size//2
        kernel[i] = 1/(math.sqrt(2*math.pi)*sigma) * \
            math.exp(-(x**2)/(2*sigma**2))
    return kernel


def gaussian_blur(image, radius):
    # Compute the 1D Gaussian kernel of size 6*sigma+1
    radius += 2
    sigma = radius / 3.0
    kernel_1d = gaussian_kernel_1d(sigma)
    radius = int(radius)

    # Convert the image to a NumPy array and apply the horizontal blur
    pixels = np.array(image.convert('RGB'))
    blurred_pixels = np.zeros_like(pixels)
    for i in range(pixels.shape[0]):
        for j in range(radius, pixels.shape[1] - radius):
            for c in range(3):
                val = 0.0
                for ki in range(-radius, radius + 1):
                    pixel = pixels[i, j + ki][c]
                    weight = kernel_1d[ki + radius]
                    val += pixel * weight
                blurred_pixels[i, j, c] = val

    # Apply the vertical blur to the horizontally blurred image
    for i in range(radius, pixels.shape[0] - radius):
        for j in range(pixels.shape[1]):
            for c in range(3):
                val = 0.0
                for ki in range(-radius, radius + 1):
                    pixel = blurred_pixels[i + ki, j][c]
                    weight = kernel_1d[ki + radius]
                    val += pixel * weight
                blurred_pixels[i, j, c] = val

    blurred_image = Image.fromarray(blurred_pixels.astype('uint8'))
    return blurred_image

def emboss_filter(image):
    width, height = image.size
    emboss_image = Image.new("RGB", (width, height), (0, 0, 0))
    
    kernel = [
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ]
    
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            r_total = 0
            g_total = 0
            b_total = 0
            
            for i in range(3):
                for j in range(3):
                    pixel = image.getpixel((x+i-1, y+j-1))
                    r_total += pixel[0] * kernel[i][j]
                    g_total += pixel[1] * kernel[i][j]
                    b_total += pixel[2] * kernel[i][j]
            
            r_total += 128
            g_total += 128
            b_total += 128
            
            r_total = max(0, min(255, r_total))
            g_total = max(0, min(255, g_total))
            b_total = max(0, min(255, b_total))
            
            emboss_image.putpixel((x, y), (r_total, g_total, b_total))
    
    return emboss_image


def adjust_contrast(image, level):
    # Convert image to numpy array
    np_image = np.array(image)

    level = level * 2

    # Compute minimum and maximum pixel values
    min_val = np.min(np_image)
    max_val = np.max(np_image)

    # Compute new minimum and maximum pixel values based on contrast level
    contrast_min = (level / 100) * (max_val - min_val) + min_val
    contrast_max = ((100 - level) / 100) * (max_val - min_val) + min_val

    # Perform contrast stretching
    np_image = (np_image - min_val) * ((contrast_max -
                                        contrast_min) / (max_val - min_val)) + contrast_min

    # Clip pixel values to valid range
    np_image = np.clip(np_image, 0, 255)

    # Convert back to Pillow image
    adjusted_image = Image.fromarray(np_image.astype('uint8'))

    return adjusted_image


def apply_sepia(image, sepia_intensity):
    # Convert image to RGBA mode to support transparency
    image = image.convert('RGBA')

    # Define sepia color
    sepia_color = (112, 66, 20)

    sepia_intensity = sepia_intensity / 10

    # Create a new blank image of the same size and mode as the original image
    sepia_image = Image.new(image.mode, image.size)

    # Apply sepia effect to each pixel in the image
    for x in range(image.width):
        for y in range(image.height):
            # Get pixel color at (x,y)
            pixel = image.getpixel((x, y))

            # Convert pixel color to sepia
            sepia_red = int(pixel[0] * (1 - sepia_intensity) +
                            sepia_color[0] * sepia_intensity)
            sepia_green = int(
                pixel[1] * (1 - sepia_intensity) + sepia_color[1] * sepia_intensity)
            sepia_blue = int(
                pixel[2] * (1 - sepia_intensity) + sepia_color[2] * sepia_intensity)

            # Set pixel color in sepia image
            sepia_image.putpixel(
                (x, y), (sepia_red, sepia_green, sepia_blue, pixel[3]))

    # Convert sepia image back to original mode and return
    return sepia_image.convert(image.mode)


def emphasize_edges(image):
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Create a new image with the same dimensions as the input image
    result_image = Image.new('RGB', image.size)

    # Apply the contour filter to the image
    contour_pixels = []
    width, height = gray_image.size
    for y in range(height):
        for x in range(width):
            if x > 0 and y > 0 and x < width-1 and y < height-1:
                # Calculate the gradient of the surrounding pixels
                gx = (-1 * gray_image.getpixel((x-1, y-1))) + (-2 * gray_image.getpixel((x-1, y))) + (-1 * gray_image.getpixel((x-1, y+1))) + \
                     (1 * gray_image.getpixel((x+1, y-1))) + (2 *
                                                              gray_image.getpixel((x+1, y))) + (1 * gray_image.getpixel((x+1, y+1)))
                gy = (-1 * gray_image.getpixel((x-1, y-1))) + (-2 * gray_image.getpixel((x, y-1))) + (-1 * gray_image.getpixel((x+1, y-1))) + \
                     (1 * gray_image.getpixel((x-1, y+1))) + (2 *
                                                              gray_image.getpixel((x, y+1))) + (1 * gray_image.getpixel((x+1, y+1)))
                gradient = int((gx**2 + gy**2)**0.5)
                if gradient > 50:
                    # Set the pixel in the result image to white
                    contour_pixels.append((255, 255, 255))
                else:
                    # Set the pixel in the result image to black
                    contour_pixels.append((0, 0, 0))
            else:
                # Set the pixel in the result image to black if it's on the edge
                contour_pixels.append((0, 0, 0))

    # Put the contour pixels in the result image
    result_image.putdata(contour_pixels)

    # Enhance the edges of the image
    edge_pixels = []
    width, height = result_image.size
    for y in range(height):
        for x in range(width):
            if x > 0 and y > 0 and x < width-1 and y < height-1:
                # Calculate the average value of the surrounding pixels
                surrounding_pixels = [
                    result_image.getpixel((x-1, y-1)),
                    result_image.getpixel((x-1, y)),
                    result_image.getpixel((x-1, y+1)),
                    result_image.getpixel((x, y-1)),
                    result_image.getpixel((x, y+1)),
                    result_image.getpixel((x+1, y-1)),
                    result_image.getpixel((x+1, y)),
                    result_image.getpixel((x+1, y+1))
                ]
                average_value = tuple(int(sum(color) / 8)
                                      for color in zip(*surrounding_pixels))
                edge_pixels.append(tuple(2*val for val in average_value))
            else:
                edge_pixels.append((0, 0, 0))

    # Put the edge pixels in the result image
    result_image.putdata(edge_pixels)

    return result_image

def threshold(image, threshold_value):
    # Get the width and height of the image
    width, height = image.size

    # Define the linear function that maps the input threshold value to a range of thresholds
    # The range of thresholds increases linearly with the input value, from 0 to 255
    threshold_range = int(threshold_value * 25.5)
    threshold_min = max(0, threshold_range - 25)
    threshold_max = min(255, threshold_range + 25)

    # Create a new image with the same dimensions as the input image
    threshold_image = Image.new('RGB', (width, height))

    # Loop over the pixels of the input image and convert each pixel to black or white based on the threshold value
    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            gray = sum(pixel) // 3
            if gray >= threshold_min and gray <= threshold_max:
                threshold_pixel = (255, 255, 255)
            else:
                threshold_pixel = (0, 0, 0)
            threshold_image.putpixel((x, y), threshold_pixel)

    return threshold_image





def update_image(original, blur, contrast, emboss, contour, flipx, flipy, grayScale, sepia, sharp, invert):
    global image

    image = original

    if blur != 0:
        t_start = time.perf_counter()
        image = gaussian_blur(image, blur)
        t_end = time.perf_counter()
        print(f'gaussian_blur: {t_end - t_start}')

    if contrast != 0:
        image = adjust_contrast(image, contrast)

    if sepia != 0:
        image = apply_sepia(image, sepia)

    if sharp != 0:
        # image = threshold(image, sharp)
        t_start = time.perf_counter()
        image = sharpen_image(image, sharp)
        t_end = time.perf_counter()
        print(f'sharpen_image: {t_end - t_start}')

    if grayScale:
        t_start = time.perf_counter()
        image = grayscale(image)
        t_end = time.perf_counter()
        print(f'grayscale: {t_end - t_start}')

    if emboss:
        t_start = time.perf_counter()
        # image = image.filter(ImageFilter.EMBOSS())
        image = emboss_filter(image)
        t_end = time.perf_counter()
        print(f'EMBOSS: {t_end - t_start}')

    if contour:
        # image = image.filter(ImageFilter.CONTOUR())
        t_start = time.perf_counter()
        # image = emphasize_edges(image)
        image = detect_edges(image)
        t_end = time.perf_counter()
        print(f'emphasize_edges: {t_end - t_start}')

    if flipx:
        # image = ImageOps.mirror(image)
        image = mirror_image(image)

    if flipy:
        t_start = time.perf_counter()
        # image = ImageOps.flip(image)
        image = flip_image_vertically(image)
        t_end = time.perf_counter()
        print(f'flip_image_vertically: {t_end - t_start}')

    if invert:
        t_start = time.perf_counter()
        image = invert_colors(image)
        t_end = time.perf_counter()
        print(f'invert_colors: {t_end - t_start}')

    bio = BytesIO()
    image.save(bio, format='PNG')

    window['-IMAGE-'].update(data=bio.getvalue())


image_path = sg.popup_get_file('Open', no_window=True)

# page layout
control_col = sg.Column([
    [sg.Frame('Blur', layout=[
              [sg.Slider(range=(0, 5), orientation='h', key='-BLUR-')]])],
    [sg.Frame('Sepia', layout=[
              [sg.Slider(range=(0, 10), orientation='h', key='-SEPIA-')]])],
    [sg.Frame('Contrast', layout=[
              [sg.Slider(range=(0, 10), orientation='h', key='-CONTRAST-')]])],
    [sg.Frame('Sharpen', layout=[
              [sg.Slider(range=(0, 10), orientation='h', key='-SHARPEN-')]])],
    [sg.Checkbox('Emboss', key='-EMBOSS-'),
     sg.Checkbox('Contour', key='-CONTOUR-')],
    [sg.Checkbox('Grayscale', key='-GRAYSCALE-'),
     sg.Checkbox('Invert colors', key='-INVERT-')],
    [sg.Checkbox('Flip x', key='-FLIPX-'),
     sg.Checkbox('Flip y', key='-FLIPY-')],
    [sg.Button('Save image', key='-SAVE-'),
     sg.Button('Exit', key='-EXIT-')]
])
image_col = sg.Column([
    [sg.Image(image_path, key='-IMAGE-')]
])

layout = [[control_col, image_col]]

original = Image.open(image_path)

# image resizing
width, height = original.size
aspect_ratio = width / height
target_ratio = 530 / 420

if aspect_ratio > target_ratio:
    new_width = 530
    new_height = int(new_width / aspect_ratio)
else:
    new_height = 420
    new_width = int(new_height * aspect_ratio)

original = original.resize((new_width, new_height))

# Create the window
window = sg.Window('Image Editor', layout, size=(800, 440))

while (True):
    event, values = window.read(timeout=50)
    if event == sg.WIN_CLOSED:
        break

    update_image(
        original,
        values['-BLUR-'],
        values['-CONTRAST-'],
        values['-EMBOSS-'],
        values['-CONTOUR-'],
        values['-FLIPX-'],
        values['-FLIPY-'],
        values['-GRAYSCALE-'],
        values['-SEPIA-'],
        values['-SHARPEN-'],
        values['-INVERT-']
    )

    if event == '-SAVE-':
        save_path = sg.popup_get_file(
            'Save', save_as=True, no_window=True) + '.png'
        image.save(save_path, 'PNG')

    if event == '-EXIT-':
        window.close()

window.close()
