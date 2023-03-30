import logging
import os
import tempfile
from uuid import uuid4
import cv2
import numpy as np
import io

#import pypdf
#from pypdf.errors import PdfReadError

from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import pikepdf

from pytesseract import image_to_string, TesseractError, image_to_osd, Output

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import langcodes

import imutils
#from alyn import deskew, skew_detect

from PIL import Image
import base64
import csv
import imghdr

# large images within PDFs cause a decompression bomb error (a form of protection from abuse)
# this setting allows the user to configure how large an image they are comfortable processing
# The tradeoff to a large max size here is memory consumption, which the user can self-regulate
# using this setting.  If they do not set the variable, it remains at the default for PIL.

MAX_IMAGE_PIXELS = os.environ.get('MAX_IMAGE_PIXELS', 178956970)  # this is the default PIL max size
Image.MAX_IMAGE_PIXELS = int(MAX_IMAGE_PIXELS)


class FileOcrResult:
    def __init__(self, pages):
        self.pages = pages


def __get_pdf_file_reader(file):
    try:
        return pikepdf.Pdf(file, strict=False)
    except pikepdf.PdfError as e:
        logging.warning(f"Error reading PDF: {str(e)}")
        return None
    except ValueError as e:
        logging.warning(f"Error reading PDF - {e.args}")
        return None


def pdf_page_as_image(filename, page_num=0, is_preview=False):
    if not is_preview:
        # high dpi and grayscale for the best OCR result
        pages = convert_from_path(filename,
                                  dpi=600,
                                  grayscale=True,
                                  first_page=page_num+1,
                                  last_page=page_num+1)
    else:
        pages = convert_from_path(filename,
                            first_page=page_num+1,
                            last_page=page_num+1)
    temp_file = f"{tempfile.gettempdir()}/{uuid4()}.png"
    pages[0].save(temp_file, format="PNG")
    # pages[page_num].save(temp_file, format="PNG")
    return temp_file


def pdf_page_as_image_obj(bytes, page_num=0, is_preview=False):
    if not is_preview:
        # high dpi and grayscale for the best OCR result
        pages = convert_from_bytes(bytes,
                                  dpi=600,
                                  grayscale=True,
                                  first_page=page_num+1,
                                  last_page=page_num+1)
    else:
        pages = convert_from_bytes(bytes,
                            first_page=page_num+1,
                            last_page=page_num+1)
        
    # Create a BytesIO object
    buffer = io.BytesIO()

    pages[0].save(buffer, format="PNG")

    buffer.seek(0)   

    # Read the bytes from the BytesIO object
    img_bytes = buffer.getvalue()

    return img_bytes



def _scale_image_object(image_object, max_resolution=2048):
    # Convert the image bytes to a NumPy array
    img_array = np.frombuffer(image_object, dtype=np.uint8)

    # Decode the image array
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    
    # Get the original width and height
    height, width = img.shape[:2]
    
    # Check if the image is already smaller than the target resolution
    if height <= max_resolution and width <= max_resolution:
        return image_object
    
    # Calculate the scale factor to fit within the maximum resolution
    scale_factor = min(max_resolution / width, max_resolution / height)
    
    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Encode the resized image as a byte array
    success, encoded_image = cv2.imencode('.png', img)
    # Save the resized image
    temp_file = f"{tempfile.gettempdir()}/{uuid4()}.png"
    #cv2.imwrite(temp_file, encoded_image)

    if not success:
        raise Exception('Failed to encode image')

    # Create a PIL Image object from the encoded image
    pil_image = Image.fromarray(encoded_image)

    # Save the resized image to a byte buffer using PIL
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    # Save the resized image
    temp_file = f"{tempfile.gettempdir()}/{uuid4()}.png"
    pil_image.save(temp_file, format="PNG")

    buffer.seek(0)   

    # Read the bytes from the BytesIO object
    #img_bytes = buffer.getvalue()

    return buffer
    

def read_text_note(filename):
    with open(filename, "r") as myfile:
        text = myfile.read()
        print(text)
    return text


def read_csv(filename):
    return csv.DictReader(open(filename))


def encode_image_base64(filename, datatype):
    encoded = base64.b64encode(open(filename, "rb").read())
    img = f"data:{datatype};base64,{encoded.decode()}"
    return img


def _scale_image(image_path, max_resolution=2048):
    # Load the image
    img = cv2.imread(image_path)
    
    # Get the original width and height
    height, width = img.shape[:2]
    
    # Check if the image is already smaller than the target resolution
    if height <= max_resolution and width <= max_resolution:
        return image_path
    
    # Calculate the scale factor to fit within the maximum resolution
    scale_factor = min(max_resolution / width, max_resolution / height)
    
    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Save the resized image
    temp_file = f"{tempfile.gettempdir()}/{uuid4()}.png"
    cv2.imwrite(temp_file, img)
    return temp_file


def __get_image(filename):
    try:
        return Image.open(filename)
    except OSError:
        print(f"Error reading image: {filename}")
        return None

#rotate the image with given theta value
def __rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)
    
    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    return rotated


def __slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta


def _rotate_image_obj(image_obj):
    if isinstance(image_obj, io.BytesIO):
        image_obj = image_obj.getvalue()
    elif isinstance(image_obj, str):
        if os.path.isfile(image_obj) and imghdr.what(image_obj):
            image_obj = cv2.imread(image_obj)
        else:
            return None    
    else:
        print("No valid image object or file does not exist or is not an image")
        print(type(image_obj))
        return None    

    # Convert the image bytes to a PIL image object
    img_pil = Image.open(io.BytesIO(image_obj))

    # Convert the PIL image object to a NumPy array
    img_array = np.array(img_pil)

    # Convert the NumPy array to a BGR color image
    if img_array.ndim == 2:
        img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Detect the orientation of the image using Tesseract OCR
    results = image_to_osd(img, output_type=Output.DICT)

    # Rotate the image to the correct orientation
    img_rotated = imutils.rotate_bound(img, angle=results["rotate"])

    # Encode the rotated image as a PNG image in memory
    _, img_encoded = cv2.imencode('.png', img_rotated)

    return img_encoded




def _rotate_image(filename):
    # Load the image
    img = cv2.imread(filename)
    imgOrientation = img.copy()

    rgb = cv2.cvtColor(imgOrientation, cv2.COLOR_BGR2RGB)

    results = image_to_osd(rgb, output_type=Output.DICT)
    # display the orientation information
    print("[INFO] detected orientation: {}".format(
        results["orientation"]))
    print("[INFO] rotate by {} degrees to correct".format(
        results["rotate"]))
    #print("[INFO] detected script: {}".format(results["script"]))

    img_rotated = imutils.rotate_bound(img, angle=results["rotate"])

    temp_file = f"{tempfile.gettempdir()}/{uuid4()}.png"
    cv2.imwrite(temp_file, img_rotated)
    return temp_file


def get_buffer_for_obj(obj):
        # Create a BytesIO object
    buffer = io.BytesIO()

    buffer.write(obj.content)
    # Seek to the beginning of the object
    buffer.seek(0)

    return buffer



def get_buffer_for_obj_bytes(obj):
        # Create a BytesIO object
    buffer = io.BytesIO()

    buffer.write(obj)
    # Seek to the beginning of the object
    buffer.seek(0)

    return buffer


def extract_text_from_pdf_object(pdf_object, language="deu+eng", auto_rotate=False):

    # try:
    #     pdf_object = open(pdf_object, "rb")
    # except Exception as e:
    #     print(f"{e.args}")

    pdf_reader = __get_pdf_file_reader(pdf_object)

    # pdf_data = pdf_object
    # try:
        # Read the bytes from the buffer
    pdf_data = pdf_object.getvalue()
    # except:
    #     try:
    #         obj_buffer = file_ocr.get_buffer_for_obj(full_path)
    #         # Read the bytes from the buffer
    #         pdf_data = obj_buffer.getvalue()
    #     except:
    #         pass

    if pdf_reader is None:
        return None
    if pdf_reader.is_encrypted:
        print('    --NOTICE: This file is encrypted and cannot be read by Joplin OCR\n')
        return None
    text = list()
    preview_file = None
    _pages_num = len(pdf_reader.pages)
    print(f"Pages: {_pages_num}")
    for i, x in enumerate(pdf_reader.pages):
        #print(f"i={i} x={x}")
        page = pdf_reader.pages[i]
        extracted_image_obj = pdf_page_as_image_obj(pdf_data, page_num=i)
        # TODO auto_rotate=False ?? 
        extracted_text_list = extract_text_from_image_object(extracted_image_obj,
                                                        language=language,
                                                        auto_rotate=auto_rotate)
        # close the buffer
        #extracted_image_obj.close()

        if extracted_text_list is not None:
            extracted_text = "".join(extracted_text_list.pages)
            print(f"Page {i + 1} of {len(pdf_reader.pages)} processed successfully.")
        else:
            extracted_text = ""
            print(f"Page {i + 1} of {len(pdf_reader.pages)} processed with no text recognized.")
        embedded_text = "" + page.extract_text()
        if len(embedded_text) > len(extracted_text): # TODO check if one is in the given language(s)
            selected_text = embedded_text
        else:
            selected_text = extracted_text
        selected_text = selected_text.strip()
        # 10 or fewer characters is probably just garbage
        if len(selected_text) > 10:
            text.extend([selected_text])
    
    return FileOcrResult(text)



def language_name(code):
    try:
        # Use langcodes to translate the language code into a language name
        return langcodes.Language.make(code).language_name()
    except:
        return "Unknown"
    
def convert_languages_list(languages_list = None):
    if languages_list:
        out_list = list()
        for language in languages_list:
            _language = language_name(language)
            if _language:
                out_list.append(_language)
    else:
        return None
    return out_list
    

def parse_languages(lang_string):
    return lang_string.split("+")




def extract_text_from_image_object(image_object, auto_rotate=False, language="eng"):
    try:
        #img = __get_image(image_object)
        # text = image_to_string(img, lang=language)

        #image_object_bytes_data = image_object.getvalue()
        if isinstance(image_object, io.BytesIO):
            image_object = image_object.getvalue()
        elif isinstance(image_object, str):
            if os.path.isfile(image_object) and imghdr.what(image_object):
                image_object = cv2.imread(image_object)
            else:
                return None    
        else:
            print("No valid image object or file does not exist or is not an image")
            print(type(image_object))
            return None    

        if auto_rotate:
            rotated_image = _rotate_image_obj(image_object)
            if rotated_image is None:
                return None
            result = extract_text_from_image_object(rotated_image, auto_rotate=False, language=language)
            #os.remove(rotated_image)
            if result is None:
               return None
            text = result.pages[0]
        else:
            # Convert the image bytes to a NumPy array
            img_array = np.frombuffer(image_object, dtype=np.uint8)

            # Decode the image array
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            text = image_to_string(rgb, lang=language)
            #text = image_to_string(img, lang=language)
            #image_object.close()
            if text is None:
                return None

        # 10 or fewer characters is probably just garbage
        if len(text.strip()) > 10:
            return FileOcrResult([text.strip()])
        else:
            return None
    except TesseractError as e:
        print(f"TesseractError {e.message}")
        return None


def extract_text_from_image(filename, auto_rotate=False, language="eng"):
    try:
        img = __get_image(filename)
        # text = image_to_string(img, lang=language)

        if auto_rotate:
            rotated_image = _rotate_image(filename)
            if rotated_image is None:
                return None
            result = extract_text_from_image(rotated_image, auto_rotate=False, language=language)
            #os.remove(rotated_image)
            if result is None:
               return None
            text = result.pages[0]
        else:
            text = image_to_string(img, lang=language)
            #text = image_to_string(img, lang=language)

            os.remove(filename)
            if text is None:
                return None

        # 10 or fewer characters is probably just garbage
        if len(text.strip()) > 10:
            return FileOcrResult([text.strip()])
        else:
            return None
    except TesseractError as e:
        print(f"TesseractError {e.message}")
        return None
