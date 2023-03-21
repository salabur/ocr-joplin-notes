import logging
import os
import tempfile
from uuid import uuid4
import cv2
import numpy as np

import pypdf
from pypdf.errors import PdfReadError
from PIL import Image
from pdf2image import convert_from_path
from pytesseract import image_to_string, TesseractError

import math

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import langcodes

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
        return pypdf.PdfReader(file, strict=False)
    except PdfReadError as e:
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


def __get_image(filename):
    try:
        return Image.open(filename)
    except OSError:
        print(f"Error reading image: {filename}")
        return None

def _rotate_image(filename):
    """
     Tries to deskew the image; will not rotate it more than 90 degrees
    :param filename:
    :return: rotated file
    """
    # Inspired by https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    image = cv2.imread(filename, cv2.IMREAD_COLOR) # Initially decode as color
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    threshold = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coordinates = np.column_stack(np.where(threshold > 0))
    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    rotated_image = cv2.warpAffine(image, matrix, (new_width, new_height),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    temp_file = f"{tempfile.gettempdir()}/{uuid4()}.png"
    cv2.imwrite(temp_file, rotated_image)
    return temp_file

# def __rotate_image(filename):
#     """
#     Tries to deskew the image; will not rotate it more than 90 degrees
#     :param filename:
#     :return: rotated file
#     """
#     # Inspired by https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
#     image = cv2.imread(filename, cv2.IMREAD_COLOR) # Initially decode as color
#     if image is None:
#         return None
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bitwise_not(gray)
#     threshold = cv2.threshold(gray, 0, 255,
#                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     coordinates = np.column_stack(np.where(threshold > 0))
#     angle = cv2.minAreaRect(coordinates)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (height, width) = image.shape[:2]
#     # Calculate the dimensions of the rotated image
#     radians = math.radians(angle)
#     new_width = abs(width * math.cos(radians)) + abs(height * math.sin(radians))
#     new_height = abs(width * math.sin(radians)) + abs(height * math.cos(radians))
#     # Round up the dimensions to the nearest integer
#     new_width, new_height = int(math.ceil(new_width)), int(math.ceil(new_height))
#     # Calculate the center of the image
#     center = (width // 2, height // 2)
#     # Get the rotation matrix and apply it to the image
#     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, matrix, (new_width, new_height),
#                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     temp_file = f"{tempfile.gettempdir()}/{uuid4()}.png"
#     cv2.imwrite(temp_file, rotated_image)
#     return temp_file


def extract_text_from_pdf(filename, language="deu+eng", auto_rotate=False):
    with open(filename, "rb") as file:
        pdf_reader = __get_pdf_file_reader(file)
        if pdf_reader is None:
            return None
        if pdf_reader.is_encrypted:
            print('    --NOTICE: This file is encrypted and cannot be read by Joplin OCR\n')
            return None
        text = list()
        preview_file = None
        _pages_num = len(pdf_reader.pages)
        print(f"Pages: {_pages_num}")
        for i in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[i]
            extracted_image = pdf_page_as_image(filename, page_num=i)
            # TODO auto_rotate=False ?? 
            extracted_text_list = extract_text_from_image(extracted_image,
                                                          language=language,
                                                           auto_rotate=auto_rotate)
            os.remove(extracted_image)
            if extracted_text_list is not None:
                extracted_text = "".join(extracted_text_list.pages)
                print(f"Page {i + 1} of {len(pdf_reader.pages)} processed successfully.")
            else:
                extracted_text = ""
                print(f"Page {i + 1} of {len(pdf_reader.pages)} processed with no text recognized.")
            embedded_text = "" + page.extract_text()
            if len(embedded_text) > len(extracted_text):
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
