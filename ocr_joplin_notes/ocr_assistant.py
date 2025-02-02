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
import ocrmypdf

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
import os
import logging
from enum import Enum

import os
import shutil
import tempfile
import platform
import time
import mimetypes
import json
import requests
import csv
from tabulate import tabulate
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
#from img_processor2 import ImageProcessor
#from api_token import get_token_suffix
from pathlib import Path
import pypdf


import io
import os
import pikepdf
import subprocess
import requests
import sys, os.path

import io
import os
import pikepdf
import ocrmypdf
#from ocrmypdf import OCRMyPDF
import requests
from PIL import Image
from typing import Tuple
import io
import subprocess
import pikepdf
#import fitz
from PIL import Image
import requests

import io
import requests
from PIL import Image
from pytesseract import pytesseract
import pikepdf
from pikepdf import PdfMatrix
from pikepdf.models import image as pikeimage

import io
import requests
from PIL import Image
from pytesseract import pytesseract
import pikepdf
from pdf2image import convert_from_bytes
from deskew import determine_skew
import deskew

import cv2
import numpy as np
import imutils
from pytesseract import Output, image_to_osd

# DEBUGGING
# from .file_ocr import FileOcr

import os
import hashlib
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import threading
import queue

from dataclasses import dataclass, field
from typing import Any

Base = declarative_base()


try:
    #from ocr_joplin_notes import file_ocr
    from ocr_joplin_notes import joplin_data_wrapper
except ModuleNotFoundError as e:
    program_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__))))
    sys.path.append(program_dir)
    #import file_ocr
    import joplin_data_wrapper
    logging.warning(f"Error Module Not Found - {e.args}")
    #print(f"Module Not Found: {e.args}")



#, joplin_data_wrapper
#import file_ocr, joplin_data_wrapper



#dev

global IS_UPLOADING
IS_UPLOADING = False
global MAX_UPLOAD_FILE_SIZE
MAX_UPLOAD_FILE_SIZE = 100000000
global NOTEBOOK_ID
NOTEBOOK_ID = ""
global NOTEBOOK_NAME
NOTEBOOK_NAME = "inbox"
global OBSERVED_FOLDERS
OBSERVED_FOLDERS="b:/temp/joplin/in"
global AUTOTAG
AUTOTAG = False
global MOVETO
MOVETO = "b:/temp/joplin/out"
# ########################
JOPLIN_TOKEN = "not-set"
if os.environ.get('JOPLIN_TOKEN') is not None:
    JOPLIN_TOKEN = "token=" + os.environ['JOPLIN_TOKEN']
else:
    print("Please set the environment variable JOPLIN_TOKEN")
    exit(1)
if os.environ.get('JOPLIN_SERVER') is not None:
    JOPLIN_SERVER = os.environ['JOPLIN_SERVER']
else:
    JOPLIN_SERVER = "http://localhost:41184"
    print("Environment variable JOPLIN_SERVER not set, using default value: http://localhost:41184")


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class DirectoryFileSystemHandler(FileSystemEventHandler):

    def _event_handler(self, path):
        filename, ext = os.path.splitext(path)
        if ext not in (".tmp", ".part", ".crdownload") and ext[:2] not in (".~"):
            filesize = self.valid_file(ext, path)
            if filesize > MAX_UPLOAD_FILE_SIZE:   # was 10000000
                print(f"Filesize = {filesize}. Maybe too big for Joplin, skipping upload")
                return False
            else:
                _priority = 10
                queue_jobs.put((_priority,('upload', f'{path}')))
                # return True
                # i = 1
                # max_retries = 5
                # while i <= max_retries:
                #     if i > 1:
                #         print(f"Retrying file upload {i} of {max_retries}...")
                #     if upload(path) < 0:
                #         time.sleep(5)
                #     else:
                #         return True
                # print(f"Tried {max_retries} times but failed to upload file {path}")
                # return False
        else:
            print("Detected temp file. Temp files are ignored.")

    def valid_file(self, ext, path):
        """Ensure file is completely written before processing"""
        size_past = -1
        while True:
            size_now = os.path.getsize(path)
            if size_now == size_past:
                print(f"File xfer complete. Size={size_now}")
                return size_now
            else:
                size_past = os.path.getsize(path)
                print(f"File transferring...{size_now}")
                time.sleep(1)
        return -1

    def on_created(self, event):
        print(event.event_type + " -- " + event.src_path)
        self._event_handler(event.src_path)

    def on_moved(self, event):
        print(event.event_type + " -- " + event.dest_path)
        self._event_handler(event.dest_path)


class ResourceType(Enum):
    PDF = "pdf"
    IMAGE = "image"


class OcrResult:
    def __init__(self, pages, input_resource_type=ResourceType.IMAGE, preview_file=None, success=True):
        self.pages = pages
        self.input_resource_type = input_resource_type
        self.preview_file = preview_file
        self.success = success



Joplin = joplin_data_wrapper.JoplinDataWrapper(JOPLIN_SERVER, JOPLIN_TOKEN)


def set_language(language):
    global LANGUAGE
    LANGUAGE = language


def set_add_previews(add_previews):
    global ADD_PREVIEWS
    ADD_PREVIEWS = True
    if add_previews == "no":
        ADD_PREVIEWS = False


def set_observed_folders(observed_folders):
    global OBSERVED_FOLDERS
    OBSERVED_FOLDERS = observed_folders
    return OBSERVED_FOLDERS


def set_mode(mode):
    global MODE
    MODE = mode

def set_tag(tag):
    global TAG
    TAG = tag

def set_exclude_tags(exclude_tags):
    global EXCLUDE_TAGS
    EXCLUDE_TAGS = exclude_tags




def set_dry_run(safe):
    global DRY_RUN
    DRY_RUN = safe


def set_autorotation(autorotation):
    global AUTOROTATION
    AUTOROTATION = True
    if autorotation == "no":
        AUTOROTATION = False


def set_autotag(autotag):
    global AUTOTAG
    AUTOTAG = True
    if autotag == "no":
        AUTOTAG = False


def set_moveto(moveto):
    global MOVETO
    if moveto == tempfile.gettempdir():
        moveto = ""
    MOVETO = moveto
    return MOVETO


def set_json_string(title, NOTEBOOK_ID, body, img=None):
    if img is None:
        return '{{ "title": {}, "parent_id": "{}", "body": {} }}'.format(
            json.dumps(title), NOTEBOOK_ID, json.dumps(body)
        )
    else:
        return '{{ "title": "{}", "parent_id": "{}", "body": {}, "image_data_url": "{}" }}'.format(
            title, NOTEBOOK_ID, json.dumps(body), img
        )


SCAN_HEADER = "<!--- OCR data inserted below --->"


class ResultTag(Enum):
    OCR_SKIPPED = "ojn_ocr_skipped"
    OCR_FAILED = "ojn_ocr_failed"
    OCR_ADDED = "ojn_ocr_added"


def initialize_notebook(notebook_name):
    global NOTEBOOK_NAME
    NOTEBOOK_NAME = notebook_name
    global NOTEBOOK_ID
    NOTEBOOK_ID = ""
    return NOTEBOOK_NAME



def set_notebook_id(notebook_name=None):
    """ Find the ID of the destination folder 
    adapted logic from jhf2442 on Joplin forum
    https://discourse.joplin.cozic.net/t/import-txt-files/692
    """
    global NOTEBOOK_NAME
    global NOTEBOOK_ID
    if notebook_name is not None:
        NOTEBOOK_NAME = initialize_notebook(notebook_name)
    try:
        res = requests.get(JOPLIN_SERVER + "/folders" + "?" + JOPLIN_TOKEN)
        folders = res.json()["items"]
        for folder in folders:
            if folder.get("title") == NOTEBOOK_NAME:
                NOTEBOOK_ID = folder.get("id")
        if NOTEBOOK_ID == "":
            for folder in folders:
                if "children" in folder:
                    for child in folder.get("children"):
                        if child.get("title") == NOTEBOOK_NAME:
                            NOTEBOOK_ID = child.get("id")
        return NOTEBOOK_ID
    except requests.ConnectionError as e:
        print("Connection Error - Is Joplin Running?")
        return "err"


def create_resource(filename):
    if NOTEBOOK_ID == "":
        set_notebook_id()
    basefile = os.path.basename(filename)
    title = os.path.splitext(basefile)[0]
    files = {
        "data": (json.dumps(filename), open(filename, "rb")),
        "props": (None, f'{{"title":"{title}", "filename":"{basefile}"}}'),
    }
    response = requests.post(JOPLIN_SERVER + "/resources" + "?" + JOPLIN_TOKEN, files=files)
    _file_ext = response.json()["file_extension"]
    _id = response.json().get("id")
    _mime = response.json().get("mime")
    _created_time = response.json().get("created_time")
    _updated_time = response.json().get("updated_time")
    _user_created_time = response.json().get("user_created_time")
    _user_updated_time = response.json().get("user_updated_time")
    # if len(set([a, b, c, d, e])) > 1:
    # # TODO update ocr sice note has changed? - noo store last change in db and then compare
    
    return response.json()


def delete_resource(resource_id):
    apitext = JOPLIN_SERVER + "/resources/" + resource_id + "?" + JOPLIN_TOKEN
    response = requests.delete(apitext)
    return response


def get_resource(resource_id):
    apitext = JOPLIN_SERVER + "/resources/" + resource_id + "?" + JOPLIN_TOKEN
    response = requests.get(apitext)
    return response

def get_files_sha3_256(file_name, file_path = OBSERVED_FOLDERS):
    if is_file_path(file_name):
        file_full_path = file_name
    else:
        file_full_path = os.path.join(file_path, file_name)
    # Get file information
    with open(file_full_path, 'rb') as f:
        file_hash = hashlib.sha3_256(f.read()).hexdigest()
    return file_hash


def get_buffer_sha3_256(file_buffer):
    # test_file = open(filename, 'rb')
    # test_bytes = io.BytesIO(test_file.read())
    # second_sha = hashlib.sha3_256(test_bytes.getvalue()).hexdigest()

    sha = hashlib.sha3_256(file_buffer.getvalue()).hexdigest()

    #file_hash = hashlib.sha3_256(f.read()).hexdigest()
    return sha



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

def is_file_path(input_str):
    """
    Returns True if the input string is a full file path, False if it is just a file name.
    """
    return os.path.sep in input_str

def ocr_pdf_image(file_path, languages):
    with open(file_path, "rb") as file:
        stream = io.BytesIO(file.read())

    try:
        pdf = pikepdf.Pdf.open(stream)
        img_data = pdf_to_image(stream)
        ocr_text = ocr_all_pages(stream, languages)
    except pikepdf.PdfError:
        img = Image.open(stream)
        img_data = correct_image_obj(img)
        ocr_text = pytesseract.image_to_string(img_data, lang=languages)
    except pikepdf.PasswordError:
        img_data = None
        ocr_text = None
        print("The PDF file is encrypted")

    return ocr_text, img_data


def pdf_to_image(stream):
    images = convert_from_bytes(stream.getvalue(), fmt="png", single_file=False)
    img_data = correct_image_obj(images[0])
    return img_data


def ocr_all_pages(stream, languages):
    ocr_text = ""
    images = convert_from_bytes(stream.getvalue(), fmt="png", single_file=False)

    for i, img in enumerate(images):
        img_data = correct_image_obj(img)
        ocr_text += pytesseract.image_to_string(img_data, lang=languages) + "\n\n"

    return ocr_text

def ocr_image(file_path, languages):
    with open(file_path, "rb") as file:
        stream = io.BytesIO(file.read())

        img = Image.open(stream)
        img_data = correct_image_obj(img)
        ocr_text = pytesseract.image_to_string(img_data, lang=languages)

    return ocr_text

# def correct_image(img):
#     img = img.convert("RGBA")
#     img = img.rotate(0, expand=True, fillcolor=(255, 255, 255))

#     # Deskew and correct orientation
#     img_gray = img.convert("L")
#     img_gray_np = np.array(img_gray)
#     angle = determine_skew(img_gray_np)
#     img = img.rotate(-angle, expand=True, fillcolor=(255, 255, 255))

#     return img

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


def correct_image_obj(img):
    # Convert the PIL Image to a NumPy array for OpenCV
    img_np = np.array(img)
    img_np = img_np[:, :, ::-1].copy()  # Convert from RGB to BGR (OpenCV uses BGR)

    # Detect the orientation and rotation angle
    try:
        results = image_to_osd(img_np, output_type=Output.DICT)
        #print("[INFO] detected orientation: {}".format(results["orientation"]))
        #print("[INFO] rotate by {} degrees to correct".format(results["rotate"]))

        # Rotate the image
        img_rotated = imutils.rotate_bound(img_np, angle=results["rotate"])
    except TesseractError as e:
        logging.warning(f'Error while getting immage rotation: {e}')
        img_rotated = img_np

    # Convert the rotated image back to a PIL Image
    img_rotated = img_rotated[:, :, ::-1].copy()  # Convert from BGR to RGB
    img_rotated_pil = Image.fromarray(img_rotated)

    return img_rotated_pil


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


def encode_image(img, datatype):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes)
    encoded_img = f"data:{datatype};base64,{encoded.decode()}"
    return encoded_img

def encode_file_base64(filename, datatype, file_path = OBSERVED_FOLDERS):
    if is_file_path(filename):
        file_full_path = filename
    else:
        file_full_path = os.path.join(file_path, filename)
            
    encoded = base64.b64encode(open(file_full_path, "rb").read())
    out = f"data:{datatype};base64,{encoded.decode()}"
    return out


def read_text_note(filename, file_path = OBSERVED_FOLDERS):
    if is_file_path(filename):
        file_full_path = filename
    else:
        file_full_path = os.path.join(file_path, filename)
            
    with open(file_full_path, "r") as myfile:
        text = myfile.read()
        print(text)
    return text


def read_csv(filename, file_path = OBSERVED_FOLDERS):
    if is_file_path(filename):
        file_full_path = filename
    else:
        file_full_path = os.path.join(file_path, filename)
            
    return csv.DictReader(open(file_full_path))





def rename_file(basefile, filePath=None):
    if filePath is None:
        # Separate the directory, filename, and extension
        dirname, filename = os.path.split(basefile)
        old_fullpath = basefile
    else:
        dirname = filePath
        filename = basefile
        old_fullpath = os.path.join(filePath, basefile)

    name, ext = os.path.splitext(filename)

    # Construct the new filename and full path
    new_filename = f"__already_imported_{name}{ext}"
    new_filepath = os.path.join(dirname, new_filename)


    # Rename the file
    os.rename(old_fullpath, new_filepath)

    return new_filepath



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










def note_ocr_from_queue(queue): # 
    """ Get the default Notebook ID and process the passed in file"""
    #print(f"(upload_from_queue) Uploading file: {_file_path}")
    # queue_note_ocr
    if not queue.empty():
        message = queue.get()
        print(f"Uploader received a message: {message}")
        _msg_data = message
        if _msg_data[0] == 'note_ocr':
            _note_id = _msg_data[1]
            #print(f"File path: {_file_path}")
            # TODO start in own thread with own queue
            IS_UPLOADING = True
            print(f'OCR running now for note: {_note_id}')
            __perform_ocr_for_note(_note_id)
            IS_UPLOADING = False

            #print(f"Resource: {_res}")
            return True
        else:
            return False

        # basefile = os.path.basename(_file_path)
        # filename = _file_path # TODO this is bullshit, only testing
        #upload(__file_path)


def upload(filename): # 
    """ Get the default Notebook ID and process the passed in file"""
    basefile = os.path.basename(filename)
    title, ext = os.path.splitext(basefile)
    body = f"{basefile} uploaded from {platform.node()}\n"
    datatype = mimetypes.guess_type(filename)[0]
    db_response = db.add_file_info(basefile) #needs accessible path
    # file_info = db.get_file_info_by_file(basefile)
    # print(f'file info - size: {file_info.size_bits} sha:{file_info.sha3_256} name: {file_info.file_name}')


    if datatype is None:
        # avoid subscript exception if datatype is None
        if ext in (".url", ".lnk"):
            datatype = "text/plain"
        else:
            datatype = ""
    if datatype == "text/plain":
        db_response = db.add_file_info(basefile, ocr_status='started')
        body += read_text_note(filename)
        values = set_json_string(title, NOTEBOOK_ID, body)
        db_response = db.add_file_info(basefile, ocr_status='ok')
    if datatype == "text/csv":
        table = read_csv(filename)
        db_response = db.add_file_info(basefile, ocr_status='started')
        body += tabulate(table, headers="keys", numalign="right", tablefmt="pipe")
        values = set_json_string(title, NOTEBOOK_ID, body)
        db_response = db.add_file_info(basefile, ocr_status='ok')

    elif datatype[:5] == "image":

        body += "\n<!---\n"
        try:
            languages = "deu+eng"
            db_response = db.add_file_info(basefile, ocr_status='started')
            body += ocr_image(filename, languages=languages) # type: ignore
        except TypeError:
            print("Unable to perform OCR on this file.")
            db_response = db.add_file_info(basefile, ocr_status='fail')

        except OSError:
            print(f"Invalid or incomplete file - {filename}")
            db_response = db.add_file_info(basefile, ocr_status='fail')
            return -1
        body += "\n-->\n"
        img = encode_file_base64(filename, datatype)
        values = set_json_string(title, NOTEBOOK_ID, body, img)
        db_response = db.add_file_info(basefile, ocr_status='ok')

    else:
        response = create_resource(filename)
        body += f"[{basefile}](:/{response['id']})"
        values = set_json_string(title, NOTEBOOK_ID, body)
        if response["file_extension"] == "pdf":
            if os.path.isfile(filename) and filename.endswith(".pdf"):

                languages = "deu+eng"
                url = "http://localhost:41184"
                token = JOPLIN_TOKEN
                db_response = db.add_file_info(basefile, ocr_status='started')
                ocr_text, img_data = ocr_pdf_image(filename, languages)
                encoded_img = encode_image(img_data, "image/png")
                
                if ocr_text:
                    body += "\n<!---\n"
                    body += ocr_text # type: ignore
                    body += "\n-->\n"
                    db_response = db.add_file_info(basefile, ocr_status='ok')
                else:
                    body += ""
                    db_response = db.add_file_info(basefile, ocr_status='fail')

                values = set_json_string(title, NOTEBOOK_ID, body, encoded_img)

            else:
                print("The file path is not valid or does not lead to a PDF file")
                db_response = db.add_file_info(basefile, ocr_status='fail')
                return -1
        else: # so its not a pdf or image etc. what now
            print("The file extension is not supported") # "should never happen" .D
            pass # TODO

    headers = {'Content-type': 'application/x-www-form-urlencoded; charset=utf-8'}
    response = requests.post(JOPLIN_SERVER + "/notes" + "?" + JOPLIN_TOKEN, data=values.encode('utf-8'), headers=headers)

    if response.status_code == 200:
        if AUTOTAG:
            apply_tags(body, response.json().get("id"))
        print(f"Placed note into notebook {NOTEBOOK_ID}: {NOTEBOOK_NAME}")
        if os.path.isdir(MOVETO):
            moveto_filename = os.path.join(MOVETO, basefile)
            print(moveto_filename)
            if os.path.exists(moveto_filename):
                print(f"{basefile} exists in moveto dir, not moving!")
                # new_filepath = rename_file(basefile, OBSERVED_FOLDERS)
                # print(f"{basefile} exists in moveto dir, not moving! \n File renamed to: {new_filepath}")
            else:
                try:
                    # Give it a few seconds to release file lock
                    time.sleep(5)
                    shutil.move(filename, MOVETO)
                except IOError:
                    print(f"File Locked-unable to move {filename}")
        return 0
    else:
        print("ERROR! NOTE NOT CREATED")
        print("Something went wrong corrupt file or note > max upload file size?")
        return -1



def apply_tags(text_to_match, note_id):
    """ Rudimentary Tag match using OCR'd text """
    res = requests.get(JOPLIN_SERVER + "/tags" + "?" + JOPLIN_TOKEN)
    tags = res.json()["items"]
    counter = 0
    for tag in tags:
        if tag.get("title").lower() in text_to_match.lower():
            counter += 1
            tag_id = tag.get("id")
            response = requests.post(
                JOPLIN_SERVER + f"/tags/{tag_id}/notes" + "?" + JOPLIN_TOKEN,
                data=f'{{"id": "{note_id}"}}',
            )
    print(f"Matched {counter} tag(s) for note {note_id}")
    return counter



def run_mode_queue(mode, tag, exclude_tags, queue):
    if not Joplin.is_valid_connection():
        return -1
    if mode == "TAG_NOTES":
        print("Tagging notes. This might take a while. You can follow the progress by watching the tags in Joplin")
        if tag is None and (exclude_tags is None or len(exclude_tags) == 0):
            Joplin.perform_on_all_note_ids(__tag_note_with_source)
        else:
            tag_id = Joplin.find_tag_id_by_title(tag)
            if tag_id is None:
                print("tag not found")
                return -1
            Joplin.perform_on_tagged_note_ids(__tag_note_with_source, tag_id, exclude_tags, tag)
        return 0
    elif mode == "DRY_RUN":
        set_dry_run(True)
        return __full_run(tag, exclude_tags)
    elif mode == "FULL_RUN":
        set_dry_run(False)
        return __full_run_queue(tag, exclude_tags, queue)
    elif mode == "OBSERV_FOLDER":
        set_dry_run(False)
        return __observ_folder_run(OBSERVED_FOLDERS)
    elif mode == "OBSERV_FOLDER_AND_SCAN":
        set_dry_run(False)
        return __observ_folder_and_scan_run(OBSERVED_FOLDERS)
    else:
        print(f"Mode {mode} not supported")
    return -1



def run_mode(mode, tag, exclude_tags):
    if not Joplin.is_valid_connection():
        return -1
    if mode == "TAG_NOTES":
        print("Tagging notes. This might take a while. You can follow the progress by watching the tags in Joplin")
        if tag is None and (exclude_tags is None or len(exclude_tags) == 0):
            Joplin.perform_on_all_note_ids(__tag_note_with_source)
        else:
            tag_id = Joplin.find_tag_id_by_title(tag)
            if tag_id is None:
                print("tag not found")
                return -1
            Joplin.perform_on_tagged_note_ids(__tag_note_with_source, tag_id, exclude_tags, tag)
        return 0
    elif mode == "DRY_RUN":
        set_dry_run(True)
        return __full_run(tag, exclude_tags)
    elif mode == "FULL_RUN":
        set_dry_run(False)
        return __full_run(tag, exclude_tags)
    elif mode == "OBSERV_FOLDER":
        set_dry_run(False)
        return __observ_folder_run(OBSERVED_FOLDERS)
    elif mode == "OBSERV_FOLDER_AND_SCAN":
        set_dry_run(False)
        return __observ_folder_and_scan_run(OBSERVED_FOLDERS)
    else:
        print(f"Mode {mode} not supported")
    return -1





def __tag_note_with_source(note_id):
    note = Joplin.get_note_by_id(note_id)
    if __is_note_ocr_ready(note):  # TODO also make it possible to tag notes without an attachment
        if note.markup_language == 1:
            language = "markup"
        else:
            language = "html"
        tag_title = "ojn_{}_{}".format(language, note.source.lower().replace(' ', '-'))
        Joplin.create_tag(tag_title)
        Joplin.tag_note(note_id, tag_title)
        return tag_title
    return None


def __is_note_ocr_ready(note):
    if __is_created_by_rest_uploader(note):
        # Skip notes already containing OCR data
        return False
    elif __has_existing_ocr_section(note):
        # Already processed by this application
        return False
    else:
        resources = Joplin.get_note_resources(note.id)
        for res in resources:
            resource = Joplin.get_resource_by_id(res.get("id"))
            if resource.mime[:5] == "image" or resource.mime == "application/pdf":
                return True
    return False


def __has_existing_ocr_section(note):
    return SCAN_HEADER in note.body


def __is_created_by_rest_uploader(note):
    """If the note starts with: <filename> uploaded from <host>
    and has an HTML comment section,
    and has at least one attachment
    assume it contains OCR data from the rest_uploader"""
    first_lines = "{}".format(note.body).split("\n", 3)
    file_name = first_lines[0].split(" uploaded from")[0]
    uploaded_from = "uploaded from" in first_lines[0]
    filename_exists = len(file_name) > 0
    comment_start = "\n<!---\n" in note.body
    comment_end = "\n-->\n" in note.body
    markup = note.markup_language == 1
    #has_attachment = len(Joplin.get_note_resources(note.id)) > 0
    return uploaded_from & filename_exists & comment_start & comment_end & markup #& has_attachment

def __perform_ocr_for_note(note_id):
    note = Joplin.get_note_by_id(note_id)
    result_tag = __ocr_resources(note)
    Joplin.create_tag(result_tag.value)
    Joplin.tag_note(note_id, result_tag.value)
    return result_tag



def __ocr_resource(resource, create_preview=True):
    mime_type = resource.mime
    full_path = Joplin._get_resource_obj(resource)
    obj_buffer = get_buffer_for_obj(full_path)
    # Read the bytes from the buffer
    bytes_data = obj_buffer.getvalue()

    try:
        if mime_type[:5] == "image":
            result = extract_text_from_image_object(obj_buffer, auto_rotate=AUTOROTATION, language=LANGUAGE)
            if result is None:
                return OcrResult(None)
            return OcrResult(result.pages, ResourceType.IMAGE)
        elif mime_type == "application/pdf":
            ocr_result = extract_text_from_pdf_object(obj_buffer, language=LANGUAGE, auto_rotate=AUTOROTATION)
            create_preview = True
            if ocr_result is None:
                return OcrResult(None, success=False)
            if create_preview:
                preview_file = _rotate_image_obj(pdf_page_as_image_obj(bytes_data, is_preview=True))
                # TODO convert
                return OcrResult(ocr_result.pages, ResourceType.PDF, preview_file)
            else:
                return OcrResult(ocr_result.pages, ResourceType.PDF)
    except (TypeError, OSError) as e:
        logging.error(f'Error while OCR: {e.args}')
        return OcrResult(None, success=False)
    finally:
        try:
            #os.remove(full_path)
            logging.info('finish one')
        except PermissionError as e:
            print("Permission Error: " + str(e))
            print("File: " + str(resource.title))
            return OcrResult(None, success=False)
        except TypeError:
            # obj in ram will be deleted by auto gc
            pass


def __ocr_resources(note):
    print(f"------------------------------------\nnote: {note.title}")
    if __is_note_ocr_ready(note):
        result = ""
        resources = Joplin.get_note_resources(note.id)
        for resource_json in resources:
            resource = Joplin.get_resource_by_id(resource_json.get("id"))
            print(f"- file: {resource.title} [{resource.mime}]")
            data = __ocr_resource(resource, create_preview=ADD_PREVIEWS and note.markup_language == 2)
            if data.success is False:
                return ResultTag.OCR_FAILED
            elif data.pages is not None and len(data.pages) > 0:
                print(f"  - pages extracted: {len(data.pages)}")
                resulting_text = ""
                if data.input_resource_type == ResourceType.PDF:
                    for i in range(len(data.pages)):
                        resulting_text += "\n---------- [{}/{}] ----------\n{}".format(i + 1, len(data.pages),
                                                                                       data.pages[i])
                else:
                    resulting_text = data.pages[0]
                result += '\n<!-- [{}]\n{}\n-->'.format(resource.title, resulting_text)
                if data.preview_file is not None:
                    title = "preview-{}.png".format(os.path.splitext(os.path.basename(resource.filename))[0])
                    if not DRY_RUN:
                        preview_file_link = __add_preview(note, title, data)
                        result += "\n{}\n".format(preview_file_link)
                    #os.remove(data.preview_file)
            else:
                print("  - No data found")
        if len(result.strip()) > 0:
            ocr_section = '\n\n{}\n{}\n'.format(SCAN_HEADER, result)
            if not DRY_RUN:
                Joplin.update_note_body(note.id, note.body + ocr_section)
                print("Result: note updated")
            else:
                print("Result: note would have been updated [dry run]")
            return ResultTag.OCR_ADDED
        else:
            print("Result: note not updated")
            return ResultTag.OCR_FAILED
    else:
        print("Result: skipped")
        return ResultTag.OCR_SKIPPED



def __add_preview(note, title, data):
    res_id = Joplin.save_preview_image_object_as_resource(note.id, data.preview_file, title)
    if note.markup_language == 1:
        preview_file_link = "![{}](:/{})".format(title, res_id)
    else:
        preview_file_link = '<img src=":/{}" alt="{}"/>'.format(res_id, title)
    return preview_file_link



# DB functions

class FileInfo(Base):
    __tablename__ = 'file_info'

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String, nullable=False, unique=True)
    sha3_256 = Column(String)
    size_bits = Column(Integer)
    datetime_added = Column(DateTime, default=datetime.datetime.utcnow)
    datetime_changed = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    ocr_status = Column(String, default='')


class NoteInfo(Base):
    __tablename__ = 'note_info'

    id = Column(Integer, primary_key=True, autoincrement=True)
    datetime_created = Column(DateTime, default=datetime.datetime.utcnow)
    datetime_changed = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    note_id = Column(String(64), unique=True)
    note_datetime_created = Column(DateTime, default=datetime.datetime.utcnow)
    note_datetime_changed = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    note_title = Column(String)
    resources = relationship('ResourceInfo', backref='note_info', lazy=True)
    file_id = Column(Integer, ForeignKey('file_info.id'))

class ResourceInfo(Base):
    __tablename__ = 'resource_info'

    id = Column(Integer, primary_key=True, autoincrement=True)
    datetime_created = Column(DateTime, default=datetime.datetime.utcnow)
    datetime_changed = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    resource_datetime_created = Column(DateTime, default=datetime.datetime.utcnow)
    resource_datetime_changed = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    resource_path = Column(String)
    note_id = Column(Integer, ForeignKey('note_info.id'))

class Database:
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_file_info(self, file_name, file_path = OBSERVED_FOLDERS, ocr_status=''):
        if is_file_path(file_name):
            file_full_path = file_name
        else:
            file_full_path = os.path.join(file_path, file_name)

        with open(file_full_path, 'rb') as f:
            file_hash = hashlib.sha3_256(f.read()).hexdigest()
        session = self.Session()

        # Check if file already exists in database
        file_info = session.query(FileInfo).filter_by(sha3_256=file_hash).first()

        if file_info is not None:
            session.close()
            return False

        # Get file information
        file_size_bits = os.path.getsize(file_full_path) * 8

        # Add file information to database
        file_info = FileInfo(
            file_name=file_name,
            sha3_256=file_hash,
            size_bits=file_size_bits,
            ocr_status=ocr_status,
        )
        session.add(file_info)
        session.commit()
        session.close()
        return True

    def add_file_info_by_sha3_256(self, sha3_256, file_size_bits, ocr_status=''):
        session = self.Session()

        # Check if file already exists in database
        file_info = session.query(FileInfo).filter_by(sha3_256=sha3_256).first()

        if file_info is not None:
            session.close()
            return

        # Add file information to database
        file_info = FileInfo(
            sha3_256=sha3_256,
            size_bits=file_size_bits,
            ocr_status=ocr_status,
        )
        session.add(file_info)
        session.commit()
        session.close()


    def get_files_sha3_256(self, file_name, file_path = OBSERVED_FOLDERS):
        if is_file_path(file_name):
            file_full_path = file_name
        else:
            file_full_path = os.path.join(file_path, file_name)
            
        # Get file information
        with open(file_full_path, 'rb') as f:
            file_hash = hashlib.sha3_256(f.read()).hexdigest()
        return file_hash

    def get_file_info_by_file(self, file_name, file_path = OBSERVED_FOLDERS):
        if is_file_path(file_name):
            file_full_path = file_name
        else:
            file_full_path = os.path.join(file_path, file_name)
            
        sha = self.get_files_sha3_256(file_name, file_path)
        file_info = self.get_file_info_by_sha3_256(sha)
        return file_info

    def get_file_info_by_sha3_256(self, sha3_256):
        session = self.Session()
        file_info = session.query(FileInfo).filter_by(sha3_256=sha3_256).first()
        session.close()
        return file_info


    def update_ocr_status(self, sha3_256, ocr_status):
        session = self.Session()
        file_info = session.query(FileInfo).filter_by(sha3_256=sha3_256).first()
        file_info.ocr_status = ocr_status
        session.commit()
        session.close()

    def update_file_info(self, file_name, file_path = OBSERVED_FOLDERS):
        if is_file_path(file_name):
            file_full_path = file_name
        else:
            file_full_path = os.path.join(file_path, file_name)
            
        session = self.Session()

        # Get file information
        with open(file_full_path, 'rb') as f:
            file_hash = hashlib.sha3_256(f.read()).hexdigest()

        # Update file information in database
        file_info = session.query(FileInfo).filter_by(sha3_256=file_hash).first()
        file_info.sha3_256 = file_hash  # may needs to be removed as it will never be updated
        file_info.file_name = file_name        
        file_info.size_bits = os.path.getsize(file_full_path) * 8
        file_info.datetime_changed = datetime.datetime.utcnow()
        session.commit()
        session.close()


    def add_note_info(self, note_id, note_title, file_name):
        session = self.Session()

        # Check if note already exists in database
        note_info = session.query(NoteInfo).filter_by(note_id=note_id).first()

        if note_info is not None:
            session.close()
            return

        # Get file information
        file_info = session.query(FileInfo).filter_by(file_name=file_name).first()

        # Add note information to database
        note_info = NoteInfo(
            note_id=note_id,
            note_title=note_title,
            file_id=file_info.id,
        )
        session.add(note_info)
        session.commit()
        session.close()

    def add_resource_info(self, note_id, resource_path):
        session = self.Session()

        # Check if resource already exists in database
        resource_info = session.query(ResourceInfo).filter_by(resource_path=resource_path).first()

        if resource_info is not None:
            session.close()
            return

        # Add resource information to database
        resource_info = ResourceInfo(
            resource_path=resource_path,
            note_id=note_id,
        )
        session.add(resource_info)
        session.commit()
        session.close()

    def get_note_info(self, note_id):
        session = self.Session()
        note_info = session.query(NoteInfo).filter_by(note_id=note_id).first()
        session.close()
        return note_info

    def update_note_info(self, note_id, note_title):
        session = self.Session()

        # Update note information in database
        note_info = session.query(NoteInfo).filter_by(note_id=note_id).first()
        note_info.note_title = note_title
        session.commit()
        session.close()

    def update_resource_info(self, resource_path):
        session = self.Session()

        # Update resource information in database
        resource_info = session.query(ResourceInfo).filter_by(resource_path=resource_path).first()
        resource_info.resource_datetime_changed = datetime.datetime.utcnow()
        session.commit()
        session.close()



def watcher_helper(path_to_watch=None, _queue=None):
    _queue = _queue
    path_to_watch = path_to_watch
    print(f'watcher starting for path: {path_to_watch}')
    watcher(path_to_watch)


def watcher(path_to_watch=None):
    if path_to_watch is None:
        path_to_watch = str(Path.home())
    event_handler = DirectoryFileSystemHandler()
    print(f"Monitoring directory: {path_to_watch}")
    observer = PollingObserver()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def scanner_run():
    print('scanner is running')
    do_something_important()

def do_something_important():
    x = 1
    for i in range(64):
        x+=1
    print('counted do_something_important')

def threading_task_manager(is_uploading, tag, exclude_tags,_queue=None):
    _queue=_queue
    _file_path = None

    _wait_time_short = 0.05
    _wait_time_long = 1
    #upload_thread = threading.Thread(target=upload_from_queue, args=(queue_upload,))
    # upload_thread = threading.Thread(target=scanner_run, args=(queue_jobs,))
    
                # upload_thread.start()
                # upload_thread.join()

    # TODO
    # create one thread for scanning of not jet OCRed or files (resources) in notes 
    # and a second one to ocr the found note
    _is_running = True
    
    global IS_UPLOADING
    IS_UPLOADING = is_uploading

    while _is_running:
        if not IS_UPLOADING:
            if not queue_jobs.empty():
                message = queue_jobs.get()
                print(f"Watcher received a message: {message}")
                _msg_data =message[1]
                if _msg_data[0] == 'upload':
                    _file_path = _msg_data[1]
                    print(f"File path: {_file_path}")
                    IS_UPLOADING = True
                    print(f'Uploading from directory: {_file_path}')
                    upload(_file_path)
                    IS_UPLOADING = False
                    time.sleep(_wait_time_short)
                elif _msg_data[0] == 'res':
                    _res = _msg_data[1]
                    print(f"Resource: {_res}")
                    time.sleep(_wait_time_short)
                elif _msg_data[0] == 'note_ocr':
                    _note_id = _msg_data[1]
                    print(f"Note ID: {_note_id} added to the OCR list. {_msg_data}")
                    queue_note_ocr.put(('note_ocr', f'{_note_id}'))
                    
                    #time.sleep(_wait_time_short)

                time.sleep(_wait_time_short)

            else:
                
                time.sleep(_wait_time_long)
                if IS_UPLOADING:
                    print(f"IS_UPLOADING: {IS_UPLOADING} ... how?")
                    # upload
                elif not queue_note_ocr.empty():
                    _queue_len = len(queue_note_ocr.queue)
                    IS_UPLOADING = True
                    print(f'Starting to OCR notes from TAG. {_queue_len} to go')
                    
                    note_ocr_from_queue(queue_note_ocr)
                    IS_UPLOADING = False




def __observ_folder_run(observed_folders): # TODO: add support for multiple observed folders
    print(f"Observing folders {observed_folders}.")
    watcher(observed_folders)

def __observ_folder_and_scan_run(observed_folders): # TODO: add support for multiple observed folders
    print(f"Observing folders and scan {observed_folders}.")
    watcher(observed_folders)


def __full_run(tag, exclude_tags):
    print("Starting OCR for tag {}.".format(tag))
    tag_id = Joplin.find_tag_id_by_title(tag)
    if tag_id is None:
        print("Tag not found or specified")
        return -1
    return Joplin.perform_on_tagged_note_ids(__perform_ocr_for_note, tag_id, exclude_tags, tag)

def __full_run_queue(tag, exclude_tags, queue):
    print("Starting OCR for tag {}.".format(tag))
    tag_id = Joplin.find_tag_id_by_title(tag)
    if tag_id is None:
        print("Tag not found or specified")
        return -1
    
    Joplin.perform_on_tagged_note_ids_queue('__perform_ocr_for_note', tag_id, exclude_tags, tag, queue)
    # Joplin.perform_on_tagged_note_ids_queue(__perform_ocr_for_note, tag_id, exclude_tags, tag)

    return 

def mainloop():
    cwd = os.path.dirname(os.path.realpath(__file__))
    # Create a new database
    db_uri = f'sqlite:///{cwd}/ocr_file_db.db'

    global db
    db = Database(db_uri)

    global queue_jobs
    queue_jobs = queue.PriorityQueue()

    global queue_pages
    queue_pages = queue.PriorityQueue()

    global queue_note_ocr
    queue_note_ocr = queue.Queue()

    set_mode('FULL_RUN')
    
    tag="ojn_markup_evernote"
    exclude_tags=None,

    set_tag(tag)
    set_exclude_tags(exclude_tags)
    set_moveto('')
    set_autotag('no')
    set_autorotation(True)
    set_add_previews(True)
    set_language('deu+eng')

    is_uploading = False # TODO : this is bullshit
    global IS_UPLOADING
    IS_UPLOADING = is_uploading

    
    watcher_thread = threading.Thread(target=watcher_helper, args=(OBSERVED_FOLDERS, queue_jobs))
    threading_task_manager_thread = threading.Thread(target=threading_task_manager, args=(is_uploading, tag, exclude_tags, queue_jobs,))
    threading_run_mode_thread = threading.Thread(target=run_mode_queue, args=(MODE, tag, exclude_tags, queue_jobs,))
    
    watcher_thread.start()
    threading_task_manager_thread.start()
    threading_run_mode_thread.start()

    print("is running")
    watcher_thread.join()
    threading_task_manager_thread.join()
    threading_run_mode_thread.join()

    

    #__observ_folder_run(OBSERVED_FOLDERS)
    print("was running")


if __name__ == "__main__":

    mainloop()
