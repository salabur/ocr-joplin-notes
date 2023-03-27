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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
#from img_processor2 import ImageProcessor
#from api_token import get_token_suffix
from pathlib import Path
import pypdf


# DEBUGGING
import file_ocr, joplin_data_wrapper



#REAL DEAL
# try:
#     from ocr_joplin_notes import file_ocr
#     from ocr_joplin_notes import joplin_data_wrapper
# except ModuleNotFoundError as e:
#     import file_ocr, joplin_data_wrapper
#     logging.warning(f"Error Module Not Found - {e.args}")
#     print(f"Module Not Found: {e.args}")





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


"""
2018-09-24 JRK
This program was created to upload files from a folder specified in the
PATH variable to Joplin. The following resource was helpful in figuring out
the logic for Watchdog:
https://stackoverflow.com/questions/18599339/python-watchdog-monitoring-file-for-changes

Tested with the following extensions:
.md
.txt
.pdf
.png
.jpg
.url

Caveat
Uploader only triggered upon new file creation, not modification


Code taken from rest_uploader: https://github.com/cerealkella/rest-uploader

"""
global MAX_UPLOAD_FILE_SIZE
MAX_UPLOAD_FILE_SIZE = 100000000

class MyHandler(FileSystemEventHandler):


    def _event_handler(self, path):
        filename, ext = os.path.splitext(path)
        if ext not in (".tmp", ".part", ".crdownload") and ext[:2] not in (".~"):
            filesize = self.valid_file(ext, path)
            if filesize > MAX_UPLOAD_FILE_SIZE:   # was 10000000
                print(f"Filesize = {filesize}. Maybe too big for Joplin, skipping upload")
                return False
            else:
                i = 1
                max_retries = 5
                while i <= max_retries:
                    if i > 1:
                        print(f"Retrying file upload {i} of {max_retries}...")
                    if upload(path) < 0:
                        time.sleep(5)
                    else:
                        return True
                print(f"Tried {max_retries} times but failed to upload file {path}")
                return False
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
    return response.json()


def delete_resource(resource_id):
    apitext = JOPLIN_SERVER + "/resources/" + resource_id + "?" + JOPLIN_TOKEN
    response = requests.delete(apitext)
    return response


def get_resource(resource_id):
    apitext = JOPLIN_SERVER + "/resources/" + resource_id + "?" + JOPLIN_TOKEN
    response = requests.get(apitext)
    return response



def upload(filename): # TODO Fix removal of img_processor
    """ Get the default Notebook ID and process the passed in file"""
    basefile = os.path.basename(filename)
    title, ext = os.path.splitext(basefile)
    body = f"{basefile} uploaded from {platform.node()}\n"
    datatype = mimetypes.guess_type(filename)[0]
    if datatype is None:
        # avoid subscript exception if datatype is None
        if ext in (".url", ".lnk"):
            datatype = "text/plain"
        else:
            datatype = ""
    if datatype == "text/plain":
        body += file_ocr.read_text_note(filename)
        values = set_json_string(title, NOTEBOOK_ID, body)
    if datatype == "text/csv":
        table = file_ocr.read_csv(filename)
        body += tabulate(table, headers="keys", numalign="right", tablefmt="pipe")
        values = set_json_string(title, NOTEBOOK_ID, body)
    elif datatype[:5] == "image":
        #img_processor = ImageProcessor(LANGUAGE)

        body += "\n<!---\n"
        try:
            body += file_ocr.extract_text_from_image_object(filename, auto_rotate=AUTOROTATION) # type: ignore
        except TypeError:
            print("Unable to perform OCR on this file.")
        except OSError:
            print(f"Invalid or incomplete file - {filename}")
            return -1
        body += "\n-->\n"
        img = file_ocr.encode_image_base64(filename, datatype)
        values = set_json_string(title, NOTEBOOK_ID, body, img)
    else:
        response = create_resource(filename)
        body += f"[{basefile}](:/{response['id']})"
        values = set_json_string(title, NOTEBOOK_ID, body)
        if response["file_extension"] == "pdf":
            if os.path.isfile(filename) and filename.endswith(".pdf"):
                try:
                    with open(filename, "rb") as pdf_file:
                        pdf_reader = pypdf.PdfReader(pdf_file)
                        if pdf_reader.is_encrypted:
                            print("The PDF file is encrypted - OCR or Preview not possible")
                        else:
                            # Special handling for PDFs
                            data = bytes(pdf_file.read())
                            obj_buffer = file_ocr.get_buffer_for_obj_bytes(data)

                            bytes_data = obj_buffer.getvalue()

                            body += "\n<!---\n"
                            body += file_ocr.extract_text_from_pdf_object(obj_buffer) # type: ignore
                            body += "\n-->\n"

                            #obj_buffer = file_ocr.get_buffer_for_obj(pdf_file)
                            # Read the bytes from the buffer
                            #bytes_data = obj_buffer.getvalue()
                            preview_file = file_ocr._rotate_image_obj(file_ocr.pdf_page_as_image_obj(bytes_data, is_preview=True))

                            # #previewfile = img_processor.PREVIEWFILE
                            # if not os.path.exists(preview_file):
                            #     previewfile = img_processor.pdf_page_to_image(filename) # type: ignore
                            img = file_ocr.encode_image_base64(preview_file, "image/png")
                            values = set_json_string(title, NOTEBOOK_ID, body, img)
                except Exception as e:
                    print(f"ERROR: {e} The PDF file is corrupted or cannot be opened")
                    return -1
            else:
                print("The file path is not valid or does not lead to a PDF file")
                return -1


    headers = {'Content-type': 'application/x-www-form-urlencoded; charset=utf-8'}
    response = requests.post(JOPLIN_SERVER + "/notes" + "?" + JOPLIN_TOKEN, data=values.encode('utf-8'), headers=headers)
    #response = requests.post(JOPLIN_SERVER + "/notes" + TOKEN, data=values) old without utf-8 therefore no german umlauts like äöü?

    if response.status_code == 200:
        if AUTOTAG:
            apply_tags(body, response.json().get("id"))
        print(f"Placed note into notebook {NOTEBOOK_ID}: {NOTEBOOK_NAME}")
        if os.path.isdir(MOVETO):
            moveto_filename = os.path.join(MOVETO, basefile)
            print(moveto_filename)
            if os.path.exists(moveto_filename):
                print(f"{basefile} exists in moveto dir, not moving!")
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

def watcher(path_to_watch=None):
    if path_to_watch is None:
        path_to_watch = str(Path.home())
    event_handler = MyHandler()
    print(f"Monitoring directory: {path_to_watch}")
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



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
    else:
        print(f"Mode {mode} not supported")
    return -1


def __full_run(tag, exclude_tags):
    print("Starting OCR for tag {}.".format(tag))
    tag_id = Joplin.find_tag_id_by_title(tag)
    if tag_id is None:
        print("Tag not found or specified")
        return -1
    return Joplin.perform_on_tagged_note_ids(__perform_ocr_for_note, tag_id, exclude_tags, tag)


def __observ_folder_run(observed_folders): # TODO: add support for multiple observed folders
    print(f"Observing folders {observed_folders}.")
    watcher(observed_folders)
    # tag_id = Joplin.find_tag_id_by_title(tag)
    # if tag_id is None:
    #     print("Tag not found or specified")
    #     return -1
    # return Joplin.perform_on_tagged_note_ids(__perform_ocr_for_note, tag_id, exclude_tags, tag)



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


class ResourceType(Enum):
    PDF = "pdf"
    IMAGE = "image"


class OcrResult:
    def __init__(self, pages, input_resource_type=ResourceType.IMAGE, preview_file=None, success=True):
        self.pages = pages
        self.input_resource_type = input_resource_type
        self.preview_file = preview_file
        self.success = success


def __ocr_resource(resource, create_preview=True):
    mime_type = resource.mime
    #full_path = Joplin.save_resource_to_file(resource)
    full_path = Joplin._get_resource_obj(resource)
    obj_buffer = file_ocr.get_buffer_for_obj(full_path)
    # Read the bytes from the buffer
    bytes_data = obj_buffer.getvalue()

    try:
        if mime_type[:5] == "image":
            result = file_ocr.extract_text_from_image_object(obj_buffer, auto_rotate=AUTOROTATION, language=LANGUAGE)
            if result is None:
                return OcrResult(None)
            return OcrResult(result.pages, ResourceType.IMAGE)
        elif mime_type == "application/pdf":
            ocr_result = file_ocr.extract_text_from_pdf_object(obj_buffer, language=LANGUAGE, auto_rotate=AUTOROTATION)
            create_preview = True
            if ocr_result is None:
                return OcrResult(None, success=False)
            if create_preview:
                # preview_file = file_ocr._rotate_image(file_ocr.pdf_page_as_image(full_path, is_preview=True))
                # preview_file = file_ocr._scale_image_object(file_ocr._rotate_image_obj(file_ocr.pdf_obj_page_as_image(bytes_data, is_preview=True)))
                preview_file = file_ocr._rotate_image_obj(file_ocr.pdf_page_as_image_obj(bytes_data, is_preview=True))
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
