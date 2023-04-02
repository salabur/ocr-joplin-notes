import json
import tempfile
import logging
import cv2

import tempfile
import json
from uuid import uuid4
import io
import time


try:
    from ocr_joplin_notes import rest
except ModuleNotFoundError as e:
    import rest
    logging.warning(f"Error Module Not Found - {e.args}")
    print(f"Module Not Found: {e.args}")
    


class JoplinNote:
    def __init__(self, json_data):
        self.id = json_data.get("id", None)
        self.title = json_data.get("title", None)
        self.body = json_data.get("body", None)
        self.source = json_data.get("source", None)
        self.markup_language = json_data.get("markup_language", None)


class JoplinResource:
    def __init__(self, json_data):
        self.id = json_data.get("id", None)
        self.filename = json_data.get("filename", None)
        self.mime = json_data.get("mime", None)
        self.title = json_data.get("title", None)


class JoplinDataWrapper:

    def __init__(self, server, token):
        self.REST = rest.RestApi(server, token)
        self.NOTES = []
        self.NOTES_amount_total = 0
        self.NOTES_amount_worked = 0

    @staticmethod
    def __paginate_by_title(page: int):
        return {'order_by': 'title',
                'limit': '100',
                'page': f"{page}"
                }

    def get_all_tags_from_note(self, note_id: str):
        if note_id is None:
            return None
        res = self.REST.rest_get('/notes/{}/tags'.format(note_id), params={'fields': 'title'})
        tags = res.json()["items"]
        list_of_tags = list([dic.get("title") for dic in tags])
        return list_of_tags

    def is_valid_connection(self):
        try:
            res = self.REST.rest_get('/notes')
            error = res.json()["error"]
            print(error)
            return False
        except KeyError as e:
            return True

    def find_tag_id_by_title(self, title: str, page: int = 1):
        if title is None:
            return None
        res = self.REST.rest_get('/tags', params=self.__paginate_by_title(page))
        tags = res.json()["items"]
        for tag in tags:
            if tag.get("title") == title:
                return tag.get("id")
        if res.json()["has_more"]:
            return self.find_tag_id_by_title(title, page + 1)
        else:
            return None

    def create_tag(self, title):
        tag_id = self.find_tag_id_by_title(title.lower())
        if tag_id is None:
            res = self.REST.rest_post("/tags", data='{{ "title" : {} }}'.format(json.dumps(title)))
            tag_id = res.json()["id"]
        return tag_id

    def delete_tag(self, title):
        tag_id = self.find_tag_id_by_title(title)
        if tag_id is not None:
            res = self.REST.rest_delete("/tags/{}".format(tag_id))
            return res.status_code
        return None

    def tag_note(self, note_id, tag_title):
        tag_id = self.find_tag_id_by_title(tag_title)
        res = self.REST.rest_post("/tags/{}/notes".format(tag_id), data='{{ "id" : {} }}'.format(json.dumps(note_id)))
        return tag_id


    def get_all_notes_with_tag_id(self, tag_id, tag_name):
        print(f"Getting all notes with tag '{tag_name}'")
        _notes_all = None
        page = 1

        res = self.REST.rest_get('/tags/{}/notes'.format(tag_id), params=self.__paginate_by_title(page))
        _notes = res.json()["items"]

        _notes_all = _notes
        more = True
        while more:
            if res.json()["has_more"]:         
                print("...")       
                page += 1
                _notes = None
                res = self.REST.rest_get('/tags/{}/notes'.format(tag_id), params=self.__paginate_by_title(page))
                _notes = res.json()["items"]
                _notes_all = _notes_all + _notes

            else:
                more = False

        return _notes_all


    def perform_on_tagged_note_ids(self, usage_function, tag_id, exclude_tags, tag_name):
        # res = self.REST.rest_get('/tags/{}/notes'.format(tag_id), params=self.__paginate_by_title(page))
        # notes = res.json()["items"]

        notes = self.get_all_notes_with_tag_id(tag_id=tag_id, tag_name=tag_name)
        notes_amount = len(notes)
        notes_ready = 0
        start_time = time.time()
        for note in notes:
            note_id = note.get("id")
            all_tags = self.get_all_tags_from_note(note_id)  # get all tags of the current note
            # check if any tag in the list exclude_tags is equal to any tag of the current notes' tags
            if len(set(exclude_tags).intersection(all_tags)) == 0:
                usage_function(note_id)
            
            else:
                note = self.get_note_by_id(note_id)
                excluded_tags = set(exclude_tags).intersection(all_tags)
                print(f"------------------------------------\nnote: {note.title}")
                print(f"Excluding this note because it contains the following excluded tags: {', '.join(excluded_tags)}")


            notes_ready+=1
            notes_perc = 100 / notes_amount * notes_ready

            elapsed_time = time.time() - start_time
            time_per_iteration = elapsed_time / notes_ready
            remaining_time = time_per_iteration * (notes_amount - notes_ready)

            remaining_str = self.format_time_string(remaining_time)
            elapsed_time_str = self.format_time_string(elapsed_time)

            print(f"{notes_ready}/{notes_amount} ({notes_perc:.2f}%) of notes ready, {remaining_str} remaining - {elapsed_time_str} elapsed")


        print("All notes processed.")
        return None



    def perform_on_tagged_note_ids_queue(self, usage_function, tag_id, exclude_tags, tag_name, queue):
        # res = self.REST.rest_get('/tags/{}/notes'.format(tag_id), params=self.__paginate_by_title(page))
        # notes = res.json()["items"]

        notes = self.get_all_notes_with_tag_id(tag_id=tag_id, tag_name=tag_name)
        notes_amount = len(notes)
        notes_ready = 0
        _priority = 100
        start_time = time.time()
        for note in notes:
            note_id = note.get("id")
            all_tags = self.get_all_tags_from_note(note_id)  # get all tags of the current note
            # check if any tag in the list exclude_tags is equal to any tag of the current notes' tags
            if len(set(exclude_tags).intersection(all_tags)) == 0:
                # usage_function(note_id)
                if usage_function == '__perform_ocr_for_note':
                    queue.put((_priority,('note_ocr', note_id)))
                else:
                    print(f"find a soloution!: {usage_function}")
            
            else:
                note = self.get_note_by_id(note_id)
                excluded_tags = set(exclude_tags).intersection(all_tags)
                print(f"------------------------------------\nnote: {note.title}")
                print(f"Excluding this note because it contains the following excluded tags: {', '.join(excluded_tags)}")


            notes_ready+=1
            notes_perc = 100 / notes_amount * notes_ready

            elapsed_time = time.time() - start_time
            time_per_iteration = elapsed_time / notes_ready
            remaining_time = time_per_iteration * (notes_amount - notes_ready)

            remaining_str = self.format_time_string(remaining_time)
            elapsed_time_str = self.format_time_string(elapsed_time)

            print(f"{notes_ready}/{notes_amount} ({notes_perc:.2f}%) of notes in queue, {remaining_str} remaining - {elapsed_time_str} elapsed")


        print("All notes processed.")
        return None



    def get_all_notes(self):
        print("Getting all notes")
        self.NOTES = None
        self.NOTES = []
        page = 1
        res = self.REST.rest_get('/notes', params=self.__paginate_by_title(page))
        notes = res.json()["items"]
        self.NOTES = self.NOTES + notes
        more = True
        while more:
            if res.json()["has_more"]:
                page += 1
                _notes = None
                res = self.REST.rest_get('/notes', params=self.__paginate_by_title(page))
                _notes = res.json()["items"]
                self.NOTES = self.NOTES + _notes

            else:
                more = False

        return self.NOTES



    def format_time_string(self, seconds = 1):
        if not seconds or seconds < 0 or seconds > 86400:
            return False
        remaining_time = seconds
        if remaining_time > 86400:
            days = int(remaining_time / 86400)
            if days == 1:
                remaining_str = time.strftime("1 day, %H:%M:%S", time.gmtime(remaining_time))
            else:
                remaining_str = time.strftime(f"{days} days, %H:%M:%S", time.gmtime(remaining_time))
        elif remaining_time > 3600:
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
        elif remaining_time > 60:
            remaining_str = time.strftime("%M:%S", time.gmtime(remaining_time))
        else:
            remaining_str = f"{remaining_time:.0f} seconds"

        return remaining_str


    def perform_on_all_note_ids(self, usage_function, page: int = 1):
        # res = self.REST.rest_get('/notes', params=self.__paginate_by_title(page))
        # notes = res.json()["items"]
        # _notes_stored = self.NOTES
        notes = self.get_all_notes()
        notes_amount = len(notes)
        notes_ready = 0
        start_time = time.time()
        for note in notes:
            usage_function(note.get("id"))

            notes_ready+=1
            notes_perc = 100 / notes_amount * notes_ready

            elapsed_time = time.time() - start_time
            time_per_iteration = elapsed_time / notes_ready
            remaining_time = time_per_iteration * (notes_amount - notes_ready)

            remaining_str = self.format_time_string(remaining_time)
            elapsed_time_str = self.format_time_string(elapsed_time)

            print(f"{notes_ready}/{notes_amount} ({notes_perc:.2f}%) of notes ready, {remaining_str} remaining - {elapsed_time_str} elapsed")


        print("All notes processed.")
        return None

    # original function        
    # def perform_on_all_note_ids(self, usage_function, page: int = 1):
    #     res = self.REST.rest_get('/notes', params=self.__paginate_by_title(page))
    #     notes = res.json()["items"]
    #     notes_amount = len(notes)
    #     for note in notes:
    #         usage_function(note.get("id"))
    #     if res.json()["has_more"]:
    #         return self.perform_on_all_note_ids(usage_function, page + 1)
    #     else:
    #         return None

    def get_note_by_id(self, note_id):
        res = self.REST.rest_get('/notes/{}'.format(note_id), params={'fields': 'id,title,body,source,markup_language'})
        return JoplinNote(res.json())

    def update_note_body(self, note_id, new_body: str):
        res = self.REST.rest_put("/notes/{}".format(note_id), values='{{ "body" : {} }}'.format(json.dumps(new_body)))

    def save_resource_to_file(self, resource: JoplinResource):
        file_download = self.REST.rest_get('/resources/{}/file'.format(resource.id), None)
        full_path = tempfile.mktemp(dir=tempfile.tempdir)
        with open(full_path, 'wb') as f:
            f.write(file_download.content)
        return full_path

    def _get_resource_obj(self, resource: JoplinResource):
        file_download = self.REST.rest_get('/resources/{}/file'.format(resource.id), None)

        return file_download
    
    def get_note_resources(self, note_id):
        res = self.REST.rest_get("/notes/{}/resources/".format(note_id), None)
        return res.json()["items"]

    def get_resource_by_id(self, resource_id):
        res = self.REST.rest_get('/resources/{}'.format(resource_id), params={'fields': 'id,title,filename,mime'})
        return JoplinResource(res.json())

    def save_preview_image_as_resource(self, note_id, filename: str, title: str):
        with open(filename, "rb") as file:
            props = f'{{"title":"{title}", "filename":"{title}.png"}}'
            files = {
                "data": (json.dumps(filename), file),
                "props": (None, props),
            }
            headers = {'Content-type': 'application/x-www-form-urlencoded; charset=utf-8'}
            res = self.REST.rest_post("/resources/{}".format(note_id), files=files, headers=headers)
            return res.json()["id"]

    def save_preview_image_object_as_resource(self, note_id, image_object, title: str):
        
        _image_object = image_object

        # Decode the image array
        img = cv2.imdecode(_image_object, cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a buffer to write the image
        buffer = io.BytesIO()

        # Encode the image to PNG format and write it to the buffer
        _, encoded = cv2.imencode('.png', rgb)
        buffer.write(encoded)

        # Reset the buffer position to the beginning
        buffer.seek(0)

        xy = buffer.getvalue()

        _filename = f"{title}.png"
        props = f'{{"title":"{title}", "filename":"{title}.png"}}'

        files = {
            "data": (json.dumps(_filename), xy),
            "props": (None, props),
        }

        #headers = {'Content-type': 'application/x-www-form-urlencoded; charset=utf-8'}
        res = self.REST.rest_post("/resources/{}".format(note_id), files=files)
        return res.json()["id"]
