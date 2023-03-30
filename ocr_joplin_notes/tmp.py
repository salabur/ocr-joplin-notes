    # sha_test = 'cb83adbd8abde0f84268ce6922f194801fc379ec00f77988e595d2dff757b216'
    # if sha_test == file_info.sha3_256:
    #     print('sha ist still the same..')
    #     #test_bytes = io.BytesIO()
    #     test_file = open(filename, 'rb')
    #     test_bytes = io.BytesIO(test_file.read())
    #     second_sha = hashlib.sha3_256(test_bytes.getvalue()).hexdigest()
    #     if second_sha == file_info.sha3_256:
    #         print('aaand sha ist still the same..')
    #         func_test1 = get_files_sha3_256(filename, OBSERVED_FOLDERS)
    #         func_test2 = get_buffer_sha3_256(test_bytes)
    #         if func_test1 == func_test2 == second_sha:
    #             print('yes, sha is the same..')



    # resources = Joplin.get_note_resources(note.id)
    # for res in resources:
    #     resource = Joplin.get_resource_by_id(res.get("id"))
    #     if resource.mime[:5] == "image" or resource.mime == "application/pdf":
    #         return True

