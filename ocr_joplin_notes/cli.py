# -*- coding: utf-8 -*-

"""Console script for ocr_joplin_notes."""
import sys
import logging
import click

try:
    from ocr_joplin_notes import run_ocr
except ModuleNotFoundError as e:
    import run_ocr
    logging.info(f"Error Module Not Found - {e.args}")
    print(f"Module Not Found: {e.args}")

#import ocr_joplin_notes


def parse_argument(arg):
    """Helper function for wild arguments"""
    if arg in ["No", "N", "NO", "OFF", "off", "n", "no"]:
        return "no"
    else:
        return "yes"

try:
    __version=ocr_joplin_notes.__version__
except NameError:
    __version = "0.0.1"
    logging.info("Error Module Not Found - Set manual version no")
    print("Error Module Not Found - Set manual version no")


@click.command()
# @click.argument(
#     "observed_folders",
#     type=click.Path(
#         exists=True,
#         file_okay=False,
#         dir_okay=True,
#         writable=False,
#         readable=True,
#         resolve_path=True,
#     ),
# )

@click.option(
    "--observed-folders",
    "observed_folders",
    default="b:/temp/joplin/in",
    help="""Define the path to the folder containing the files to be processed.""",
)

@click.option(
    "--mode",
    "mode",
    default="OBSERV_FOLDER",
    help="""Specify the mode""",
)
@click.option(
    "--tag",
    "tag",
    default="ojn_markup_evernote", #None,
    help="""Specify the Joplin tag""",
)
@click.option(
    "--exclude_tags",
    "exclude_tags",
    default=["ojn_ocr_added","ojn_ocr_skipped","ojn_ocr_failed"],
    multiple=True,
    help="""Specify the Joplin tags to be excluded""",
)
@click.option(
    "-l",
    "--language",
    "language",
    default="deu+eng",
    help="""Specify the OCR Language. Refer to the Tesseract documentation found here: 
    https://github.com/tesseract-ocr/tesseract/wiki""",
)
@click.option(
    "--add-previews",
    "add_previews",
    default="yes",
    help="""Specify whether to add preview images to the note, when a PDF file is processed. """
    """Default = yes (specify 'no' to disable). """,
)
@click.option(
    "--autorotation",
    "autorotation",
    default="yes",
    help="""Specify whether to rotate images."""
         """ Default = yes (specify 'no' to disable). """,
)
@click.version_option(version=__version)
def main(
        mode="OBSERV_FOLDER",
        tag="ojn_markup_evernote",
        exclude_tags=None,
        language="deu+eng",
        add_previews="yes",
        autorotation="yes",
        destination="inbox",
        observed_folders="b:/temp/joplin/in",
        moveto="b:/temp/joplin/out",
):
    f""" Console script for ocr_joplin_notes.
         ocr_joplin_nodes <mode> 
    """
    run_ocr.set_mode(mode)
    notebook_id = run_ocr.set_notebook_id(destination.strip())
    if notebook_id == "err":
        click.echo("Joplin may not be running, please ensure it is open.")
        click.echo("     will check again when processing a file.")
    elif notebook_id == "":
        click.echo(f"Invalid Notebook, check to see if {destination.strip()} exists.")
        click.echo(f"Please specify a valid notebook. Quitting application.")
        return 0
    else:
        click.echo(f"Found Notebook ID: {notebook_id}")
    run_ocr.set_language(language)
    run_ocr.set_autorotation(parse_argument(autorotation))
    run_ocr.set_add_previews(parse_argument(add_previews))
    click.echo("Mode: " + mode)
    if tag is not None:
        click.echo("Tag: " + tag)
    if exclude_tags is not None:
        click.echo("Exclude Tags: " + str(exclude_tags))
    click.echo("Language: " + language)
    click.echo("Add previews: " + add_previews)
    click.echo("Autorotation: " + autorotation)
    
    observed_folders = run_ocr.set_observed_folders(observed_folders)
    moveto = run_ocr.set_moveto(moveto)


    if observed_folders == "":
        click.echo("Files will remain in the monitoring directory")
    else:
        click.echo(f"Folder {observed_folders} will be observed")
    
        if moveto == "":
            click.echo("Files will remain in the monitoring directory")
        else:
            click.echo("File move to location: " + moveto)


    res = run_ocr.run_mode(mode, tag, exclude_tags)
    if res == 0:
        click.echo("Finished")
        return 0
    else:
        click.echo("Aborted")
        return res


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
