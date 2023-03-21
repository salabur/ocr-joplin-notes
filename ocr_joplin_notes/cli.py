# -*- coding: utf-8 -*-

"""Console script for ocr_joplin_notes."""
import sys
import logging
import click

try:
    from ocr_joplin_notes import run_ocr
except ModuleNotFoundError as e:
    import run_ocr
    logging.warning(f"Error Module Not Found - {e.args}")
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
    logging.warning("Error Module Not Found - Set manual version no")
    print("Error Module Not Found - Set manual version no")


@click.command()
@click.option(
    "--mode",
    "mode",
    default="FULL_RUN",
    help="""Specify the mode""",
)
@click.option(
    "--tag",
    "tag",
    default="ocr-test", #None,
    help="""Specify the Joplin tag""",
)
@click.option(
    "--exclude_tags",
    "exclude_tags",
    default=None,
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
        mode="FULL_RUN",
        tag=None,
        exclude_tags=None,
        language="deu+eng",
        add_previews="yes",
        autorotation="yes",
):
    f""" Console script for ocr_joplin_notes.
         ocr_joplin_nodes <mode> 
    """
    run_ocr.set_mode(mode)
    
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
    res = run_ocr.run_mode(mode, tag, exclude_tags)
    if res == 0:
        click.echo("Finished")
        return 0
    else:
        click.echo("Aborted")
        return res


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
