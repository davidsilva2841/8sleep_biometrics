import re
import subprocess


def convert_pascal_case_to_camel_case(text):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()


def convert_camel_case_to_pascal_case(text):
    return ''.join(word.title() for word in text.split('_'))

def copy_to_clipboard(text):
    subprocess.run("pbcopy", universal_newlines=True, input=text)

