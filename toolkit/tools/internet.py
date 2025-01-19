from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import requests

from toolkit.config import TEMP_FOLDER_PATH
from toolkit.logger import get_logger
log = get_logger()



def get_links_from_webpage(url, extension_filters=None):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f'Bad response: \n{resp.text}')

    soup = BeautifulSoup(resp.content, 'html.parser')
    elements = soup.find_all('a')

    if extension_filters is None: extension_filters =[]
    links = []
    for element in elements:
        link = element.get('href')

        if link:
            if extension_filters:
                for extension in extension_filters:
                    if link.lower().endswith(extension.lower()):
                        links.append(link)
                        break
            else:
                links.append(link)
    return links


def download_file(url, folder_path='', file_name='', file_path=''):
    """
    Downloads a file from the web.

    Args:
        url (str): Url to download file
        folder_path (str): Destination folder
        file_name (str): Save file name as
        file_path (str): Full file path
    """

    if not file_path:
        if not folder_path:
            folder_path = TEMP_FOLDER_PATH
        if not file_name:
            file_name = url.split('/')[-1]
        file_path = folder_path + file_name

    log.debug(f'Downloading file | URL: {url} | File path: {file_path}')
    urlretrieve(url, file_path)
    return file_path

