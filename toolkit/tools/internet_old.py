"""OLD"""
# from bs4 import BeautifulSoup
# from urllib.request import urlretrieve
# import requests
#
# from toolkit.logger import get_logger
# from toolkit.tools import progress_bar, append_to_json_array
# log = get_logger()
#
#
# class FileLink:
#     def __init__(self, url, file_name):
#         self.url = url
#         self.file_name = file_name
#
#
#
# def get_all_links(url):
#     """
#     Get all links from a url.
#
#     Args:
#         url (str):
#
#     Returns:
#         list: URL links list
#     """
#     resp = requests.get(url)
#     if resp.status_code != 200:
#         raise Exception(f'Bad response: \n{resp.text}')
#
#     soup = BeautifulSoup(resp.content, 'html.parser')
#     return soup.find_all('a')
#
#
# def get_all_file_links(url, extension=None, base_url=None, keyword=None):
#     """
#     Get all links from a url.
#
#     Args:
#         url (str):
#         extension (str): Extension to look for (Examples: .zip, .csv, etc.)
#         base_url (str): Add a base url to the file link
#         keyword (str):
#
#     Returns:
#         (list of FileLink): List of FileLink objects
#     """
#
#     links = get_all_links(url)
#
#     urls = []
#     for link in links:
#         url = link.get('href')
#         if url is not None:
#             if base_url:
#                 url = f'{base_url}{url}'
#             if extension and url.endswith(extension):
#                 urls.append(FileLink(url, url.split('/')[-1]))
#             elif keyword and keyword in url:
#                 urls.append(FileLink(url, url.split('/')[-1]))
#
#
#     return urls
#
#
# def download_file(url, folder_path='', file_name='', file_path=''):
#     """
#     Downloads a file from the web.
#
#     Args:
#         url (str): Url to download file
#         folder_path (str): Destination folder
#         file_name (str): Save file name as
#         file_path (str): Full file path
#     """
#
#     if not file_path:
#         if not folder_path:
#             folder_path = TEMP_FOLDER_PATH
#         if not file_name:
#             file_name = url.split('/')[-1]
#         file_path = folder_path + file_name
#
#     log.debug(f'Downloading file | URL: {url} | File path: {file_path}')
#     urlretrieve(url, file_path)
#
#
# def download_files(file_links, to_folder, log_path=''):
#     """
#     Downloads a list of files
#
#     Args:
#         file_links (list of FileLink): List of FileLink objects
#         to_folder (str): Folder to download to
#         log_path (str): Log file path to append file to download_history
#     """
#     bar = progress_bar(len(file_links))
#     for file_link in file_links:
#         download_file(file_link.url, file_path=f'{to_folder}{file_link.file_name}')
#
#         bar.update()
#
#
#
# def extract_new_files(file_links, logs):
#     """
#     Checks download history and looks for new file links
#
#     Args:
#         file_links (list of FileLink): List of FileLink objects
#         logs (dict):
#
#     Returns:
#         (list of FileLink): List of FileLink objects
#     """
#     new_files = []
#     for file_link in file_links:
#         if file_link.file_name not in logs['download_history']:
#             new_files.append(file_link)
#
#     return new_files
#
