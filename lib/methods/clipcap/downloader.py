#@title Drive Downloader
import gdown

class Downloader(object):
    def __init__(self):
        pass

    def download_file(self, file_id, file_dst):
        gdown.download(id=file_id, output=file_dst)
        #!gdown --id $file_id -O $file_dst

downloader = Downloader()