
import os
import re
import requests
import time

if __name__ == "__main__":
    folder = "/home/jintao/Desktop/0_coding/0_python/1_ml_alg/mini-lightning/mini_lightning"
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isdir(path):
            continue
        with open(path) as f:
            text = f.read()
        url_list = re.findall(r"https://\S+", text)
        for url in url_list:
            resp = requests.get(url)
            print(resp.status_code, url, path)
            if resp.status_code != 200:
                time.sleep(1)
