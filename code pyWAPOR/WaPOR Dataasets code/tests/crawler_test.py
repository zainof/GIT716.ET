import os
import glob
import requests
from pywapor.general.logger import adjust_logger
from pywapor.collect.protocol.crawler import _download_url, _download_urls, download_url, download_urls

def test_1(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    url = "https://httpstat.us/200?test_200.txt"
    fp = os.path.join(folder, "test1_200.txt")
    session = requests.Session()
    out_fp = _download_url(url, fp, session = session, waitbar = True, headers = None)
    assert out_fp == fp

def test_2(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    url = "https://httpstat.us/400?test_400.txt"
    fp = os.path.join(folder, "test2_400.txt")
    # session = requests.Session()
    try:
        _ = download_url(url, fp, session = None, waitbar = True, headers = None, 
                            max_tries = 2, wait_sec = 2)
    except Exception as e:
        error = e
    assert "Could not download" in error.args[0]

def test_3(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    codes = [200, 200, 200, 200, 200, 400]
    urls = [f"https://httpstat.us/{code}?test_{code}_{i}.txt" for i, code in enumerate(codes)]

    session = requests.Session()
    out_fps = _download_urls(urls, folder, session, fps = None, parallel = 0, headers = None,
                                max_tries = 2, wait_sec = 2)
    ref_out_fps = [os.path.join(folder, f"{code}?test_{code}_{i}.txt") for i, code in enumerate(codes) if code < 400]
    assert out_fps == ref_out_fps

def test_4(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    codes = [200, 200, 200, 200, 200, 400, 200, 200, 401, 403, 
            404, 200, 200, 500, 501, 200, 200, 200, 200]
    urls = [f"https://httpstat.us/{code}?test_{code}_{i}.txt" for i, code in enumerate(codes)]

    session = requests.Session()
    def checker_function(fp):
        return os.path.isfile(fp)
    out_fps = download_urls(urls, folder, session = session, fps = None, parallel = 0, 
                    headers = None, checker_function = checker_function,
                    max_tries = 2, wait_sec = 2, meta_max_tries = 2)
    ref_out_fps = [os.path.join(folder, f"{code}?test_{code}_{i}.txt") for i, code in enumerate(codes) if code < 400]
    assert out_fps == ref_out_fps

def test_5(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    codes = [200, 200, 200, 200, 200, 400, 200, 200, 401, 403, 
            404, 400, 200, 200, 500, 501, 500, 200, 200, 200, 200]
    urls = [f"https://httpstat.us/{code}?test_{code}_{i}.txt" for i, code in enumerate(codes)]
    test_folder = os.path.join(folder, "test5")
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    for fn in glob.glob(os.path.join(test_folder, "*")):
        os.remove(fn)
    session = requests.Session()
    def checker_function(fp):
        i = int(os.path.split(fp)[-1].split("_")[-1][:-4])
        if (i % 2) == 0:
            is_relevant = False
        else:
            is_relevant = True
        return is_relevant
    out_fps = download_urls(urls, test_folder, session = session, fps = None, parallel = 0, 
                    headers = None, checker_function = checker_function,
                    max_tries = 2, wait_sec = 2, meta_max_tries = 2)
    ref_out_fps = [os.path.join(test_folder, f"{code}?test_{code}_{i}.txt") for i, code in enumerate(codes) if code < 400 and (i % 2) != 0]
    assert out_fps == ref_out_fps

def test_6(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    codes = [200, 200, 200, 200, 200, 400, 200, 200, 401, 403, 
            404, 400, 200, 200, 500, 501, 500, 200, 200, 200, 200]
    urls = [f"https://httpstat.us/{code}?test_{code}_{i}.txt" for i, code in enumerate(codes)]

    session = requests.Session()
    checker_function = None
    fps = [os.path.join(folder, f"{i}_{code}.txt") for i, code in enumerate(codes)]
    out_fps = download_urls(urls, folder, session = session, fps = fps, parallel = 0, 
                    headers = None, checker_function = checker_function,
                    max_tries = 2, wait_sec = 2, meta_max_tries = 2)
    ref_out_fps = [fp for code, fp in zip(codes, fps) if code < 400]
    assert out_fps == ref_out_fps

def test_7(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    codes = [200, 200, 200, 200, 200, 400, 200, 200, 401, 403, 
            404, 400, 200, 200, 500, 501, 500, 200, 200, 200, 200]
    urls = [f"https://httpstat.us/{code}?test_{code}_{i}.txt" for i, code in enumerate(codes)]

    session = requests.Session()
    def checker_function(fp):
        i = int(os.path.split(fp)[-1].split("_")[0])
        if (i % 2) == 0:
            is_relevant = False
        else:
            is_relevant = True
        return is_relevant
    fps = [os.path.join(folder, f"{i}_{code}.txt") for i, code in enumerate(codes)]
    out_fps = download_urls(urls, folder, session = session, fps = fps, parallel = 0, 
                    headers = None, checker_function = checker_function,
                    max_tries = 2, wait_sec = 2, meta_max_tries = 2)
    ref_out_fps = [fp for i, (code, fp) in enumerate(zip(codes, fps)) if code < 400 and (i % 2) != 0]
    assert out_fps == ref_out_fps