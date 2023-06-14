"""Importing `log` from this module starts a logging handler, which can be logged to 
with `log.info("log this info")` etc.
"""
from python_log_indenter import IndentedLoggerAdapter
import logging
import os
from tqdm.contrib.logging import logging_redirect_tqdm
import warnings

log_settings = logging.getLogger(__name__)
__formatter__ = logging.Formatter(fmt='%(message)s', datefmt = '%Y/%m/%d %H:%M:%S')
__handler__ = logging.StreamHandler()
__handler__.setFormatter(__formatter__)
__handler__.setLevel("INFO")
log_settings.addHandler(__handler__)
log = IndentedLoggerAdapter(log_settings)

def adjust_logger(log_write, folder, log_level, testing = False):
    """Function to adjust the default logging settings that get initiated by
    importing this module.

    Parameters
    ----------
    log_write : bool
        Stop or start writing to log.
    folder : str
        Path to folder in which to store `"log.txt"`.
    log_level : str
        Set the log level.
    """
    if log_write and any([isinstance(x, logging.FileHandler) for x in log_settings.handlers]):
        handlers_to_remove = [x for x in log_settings.handlers if isinstance(x, logging.FileHandler)]
        for handler in handlers_to_remove:
            log_settings.removeHandler(handler)

    if log_write and not any([isinstance(x, logging.FileHandler) for x in log_settings.handlers]):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        handler = logging.FileHandler(filename = os.path.join(folder, "log.txt"))
        formatter = logging.Formatter('{asctime} {levelname:>9s}: {message}', style='{')
        handler.setFormatter(formatter)
        handler.setLevel("DEBUG")
        log_settings.addHandler(handler)
        logging_redirect_tqdm(loggers = log_settings.handlers)
        log_settings.propagate = False

    if testing:
        log_settings.propagate = True
        def fancy_warning(message):
            IndentedLoggerAdapter(log_settings).warning(message)
            warnings.warn(message, stacklevel=2)
        log.warning = fancy_warning

    log_settings.setLevel(log_level)