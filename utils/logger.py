import requests
import time
from tqdm import tqdm
import logging
import logging.handlers
from pathlib import Path

# adds logger, restrict it to X lines
local_dir = "/".join(__file__.split("/")[:-2])
Path(f"{local_dir}/logs.txt").touch(exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
handler = logging.handlers.RotatingFileHandler(
        filename=f"{local_dir}/logs.txt",
        mode="a",
        encoding=None,
        delay=0,
        maxBytes=1024*1024*100,  # max 50mb
        backupCount=3,
        )
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(handler)
# delete any additional log file
Path(f"{local_dir}/logs.txt.3").unlink(missing_ok=True)


def coloured_log(color_asked):
    """used to print color coded logs"""
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs

    if color_asked == "white":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.error(string)
            tqdm.write(col_rst + string + col_rst, **args)
    elif color_asked == "yellow":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.error(string)
            tqdm.write(col_yel + string + col_rst, **args)
    elif color_asked == "red":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.error(string)
            tqdm.write(col_red + string + col_rst, **args)
    return printer


whi = coloured_log("white")
yel = coloured_log("yellow")
red = coloured_log("red")

# phone notification
def create_ntfy_func(url):
    def ntfy_func(text):
        red(text)
        requests.post(
                url=url,
                headers={"Title": "DocTools Summary"},
                data=text.encode("utf-8"),
                )
        return text
    return ntfy_func
