import time
from tqdm import tqdm
import logging
from pathlib import Path

# adds logger, restrict it to X lines
local_dir = "/".join(__file__.split("/")[:-2])
Path(f"{local_dir}/logs.txt").touch(exist_ok=True)
Path(f"{local_dir}/logs.txt").write_text(
    "\n".join(
        Path(f"{local_dir}/logs.txt").read_text().split("\n")[-100_000:]))
logging.basicConfig(filename=f"{local_dir}/logs.txt",
                    filemode='a',
                    format=f"{time.ctime()}: %(message)s",
                    force=True,
                    )
log = logging.getLogger()
log.setLevel(logging.INFO)


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
