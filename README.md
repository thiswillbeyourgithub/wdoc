## How to install
    * `git clone`
    * `python -m pip install -r requirements.txt`
    * some package used to parse input files will certainly not be installed. Use pip install and pay attention to the error message
        * for youtube: `python -m pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"`
        * for urls: `python -m pip install goose3`
    * `python __init__.py --help`

## supported filetype:
    * youtube videos
    * youtube playlist
    * pdf (local or via url remote)
    * txt file, markdown etc
    * anki
    * string (just paste your text into the app)
    * json_list (you give as argument a path to a file where each line is a json_list that contains the loader arguments. This can be used for example to load several files in a row)
    * recursive (you give a path and a regex pattern and a filetype, it finds all the files)
    * link_file (you give a text file where each line is a url, proper filetype for each url will be infered)
    * infer (will try to guess)

## supported tasks:
    * query: give documents and asks questions about it.
    * summary: give documents and read a summary. The summary prompt can be found in `utils/prompts.py`.
    * summary_then_query: summarize the document then allow you to query directly about it.
    * summarize_link_file: with filetype==link_file this summarizes all the links and adds it to an output file. (logseq format is supported)
