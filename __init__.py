from datetime import datetime
import re
import textwrap
from pathlib import Path
import fire
import os
from tqdm import tqdm
from datetime import datetime
import signal
import pdb
from nltk.corpus import stopwords

from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain


from utils.prompts import refine_prompt, summarize_prompt, summary_rules
from utils.llm import load_llm, AnswerConversationBufferMemory
from utils.file_loader import load_doc, load_embeddings
from utils.misc import embed_cache
from utils.logger import whi, yel, red
from utils.cli import ask_user

os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

class OmniQA:
    def __init__(
            self,
            model="openai",
            task="query",
            filetype=None,
            local_llm_path=None,
            #embed_model="openai",
            embed_model="distiluse-base-multilingual-cased-v1",
            #embed_model = "paraphrase-multilingual-mpnet-base-v2",
            #embed_model = "msmarco-distilbert-cos-v5",
            #embed_model = "all-mpnet-base-v2",
            stopwords_lang=None,
            saveas=".cache/latest_docs_and_embeddings",
            loadfrom=None,
            top_k=3,
            debug=False,
            llm_verbosity=True,
            help=False,
            h=False,
            **kwargs,
            ):
        """
        Parameters
        ----------
        --task str, default query
            possibilities:
                * query means to load the input files then wait for user question.
                * summary means the input will be passed through a summarization prompt.
                * summary_then_query
                * summarize_link_file takes in --filetype must be link_file

        --filetype str, default None
            the type of input. Depending on the value, different other parameters
            are needed. If json_list is used, the line of the input file can contain
            any of those parameters as long as they are as json. You can find
            an example of json_list file in utils/file_list.txt

            Supported values => relevant parameters
                * youtube => --path must be a link to youtube --language=["fr","en"] to use french transcripts if possible, english otherwise --translation=en to use the transcripts after translation to english
                * youtube_playlist => --path must link to a youtube playlist. language and translation are set to their default value of fr,en and en
                * pdf => --path is path to pdf
                * txt => --path is path to txt
                * url => --path must be a valid http(s) link
                * anki => --anki_profile is the name of the profile --anki_deck the beginning of the deckname --anki_notetype the beginning of the notetype to keep --anki_fields list of fields to keep
                * string => no other parameters needed, will ask to provide a string
                * json_list => --path is path to a txt file that contains a json for each line containing at least a filetype and a path key/value but can contain any parameters described here
                * recursive => --path is the starting path --pattern is the globbing patterns to append --exclude and --include can be a list of regex applying to found paths (include is run first then exclude, if the pattern is only lowercase it will be case insensitive) --recursed_filetype is the filetype to use for each of the found path
                * link_file => --path must point to a file where each line is a link that will be summarized. The resulting summary will be added to --out_file. Links that have already been summarized in out_file will be skipped (the out_file is never overwritten). If a line is a markdown linke like [this](link) then it will be parsed as a link. Empty lines and starting with # are ignored. If argument --out_file_logseq_mode is present, the formatting will be compatible with logseq.
                * "infer" => can often be used in the backend to try to guess the proper filetype. Experimental.

        --model str, default openai
            either gpt4all, llama, openai or fake/test/testing to use a fake answer.

        --local_llm_path str
            if model is not openai, this needs to point to a compatible model

        --embed_model str, default "openai"
            Either 'openai' or sentence_transformer embedding model to use.
            If you change this, the embedding cache will be usually
            need to be recomputed with new elements (the hash
            used to check for previous values includes the name of the model
            name)

        --stopwords_lang, str, default None
            if not None must be a list like ["french", "english"] of language
            from which to load stopwords that will be used just before
            computing embeddings. This is especially useful if you use
            embedding models like GLOVE for example.

        --saveas str, default .cache/latest_docs_and_embeddings
            only used if task is query
            save the latest 'inputs' to a file. Can be loaded again with
            --loadfrom to speed up loading time. This loads both the
            split documents and embeddings but will not update itself if the
            original files have changed.

        --loadfrom str, default None
            if not filetype argument is given, loadfrom will be set to the
            same default value as saveas
            For more, see --saveas

        --top_k int, default 3
            retrieval argument

        --debug bool, default False
            if True will open a debugger instead before crashing

        --llm_verbosity, default True
            if True, will print the intermediate reasonning steps of LLMs

        --help or -h, default False
            if True, will return this documentation.
        """
        if help:
            print(self.__init__.__doc__)
            raise SystemExit()

        # checking argument validity
        assert "loaded_docs" not in kwargs, "'loaded_docs' cannot be an argument as it is used internally"
        assert "loaded_embeddings" not in kwargs, "'loaded_embeddings' cannot be an argument as it is used internally"
        assert task in ["query", "summary", "summary_then_query", "summarize_link_file"], "invalid task value"
        if task in ["summary", "summary_then_query"]:
            assert not loadfrom, "can't use loadfrom if task is summary"
        assert (task == "summarize_link_file" and filetype == "link_file"
                ) or (task != "summarize_link_file" and filetype != "link_file"
                        ), "summarize_link_file must be used with filetype link_file"
        if task == "summarize_link_file":
            assert "path" in kwargs, 'missing path arg for summarize_link_file'
            assert "out_file" in kwargs, 'missing "out_file" arg for summarize_link_file'
            assert kwargs["out_file"] != kwargs["path"], "can't use same 'path' and 'out_file' arg"
        if filetype and loadfrom:
            filetype = None
            loadfrom = str(embed_cache.parent / "latest_docs_and_embeddings")
        assert "/" not in embed_model, "embed model can't contain slash"

        for k in kwargs:
            assert k in [
                    "anki_profile", "anki_notetype", "anki_fields", "anki_deck",
                    "path", "include", "exclude",
                    "out_file", "out_file_logseq_mode",
                    ], f"Unexpected keyword argument: '{k}'"

        if filetype == "string":
            top_k = 1
            red("Input is 'string' so setting 'top_k' to 1")

        # storing as attributes
        self.model = model
        self.task = task
        self.filetype = filetype
        self.local_llm_path = local_llm_path
        self.embed_model = embed_model
        self.saveas = saveas
        self.loadfrom = loadfrom
        self.top_k = top_k
        self.debug = debug
        self.kwargs = kwargs
        self.stopwords_lang = stopwords_lang
        self.llm_verbosity = llm_verbosity

        # loading stop words
        if self.stopwords_lang:
            try:
                stops = []
                for lang in self.stopwords_lang:
                    stops += stopwords.words(lang)
                self.stops = list(set(stops))
            except Exception as e:
                red(f"Error when extracting stop words: {e}\n\n"
                     "Setting stop words list to None.")
                self.stops = None
            self.stopw_compiled = [re.compile(r"\b" + s + r"\b") for s in self.stops]
            self.kwargs["stopwords"] = self.stopw_compiled

        if self.debug:
            # make the script interruptible
            signal.signal(signal.SIGINT, (lambda signal, frame : pdb.set_trace()))

        # compile include / exclude regex
        if "include" in self.kwargs:
            for i, inc in enumerate(self.kwargs["include"]):
                if inc == inc.lower():
                    self.kwargs["include"][i] = re.compile(inc, flags=re.IGNORECASE)
                else:
                    self.kwargs["include"][i] = re.compile(inc)
        if "exclude" in self.kwargs:
            for i, exc in enumerate(self.kwargs["exclude"]):
                if exc == exc.lower():
                    self.kwargs["exclude"][i] = re.compile(exc, flags=re.IGNORECASE)
                else:
                    self.kwargs["exclude"][i] = re.compile(exc)

        # loading llm
        self.llm, self.callback = load_llm(model, local_llm_path)

        # loading documents
        if not loadfrom:
            self.loaded_docs = load_doc(self.filetype, self.debug, **self.kwargs)
        else:
            self.loaded_docs = None  # will be loaded when embeddings are loaded

        out = self.process_task()

        whi("Done.\nOpenning debugger.")
        breakpoint()


    def process_task(self):
        red("\nProcessing task")

        if self.task == "summarize_link_file":
            d = datetime.today()
            today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
            link_list = []
            for d in self.loaded_docs:
                assert "subitem_link" in d.metadata, "missing 'subitem_link' in a doc metadata"
                if d.metadata["subitem_link"] not in link_list:
                    link_list.append(d.metadata["subitem_link"])

            total_cost = [0, 0]
            total_length_saved = 0
            for doc in tqdm(link_list, desc="Summarizing links"):
                relevant_docs = [d for d in self.loaded_docs if d.metadata["subitem_link"] == doc]
                assert relevant_docs
                with open(self.kwargs["out_file"], "r") as f:
                    content = f.read()
                    if doc in content:
                        whi(f"Skipping doc that were already summarized in out_file: '{doc}'")
                        continue

                if "title" in relevant_docs[0].metadata:
                    title = f"Here's the text title:\n'''\n{relevant_docs[0].metadata['title'].strip()}\n'''\n"
                else:
                    title = ""
                with self.callback() as cb:
                    chain = load_summarize_chain(
                            self.llm,
                            chain_type="refine",
                            return_intermediate_steps=True,
                            question_prompt=summarize_prompt.partial(title=title, rules=summary_rules),
                            refine_prompt=refine_prompt.partial(title=title, rules=summary_rules),
                            verbose=self.llm_verbosity,
                            )

                    out = chain(
                            {"input_documents": relevant_docs},
                            return_only_outputs=True,
                            )

                outtext = out["output_text"]
                outtext = outtext.replace("* ", "- ")
                outtext = outtext.replace("- - ", "- ")

                red(f"Tokens used: '{cb.total_tokens}' (${cb.total_cost})")
                total_cost[0] += cb.total_tokens
                total_cost[1] += cb.total_cost

                red(f"\n\nSummary of '{doc}':\n{outtext}")

                # parse metadata
                if title:
                    item_name = f"{relevant_docs[0].metadata['title']} - {doc}"
                else:
                    item_name = doc
                if "length" in relevant_docs[0].metadata:
                    leng = int(relevant_docs[0].metadata["length"]) / 60
                    total_length_saved += leng
                else:
                    leng = None
                if "author" in relevant_docs[0].metadata:
                    author = relevant_docs[0].metadata["author"]
                else:
                    author = None

                if "out_file_logseq_mode" in self.kwargs:
                    header = f"\n- TODO {item_name}"
                    header += "\n  collapsed:: true"
                    header += f"\n  summarization_date:: {today}"
                    header += "\n  block_type:: langchain_OnmiQA_summary"
                    header += f"\n  token_cost:: {cb.total_tokens}"
                    header += f"\n  dollar_cost:: {cb.total_cost}"
                    if leng:
                        header += f"\n  minutes_saved:: {leng:.1f}"
                    if author:
                        header += f"\n  author:: {author}"

                else:
                    header = f"\n- {item_name}    cost: {cb.total_tokens} (${cb.total_cost})"
                    if leng:
                        header += f"    {leng:.1f} minutes"
                    if author:
                        header += f"    by '{author}'"

                # save to file
                with open(self.kwargs["out_file"], "a") as f:
                    f.write(header)
                    for bulletpoint in outtext.split("\n"):
                        f.write("\n")
                        # make sure the line begins with a bullet point
                        if not bulletpoint.strip().startswith("- "):
                            begin_space = re.search(r"^(\s+)", bulletpoint)
                            if not begin_space:
                                begin_space = ""
                            bulletpoint = begin_space + "- " + bulletpoint
                        f.write(f"    {bulletpoint}")
                    f.write("\n\n\n")

                red(f"Total cost of this run: '{total_cost[0]}' (${total_cost[1]})")
                red(f"Total time saved by this run: {total_length_saved:.1f} minutes")

            if total_cost[0] != 0 and total_cost[1] != 0:
                with open(self.kwargs["out_file"], "a") as f:
                    f.write(f"- Total cost of this run: '{total_cost[0]}' (${total_cost[1]})\n")
                    f.write(f"- Total time saved by this run: plausibly {total_length_saved:.1f} minutes\n\n\n")

            whi("Done summarizing link. Exiting.")
            raise SystemExit()

        if self.task in ["summary", "summary_then_query"]:
            with self.callback() as cb:
                chain = load_summarize_chain(
                        self.llm,
                        chain_type="refine",
                        return_intermediate_steps=True,
                        question_prompt=summarize_prompt.partial(rules=summary_rules),
                        refine_prompt=refine_prompt.partial(rules=summary_rules),
                        verbose=self.llm_verbosity,
                        )
                out = chain(
                        {"input_documents": self.loaded_docs},
                        return_only_outputs=True,
                        )
            red(f"Tokens used: '{cb.total_tokens}' (${cb.total_cost})")

            red("\n\nSummary:")
            for bulletpoint in out["output_text"].split("\n"):
                red(bulletpoint)

            if self.task == "summary_then_query":
                whi("Done summarizing. Switching to query mode.")
            else:
                whi("Done summarizing. Exiting.")
                raise SystemExit()

        # load embeddings, used for querying
        self.loaded_embeddings = load_embeddings(
                self.embed_model, self.loadfrom, self.saveas, self.debug, self.loaded_docs, self.kwargs)

        assert self.task in ["query", "summary_then_query"]

        # set default ask_user argument
        multiline = False
        memory = AnswerConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True)

        while True:
            try:
                with self.callback() as cb:
                    query, self.top_k, multiline = ask_user(
                            "\n\nWhat is your question? (Q to quit)\n",
                            top_k=self.top_k,
                            multiline=multiline,
                            task=self.task,
                            )
                    # while True:
                    #     docs_and_scores = self.loaded_embeddings.similarity_search_with_score(
                    #             query,
                    #             k=self.top_k,
                    #             )
                    #     breakpoint()
                    retriever = self.loaded_embeddings.as_retriever(
                            search_kwargs={
                                "k": self.top_k,
                                "distance_metric": "cos",
                                })
                    chain = ConversationalRetrievalChain.from_llm(
                            llm=self.llm,
                            chain_type="map_reduce",
                            retriever=retriever,
                            return_source_documents=True,
                            verbose=self.llm_verbosity,
                            memory=memory,
                            )

                    ans = chain(
                            inputs={
                                "question": query,
                                },
                            return_only_outputs=False,
                            include_run_info=True,
                            )

                whi("\n\nSources:")
                for doc in ans["source_documents"]:
                    for toprint in [
                            "filetype", "path", "nid", "anki_deck", "ntags"]:
                        if toprint in doc.metadata:
                            val = doc.metadata[toprint]
                            if toprint == "ntags":
                                val = ",".join(val)
                            yel(f"    * {toprint}: {val}")
                    content = doc.page_content.strip()
                    wrapped = textwrap.wrap(content, width=120)
                    whi(f"    * content: {wrapped[0]}")
                    for w in wrapped[1:]:
                        whi(f"        {w}")
                    print("\n\n")

                red(f"Answer:\n{ans['answer']}\n")

                yel(f"Tokens used: '{cb.total_tokens}' (${cb.total_cost})")

            except Exception as err:
                whi(f"Error: '{err}'")
                if self.debug:
                    breakpoint()
                else:
                    raise


if __name__ == "__main__":
    instance = fire.Fire(OmniQA)
