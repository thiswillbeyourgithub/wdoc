from pathlib import Path
import time
from datetime import datetime
import re
import textwrap
import fire
import os
from tqdm import tqdm
import signal
import pdb
from nltk.corpus import stopwords

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from utils.llm import load_llm, AnswerConversationBufferMemory
from utils.file_loader import load_doc, load_embeddings, get_tkn_length, average_word_length, wpm
from utils.misc import embed_cache
from utils.logger import whi, yel, red
from utils.cli import ask_user
from utils.tasks import do_summarize

os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

class DocToolsLLM:
    VERSION = 0.3
    def __init__(
            self,
            model="openai",
            task="query",
            filetype=None,
            local_llm_path=None,
            embed_model="openai",
            # embed_model = "distiluse-base-multilingual-cased-v1",
            # embed_model = "paraphrase-multilingual-mpnet-base-v2",
            # embed_model = "msmarco-distilbert-cos-v5",
            # embed_model = "all-mpnet-base-v2",
            stopwords_lang=None,
            saveas=".cache/latest_docs_and_embeddings",
            loadfrom=None,

            top_k=3,
            n_to_combine=1,
            n_summpasscheck=3,

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
            an example of json_list file in utils/json_list_example.txt

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

        --n_to_combine int, default 1
            when creating the summary of a long document split into
            chunks, this value is the chunk index number at which
            the summarization check will be called.

            A value of 1 means that when the summarizer is done
            processing the 2nd (reminder that index start at 0)
            chunk of text the summaries of
            chunk 0 and chunk 1 (up to n_to_combine) will be 
            concatenated, then will be passed n_summpasscheck times into
            the llm that is prompted with reformulating and compacting the
            summary.

            As the summary of the last chunk is shown to the llm as example
            when summarizing its own chunk this has the effect of increasing
            summary compactness and quality.

            If you increase n_to_combine to more than 1, you will have
            a considerably shorter and to the point summary for the whole
            document. Same idea for setting n_summpasscheck too high.

        --n_summpasscheck int, default 3
            see --n_to_combine

n_to_combine = 1  # careful, indices start at 0. So setting n_to_combine at 3 means that the first 4 paragraphs will get combined into one. This will certainly lose meaningful information.
n_passcheck = 3  # number of check to do

        --debug bool, default False
            if True will open a debugger instead before crashing, also use
            sequential processing instead of multithreading and enable
            langchain tracing.

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
        self.n_summpasscheck = n_summpasscheck
        self.n_to_combine = n_to_combine

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
            os.environ["LANGCHAIN_TRACING"] = "true"

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
            self.loaded_docs = load_doc(
                    filetype=self.filetype,
                    debug=self.debug,
                    task=self.task,
                    **self.kwargs)
        else:
            self.loaded_docs = None  # will be loaded when embeddings are loaded

        _ = self.process_task()

        whi("Done with tasks.")
        if self.debug:
            breakpoint()


    def process_task(self):
        red("\nProcessing task")

        if self.task in ["summarize_link_file", "summary", "summary_then_query"]:
            total_tkn_cost = 0
            total_dol_cost = 0
            total_docs_length = 0
            total_summary_length = 0
            links_todo = set()
            already_done = set()
            failed = []

            # get the list of documents from the same source. Also checks if
            # it's not part of the output file if task is "summarize_link_file"
            if self.task == "summarize_link_file":
                if not Path(self.kwargs["out_file"]).exists():
                    Path(self.kwargs["out_file"]).touch()
                with open(self.kwargs["out_file"], "r") as f:
                    output_content = f.read()
                for d in self.loaded_docs:
                    assert "subitem_link" in d.metadata, "missing 'subitem_link' in a doc metadata"

                    link = d.metadata["subitem_link"]
                    if link in already_done or link in links_todo:
                        continue
                    if link in output_content:
                        whi(f"Skipping link : already summarized in out_file: '{link}'")
                        already_done.add(link)
                        continue
                    links_todo.add(link)
            else:
                for d in self.loaded_docs:
                    links_todo.add(d.metadata["path"])
                assert len(links_todo) == 1, f"Invalid length of links_todo for this task: '{len(links_todo)}'"

            # estimate price before summarizing, in case you put the bible in there
            full_tkn = sum([get_tkn_length(doc.page_content) for doc in self.loaded_docs if doc.metadata["subitem_link"] in links_todo])
            red(f"Total number of tokens in documments to summarize: '{full_tkn}'")
            # a conservative estimate is that it takes 2 times the number
            # of tokens of a document to summarize it
            estimate_tkn = 2.4 * full_tkn  # empirical value: 2.37 times the doc tokens
            estimate_dol = estimate_tkn / 1000 * 0.0016  # empirical value: $0.001579 for 1k tokens
            red(f"Conservative estimate of the cost to summarize: ${estimate_dol:.4f} for {estimate_tkn} tokens.")
            if estimate_dol > 1:
                raise Exception(red("Cost estimate > $1 which is absurdly high. Has something gone wrong? Quitting."))

            if self.model == "openai":
                # increase likelyhood that chatgpt will use indentation by
                # biasing towards adding space.
                self.llm.model_kwargs["logit_bias"] = {
                        220: 5,  # ' '
                        532: 5,  # ' -'
                        9: 5,  # '*'
                        1635: 5,  # ' *'
                        }

            for link in tqdm(links_todo, desc="Summarizing documents", disable=(not len(links_todo) - 1)):

                if self.task == "summarize_link_file":
                    # get only the docs that match the link
                    relevant_docs = [d for d in self.loaded_docs if d.metadata["subitem_link"] == link]
                else:
                    relevant_docs = self.loaded_docs
                assert relevant_docs, 'Empty relevant_docs!'

                # parse metadata from the doc
                metadata = []
                if "title" in relevant_docs[0].metadata:
                    item_name = f"{relevant_docs[0].metadata['title'].strip()} - {link}"
                    metadata.append(f"Title: '{item_name}'")
                else:
                    item_name = link
                if "docs_reading_time" in relevant_docs[0].metadata:
                    leng = relevant_docs[0].metadata["docs_reading_time"]
                    total_docs_length += leng
                    metadata.append(f"Duration: {leng:.1f} minutes")
                else:
                    leng = None
                if "author" in relevant_docs[0].metadata:
                    author = relevant_docs[0].metadata["author"].strip()
                    metadata.append(f"Author: '{author}'")
                else:
                    author = None

                if metadata:
                    metadata = "Here's additional information about the text:\n'''" + "\n".join(metadata)
                    metadata += "\nArticle chunk number: [PROGRESS]"
                    metadata += "\n'''\n"
                else:
                    metadata = ""

                # summarize each chunk of the link and return one text
                summary, doc_total_tokens, doc_total_cost = do_summarize(
                        docs=relevant_docs,
                        n_to_combine=self.n_to_combine,
                        n_summpasscheck=self.n_summpasscheck,
                        metadata=metadata,
                        model=self.model,
                        llm=self.llm,
                        callback=self.callback,
                        verbose=self.llm_verbosity,
                        )

                # get reading length of the summary
                reading_length = len(summary) / average_word_length / wpm
                total_summary_length += reading_length

                total_tkn_cost += doc_total_tokens
                total_dol_cost += doc_total_cost

                red(f"Tokens used for this doc: '{doc_total_tokens}' (${doc_total_cost:.5f})")

                # make sure to use the same markdown formatting
                summary = summary.replace("* ", "- ")
                summary = summary.replace("- - ", "- ")

                red(f"\n\nSummary of '{link}':\n{summary}")

                red(f"Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})")
                red(f"Total time saved by this run: {total_docs_length:.1f} minutes")


                if "out_file" in self.kwargs:
                    if "out_file_logseq_mode" in self.kwargs:
                        header = f"\n- TODO {item_name}"
                        header += "\n  collapsed:: true"
                        header += f"\n  DocToolsLLM_version:: {self.VERSION}"
                        header += f"\n  DocToolsLLM_model:: {self.model}"
                        header += "\n  block_type:: DocToolsLLM_summary"
                        header += f"\n  summarization_date:: {today}"
                        header += f"\n  summarization_timestamp:: {int(time.time())}"
                        header += f"\n  token_cost:: {doc_total_tokens}"
                        header += f"\n  dollar_cost:: {doc_total_cost:.5f}"
                        header += f"\n  summary_reading_length:: {reading_length}"
                        if leng:
                            header += f"\n  doc_reading_length:: {leng}"
                        if author:
                            header += f"\n  author:: {author}"

                    else:
                        header = f"\n- {item_name}    cost: {doc_total_tokens} (${doc_total_cost:.5f})"
                        if leng:
                            header += f"    {leng:.1f} minutes"
                        if author:
                            header += f"    by '{author}'"
                        header += f"    DocToolsLLM version {self.VERSION} with model {self.model}"

                    # save to output file
                    with open(self.kwargs["out_file"], "a") as f:
                        f.write(header)
                        for bulletpoint in summary.split("\n"):
                            f.write("\n")
                            # make sure the line begins with a bullet point
                            if not bulletpoint.strip().startswith("- "):
                                begin_space = re.search(r"^(\s+)", bulletpoint)
                                if not begin_space:
                                    begin_space = [""]
                                bulletpoint = begin_space[0] + "- " + bulletpoint
                            f.write(f"    {bulletpoint}")
                        f.write("\n\n\n")


            if "out_file" in self.kwargs:
                # after summarizing all links, append to output file the total cost
                if total_tkn_cost != 0 and total_dol_cost != 0:
                    with open(self.kwargs["out_file"], "a") as f:
                        f.write(f"- Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})\n")
                        f.write(f"- Total time saved by this run: {total_docs_length - total_summary_length:.1f} minutes\n\n\n")

            # and write to input file a summary too
            if "out_file" in self.kwargs:
                try:
                    with open(self.kwargs["path"], "a") as f:
                        f.write(f"\n\n")
                        f.write(f"- Done with summaries of {today}\n")
                        f.write(f"    - Number of links summarized: {len(links_todo) - len(failed)}/{len(links_todo) + len(already_done)}\n")
                        if failed:
                            f.write(f"    - Number of links failed: {len(failed)}:\n")
                            for f in failed:
                                f.write(f"        - {f}\n")
                        f.write(f"    - Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})\n")
                        f.write(f"    - Total time saved by this run: plausibly {total_docs_length:.1f} minutes\n")
                except Exception as err:
                    red(f"Exception when writing end of run details to input file: '{err}'")

            if self.task == "summary_then_query":
                whi("Done summarizing. Switching to query mode.")
                if self.model == "openai":
                    del self.llm.model_kwargs["logit_bias"]
            else:
                whi("Done summarizing. Exiting.")
                raise SystemExit()

        # load embeddings for querying
        self.loaded_embeddings = load_embeddings(
                self.embed_model, self.loadfrom, self.saveas, self.debug, self.loaded_docs, self.kwargs)

        assert self.task in ["query", "summary_then_query"]

        # set default ask_user argument
        multiline = False

        # conversational memory
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
                            return_generated_question=True,
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

                    # docs = self.loaded_embeddings.similarity_search(
                    #         query,
                    #         k=self.top_k,
                    #         )
                    # chain = load_qa_with_sources_chain(
                    #         llm=self.llm,
                    #         #retriever=retriever,
                    #         chain_type="map_reduce",
                    #         #prompt=query_prompt,
                    #         combine_prompt=combine_prompt,
                    #         #return_map_steps=True,
                    #         #return_source_documents=True,
                    #         verbose=self.llm_verbosity,
                    #         input_key="question",
                    #         )

                    # ans = chain(
                    #         inputs={
                    #             "question": query,
                    #             "input_documents": docs,
                    #             },
                    #         return_only_outputs=False,
                    #         include_run_info=True,
                    #         )

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

                yel(f"Tokens used: '{cb.total_tokens}' (${cb.total_cost:.5f})")

            except Exception as err:
                whi(f"Error: '{err}'")
                raise


if __name__ == "__main__":
    instance = fire.Fire(DocToolsLLM)
