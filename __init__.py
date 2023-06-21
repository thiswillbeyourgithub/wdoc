import re
import textwrap
from pathlib import Path
import fire
import os
from tqdm import tqdm
from datetime import datetime
import signal
import pdb

from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain


from utils.prompts import refine_prompt, PROMPT
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
            #embed_model="distiluse-base-multilingual-cased-v1",
            #embed_model = "paraphrase-multilingual-mpnet-base-v2",
            #embed_model = "msmarco-distilbert-cos-v5",
            #embed_model = "all-mpnet-base-v2",
            saveas=".cache/latest_docs_and_embeddings",
            loadfrom=None,
            top_k=3,
            debug=False,
            **kwargs,
            ):
        """
        Parameters
        ----------
        --task str, default query
            either query or summary. query means to load the input files then wait
            for user question. summary means the input will be passed through a
            summarization prompt to get the idea.

        --filetype str, default None
            the type of input. Depending on the value, different other parameters
            are needed. If path_list is used, the line of the input file can contain
            any of those parameters as long as they are as json. You can find
            an example of path_list file in utils/file_list.txt

            Supported values => relevant parameters
                * youtube => --path must be a link to youtube --language=fr to use french transcripts --translation=en to use the transcripts after translation to english
                * pdf => --path is path to pdf
                * txt => --path is path to txt
                * anki => --anki_profile is the name of the profile --anki_deck the beginning of the deckname --anki_notetype the beginning of the notetype to keep --anki_fields list of fields to keep
                * string => no other parameters needed, will ask to provide a string
                * path_list => --path is path to a txt file that contains a json for each line containing at least a filetype and a path key/value but can contain any parameters described here
                * recursive => --path is the starting path --pattern is the globbing patterns to append --exclude and --include can be a list of regex applying to found paths (include is run first then exclude, if the pattern is only lowercase it will be case insensitive) --recursed_filetype is the filetype to use for each of the found path

        --model str, default openai
            either gpt4all, llama, openai or fake/test/testing to use a fake answer.

        --local_llm_path str
            if model is not openai, this needs to point to a compatible model

        --embed_model str, default "distiluse-base-multilingual-cased-v1"
            sentence_transformer embedding model to use. If you change this,
            the embedding cache will be populated with new elements (the hash
            used to check for previous values includes the name of the sbert model)

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
        """

        # checking argument validity
        assert "loaded_docs" not in kwargs, "'loaded_docs' cannot be an argument as it is used internally"
        assert "loaded_embeddings" not in kwargs, "'loaded_embeddings' cannot be an argument as it is used internally"
        assert task in ["query", "summary"], "invalid task value"
        if task == "summary":
            assert not loadfrom, "can't use loadfrom if task is summary"
        if filetype and loadfrom:
            filetype = None
            loadfrom = str(embed_cache.parent / "latest_docs_and_embeddings")

        for k in kwargs:
            assert k in [
                    "anki_profile", "anki_notetype", "anki_fields", "anki_deck",
                    "path", "include", "exclude",
                    ], f"Unexpected keyword argument: '{k}'"

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

        if self.task == "summary":
            with self.callback() as cb:
                chain = load_summarize_chain(
                        self.llm,
                        chain_type="refine",
                        return_intermediate_steps=True,
                        question_prompt=PROMPT,
                        refine_prompt=refine_prompt,
                        verbose=True,
                        )
                out = chain(
                        {"input_documents": self.loaded_docs},
                        return_only_outputs=True,
                        )
            red(f"Tokens used: '{cb.total_tokens}' (${cb.total_cost})")

            red("\n\nSummary:")
            for bulletpoint in out["output_text"].split("\n"):
                red(bulletpoint)

            whi("Switching to query mode.")
            self.task = "query"

        # load embeddings, either for query or to query on what was just zummaried
        self.loaded_embeddings = load_embeddings(
                self.embed_model, self.loadfrom, self.saveas, self.debug, self.loaded_docs)

        assert self.task == "query"

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
                            verbose=True,
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
