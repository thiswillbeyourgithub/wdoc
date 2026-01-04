#!/usr/bin/env python3
"""
Gradio web UI for wdoc document processing.

This provides a browser-based interface to wdoc's query, summarize, and parse
capabilities. Designed to run in a Docker container with volume-mounted vectorstore.

This file was created with assistance from aider.chat.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Any, get_args
import tempfile
import sys

import gradio as gr
from loguru import logger

# Set environment variable to indicate running in Docker
os.environ["WDOC_IN_DOCKER"] = "true"

from wdoc.wdoc import wdoc
from wdoc.utils.misc import filetype_arg_types
from wdoc.utils.env import EnvDataclass


def list_vectorstore_dirs(vectorstore_path: str = "/app/vectorstore") -> list:
    """
    List all directories in the vectorstore path.

    Parameters
    ----------
    vectorstore_path : str
        Path to the vectorstore directory

    Returns
    -------
    list
        List of directory names in the vectorstore path
    """
    try:
        if not os.path.exists(vectorstore_path):
            logger.warning(f"Vectorstore path not found: {vectorstore_path}")
            return []

        # Get all directories in the vectorstore path
        dirs = [
            d
            for d in os.listdir(vectorstore_path)
            if os.path.isdir(os.path.join(vectorstore_path, d))
        ]
        return sorted(dirs)
    except Exception as e:
        logger.error(f"Error listing vectorstore directories: {str(e)}")
        return []


def read_log_file(
    log_path: str = "/home/wdoc/.local/state/wdoc/log/logs.txt", max_lines: int = 5000
) -> str:
    """
    Read the last N lines from wdoc's log file.

    Parameters
    ----------
    log_path : str
        Path to the log file
    max_lines : int
        Maximum number of lines to read from the end of the file

    Returns
    -------
    str
        The last max_lines lines from the log file
    """
    try:
        if not os.path.exists(log_path):
            return f"Log file not found at {log_path}"

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            # Get last max_lines lines
            return "".join(lines[-max_lines:])
    except Exception as e:
        return f"Error reading log file: {str(e)}"


def process_document(
    task: str,
    path_text: str,
    uploaded_file: Optional[Any],
    query_text: str,
    model: str,
    filetype: str,
    llm_verbosity: bool,
    disable_llm_cache: bool,
    dollar_limit: float,
    embed_model: str,
    top_k: str,
    query_retrievers: str,
    query_eval_model: str,
    query_eval_check_number: int,
    summary_n_recursion: int,
    load_embeds_from: str,
    save_embeds_as: str,
    env_args: dict,
    filetype_args: dict,
) -> Tuple[str, str]:
    """
    Process a document using wdoc based on selected task.

    Parameters
    ----------
    task : str
        The task to perform: 'query', 'summarize', or 'parse'
    path_text : str
        Text input for path (URL or file path)
    uploaded_file : Optional[Any]
        Gradio file upload object
    query_text : str
        Query string for query task
    model : str
        Model name to use
    filetype : str
        File type hint for wdoc

    Returns
    -------
    Tuple[str, str]
        (markdown_output, download_content) - the rendered output and downloadable text
    """
    try:
        # Set environment variables from env_args
        for key, value in env_args.items():
            if value is not None and value != "":
                # Convert value to string for environment variable
                os.environ[key] = str(value)
                logger.debug(f"Set environment variable {key}={value}")

        # Determine path: uploaded file takes precedence over text input
        if uploaded_file is not None:
            path = uploaded_file.name
            logger.info(f"Processing uploaded file: {path}")
        elif path_text and path_text.strip():
            path = path_text.strip()
            logger.info(f"Processing path from text: {path}")
        else:
            return (
                "❌ **Error**: Please provide either a file upload or a path/URL.",
                "",
            )

        # Get vectorstore path from environment for query tasks
        vectorstore_path = os.getenv("WDOC_VECTORSTORE_PATH", "/app/vectorstore")

        # Filter out empty/None filetype arguments
        filtered_filetype_args = {
            k: v
            for k, v in filetype_args.items()
            if v is not None and v != "" and v != []
        }

        # Create wdoc instance based on task
        if task == "query":
            if not query_text or not query_text.strip():
                return "❌ **Error**: Query text is required for query task.", ""

            logger.info(f"Starting query task with query: {query_text}")

            # Process load_embeds_from - convert "None" or empty to None, otherwise use full path
            load_embeds = None
            if load_embeds_from and load_embeds_from != "None":
                load_embeds = os.path.join(vectorstore_path, load_embeds_from)

            # Process save_embeds_as - use default if empty
            save_embeds = (
                save_embeds_as.strip()
                if save_embeds_as.strip()
                else "{user_cache}/latest_docs_and_embeddings"
            )

            instance = wdoc(
                task="query",
                path=path,
                filetype=filetype,
                model=model,
                embed_model=embed_model,
                top_k=top_k,
                query_retrievers=query_retrievers,
                query_eval_model=query_eval_model,
                query_eval_check_number=int(query_eval_check_number),
                load_embeds_from=load_embeds,
                save_embeds_as=save_embeds,
                llm_verbosity=llm_verbosity,
                disable_llm_cache=disable_llm_cache,
                dollar_limit=dollar_limit,
                **filtered_filetype_args,
            )
            result = instance.query_task(query=query_text.strip())
            output_md = f"# Query Result\n\n**Query**: {query_text}\n\n## Answer\n\n{result['final_answer']}"

            # Add metadata if available
            if "sources" in result:
                output_md += f"\n\n## Sources\n\n{result['sources']}"
            if "total_cost" in result:
                output_md += f"\n\n---\n*Total cost: ${result['total_cost']:.4f}*"

        elif task == "summarize":
            logger.info("Starting summarize task")
            instance = wdoc(
                task="summarize",
                path=path,
                filetype=filetype,
                model=model,
                summary_n_recursion=int(summary_n_recursion),
                llm_verbosity=llm_verbosity,
                disable_llm_cache=disable_llm_cache,
                dollar_limit=dollar_limit,
                **filtered_filetype_args,
            )
            result = instance.summary_task()
            output_md = f"# Summary\n\n{result['summary']}"

            # Add metadata if available
            if "doc_total_cost" in result:
                output_md += f"\n\n---\n*Total cost: ${result['doc_total_cost']:.4f}*"

        elif task == "parse":
            logger.info("Starting parse task with text format")
            result = wdoc.parse_doc(
                path=path,
                filetype=filetype,
                format="text",
                **filtered_filetype_args,
            )

            # The text is parsed as markdown and rendered directly
            output_md = f"# Parsed Document\n\n{result}"
        else:
            return f"❌ **Error**: Unknown task '{task}'", ""

        logger.info(f"Task completed successfully: {task}")

        return output_md, output_md

    except Exception as e:
        logger.exception(f"Error processing document: {e}")
        error_md = (
            f"# ❌ Error\n\nAn error occurred while processing:\n\n```\n{str(e)}\n```"
        )

        return error_md, error_md


def create_interface() -> gr.Blocks:
    """
    Create the Gradio Blocks interface for wdoc.

    Returns
    -------
    gr.Blocks
        The configured Gradio interface
    """
    with gr.Blocks(
        title="wdoc Web UI",
        theme=gr.themes.Soft(),
        css="""
        .log-output textarea {
            font-family: 'Courier New', monospace !important;
            font-size: 11px !important;
            white-space: pre !important;
            overflow-wrap: normal !important;
            word-break: keep-all !important;
        }
        """,
    ) as interface:
        gr.Markdown(
            "# 📚 wdoc Web Interface (Experimental)\n\nProcess documents with AI-powered query, summarization, and parsing."
        )

        with gr.Tabs() as main_tabs:
            with gr.Tab("Input", id=0):
                # Task selection
                task = gr.Radio(
                    choices=["query", "summarize", "parse"],
                    value="summarize",
                    label="Task",
                    info="Select the operation to perform on the document",
                )

                # Path input options with tabs
                gr.Markdown("### Document Input")
                with gr.Tabs() as input_tabs:
                    with gr.Tab("Path or URL", id="text_tab"):
                        path_text = gr.Textbox(
                            label="Path or URL",
                            placeholder="https://example.com/document.pdf or /path/to/file.txt",
                            lines=1,
                        )
                    with gr.Tab("Upload File", id="file_tab"):
                        uploaded_file = gr.File(
                            label="Upload File",
                            file_types=[".txt", ".pdf", ".docx", ".html", ".md"],
                        )

                # Task-specific inputs
                with gr.Group(visible=False) as query_group:
                    query_text = gr.Textbox(
                        label="Query",
                        placeholder="What question do you want to ask about the document?",
                        lines=3,
                    )

                # Parse task has no additional options - always uses text format
                parse_group = gr.Group(visible=False)

                # Model and filetype settings
                with gr.Accordion("Advanced Settings", open=False):
                    model = gr.Textbox(
                        label="Model",
                        value=os.getenv("WDOC_DEFAULT_MODEL", "openai/gpt-4o-mini"),
                        placeholder="e.g., openai/gpt-4o-mini",
                    )
                    filetype = gr.Dropdown(
                        choices=[
                            "auto",
                            "anki",
                            "epub",
                            "html",
                            "json_dict",
                            "local_audio",
                            "local_html",
                            "local_video",
                            "logseq_markdown",
                            "online_media",
                            "online_pdf",
                            "pdf",
                            "powerpoint",
                            "string",
                            "text",
                            "txt",
                            "url",
                            "word",
                            "youtube",
                        ],
                        value="auto",
                        label="File Type",
                        info="Auto-detect or specify the file type",
                    )

                    # Query/Search-specific settings
                    with gr.Group(visible=False) as query_settings_group:
                        gr.Markdown("### Query/Search Settings")
                        embed_model = gr.Textbox(
                            label="Embedding Model",
                            value=os.getenv(
                                "WDOC_DEFAULT_EMBED_MODEL",
                                "openai/text-embedding-3-small",
                            ),
                            placeholder="e.g., openai/text-embedding-3-small",
                            info="Model to use for document embeddings",
                        )
                        top_k = gr.Textbox(
                            label="Top K",
                            value="auto_200_500",
                            placeholder="e.g., auto_200_500 or 100",
                            info="Number of documents to retrieve (auto_N_M or integer)",
                        )
                        query_retrievers = gr.Textbox(
                            label="Query Retrievers",
                            value="basic_multiquery",
                            placeholder="e.g., basic_multiquery",
                            info="Retriever strategy (basic, multiquery, knn, svm, parent)",
                        )
                        query_eval_model = gr.Textbox(
                            label="Query Eval Model",
                            value=os.getenv(
                                "WDOC_DEFAULT_QUERY_EVAL_MODEL", "openai/gpt-4o-mini"
                            ),
                            placeholder="e.g., openai/gpt-4o-mini",
                            info="Model to use for evaluating document relevance",
                        )
                        query_eval_check_number = gr.Number(
                            label="Query Eval Check Number",
                            value=3,
                            minimum=1,
                            precision=0,
                            info="Number of evaluation checks per document",
                        )

                        gr.Markdown("### Vectorstore Settings")
                        with gr.Row():
                            load_embeds_from = gr.Dropdown(
                                label="Load Embeddings From",
                                choices=["None"] + list_vectorstore_dirs(),
                                value="None",
                                info="Select existing vectorstore to load, or None to create new",
                                allow_custom_value=False,
                            )
                            refresh_vectorstore_btn = gr.Button(
                                "🔄 Refresh",
                                size="sm",
                                scale=0,
                            )
                        save_embeds_as = gr.Textbox(
                            label="Save Embeddings As",
                            value="{user_cache}/latest_docs_and_embeddings",
                            placeholder="{user_cache}/latest_docs_and_embeddings",
                            info="Path to save embeddings (use {user_cache} for default cache location)",
                        )

                    # Summary-specific settings
                    with gr.Group(visible=False) as summary_settings_group:
                        gr.Markdown("### Summary Settings")
                        summary_n_recursion = gr.Number(
                            label="Summary Recursion Depth",
                            value=0,
                            minimum=0,
                            precision=0,
                            info="Number of recursive summarization steps",
                        )

                    gr.Markdown("### Misc")
                    llm_verbosity = gr.Checkbox(
                        label="LLM Verbosity",
                        value=False,
                        info="Enable verbose logging for LLM calls",
                    )
                    disable_llm_cache = gr.Checkbox(
                        label="Disable LLM Cache",
                        value=False,
                        info="Disable caching of LLM responses",
                    )
                    dollar_limit = gr.Number(
                        label="Dollar Limit",
                        value=5,
                        minimum=0,
                        info="Maximum allowed cost in dollars",
                    )

                    # Environment variables accordion
                    with gr.Accordion("Environment Variables", open=False):
                        gr.Markdown("### Environment Variable Settings")
                        gr.Markdown(
                            "*These control wdoc's behavior. Changes will be applied for this request only.*"
                        )

                        env_arg_components = {}

                        # Create input components for EnvDataclass fields
                        for (
                            field_name,
                            field_obj,
                        ) in EnvDataclass.__dataclass_fields__.items():
                            # Skip internal fields and dummy variables
                            if (
                                field_name.startswith("_")
                                or field_name == "WDOC_DUMMY_ENV_VAR"
                            ):
                                continue

                            # Convert field_name to a more readable label
                            label = field_name.replace("_", " ").title()
                            arg_type = field_obj.type
                            default_value = field_obj.default

                            # Determine the type and create appropriate component
                            type_str = str(arg_type)

                            # Special handling for WDOC_STRICT_DOCDICT (Union[bool, Literal["strip"]])
                            if field_name == "WDOC_STRICT_DOCDICT":
                                env_arg_components[field_name] = gr.Radio(
                                    choices=["false", "strip"],
                                    label=label,
                                    value="strip"
                                    if default_value == "strip"
                                    else "false",
                                    info="Choose strictness mode for DocDict validation",
                                )
                            # Use radio buttons for specific Literal type fields
                            elif field_name in [
                                "WDOC_TYPECHECKING",
                                "WDOC_BEHAVIOR_EXCL_INCL_USELESS",
                                "WDOC_IMPORT_TYPE",
                                "WDOC_AUDIO_BACKEND",
                                "WDOC_YOUTUBE_AUDIO_BACKEND",
                                "WDOC_LOADING_FAILURE",
                                "WDOC_DDG_SAFE_SEARCH",
                            ]:
                                # Extract choices from Literal type
                                choices = list(get_args(arg_type))
                                env_arg_components[field_name] = gr.Radio(
                                    choices=choices,
                                    label=label,
                                    value=default_value,
                                    info=f"Choose from: {', '.join(str(c) for c in choices)}",
                                )
                            elif "Literal" in type_str:
                                # Extract choices from Literal type - use dropdown for others
                                choices = list(get_args(arg_type))
                                env_arg_components[field_name] = gr.Radio(
                                    choices=choices,
                                    label=label,
                                    value=default_value,
                                    info=f"Choose from: {', '.join(str(c) for c in choices)}",
                                )
                            elif arg_type == bool or "bool" in type_str:
                                env_arg_components[field_name] = gr.Checkbox(
                                    label=label,
                                    value=default_value
                                    if isinstance(default_value, bool)
                                    else False,
                                )
                            elif arg_type == int or "int" in type_str:
                                env_arg_components[field_name] = gr.Number(
                                    label=label,
                                    value=default_value
                                    if isinstance(default_value, (int, float))
                                    else 0,
                                    precision=0,
                                )
                            elif arg_type == float or "float" in type_str:
                                env_arg_components[field_name] = gr.Number(
                                    label=label,
                                    value=default_value
                                    if isinstance(default_value, (int, float))
                                    else 0.0,
                                )
                            elif "Optional[str]" in type_str:
                                env_arg_components[field_name] = gr.Textbox(
                                    label=label,
                                    value=default_value
                                    if isinstance(default_value, str)
                                    else "",
                                    info="Leave empty for None",
                                )
                            else:  # Default to textbox for str and other types
                                env_arg_components[field_name] = gr.Textbox(
                                    label=label,
                                    value=default_value
                                    if isinstance(default_value, str)
                                    else "",
                                )

                    # Filetype-specific arguments accordion
                    with gr.Accordion("Filetype Arguments", open=False):
                        gr.Markdown("### Filetype-Specific Settings")
                        gr.Markdown(
                            "*These arguments are specific to certain filetypes. "
                            "Only fill in what's relevant for your document type.*"
                        )

                        filetype_arg_components = {}

                        # Create input components based on the type from filetype_arg_types
                        for arg_name, arg_type in filetype_arg_types.items():
                            # Convert arg_name to a more readable label
                            label = arg_name.replace("_", " ").title()

                            # Determine the type and create appropriate component
                            type_str = str(arg_type)

                            if "Literal" in type_str:
                                # Extract choices from Literal type
                                choices = list(get_args(arg_type))
                                filetype_arg_components[arg_name] = gr.Radio(
                                    choices=choices,
                                    label=label,
                                    value=None,
                                    info=f"Choose from: {', '.join(choices)}",
                                )
                            elif arg_type == bool:
                                filetype_arg_components[arg_name] = gr.Checkbox(
                                    label=label,
                                    value=False,
                                )
                            elif arg_type == int:
                                filetype_arg_components[arg_name] = gr.Number(
                                    label=label,
                                    value=None,
                                    precision=0,
                                )
                            elif arg_type == float:
                                filetype_arg_components[arg_name] = gr.Number(
                                    label=label,
                                    value=None,
                                )
                            elif arg_type == dict or "dict" in type_str:
                                filetype_arg_components[arg_name] = gr.Textbox(
                                    label=label,
                                    value=None,
                                    placeholder='{"key": "value"}',
                                    info="Enter as JSON",
                                )
                            elif "List" in type_str or arg_type == list:
                                filetype_arg_components[arg_name] = gr.Textbox(
                                    label=label,
                                    value=None,
                                    placeholder="item1, item2, item3",
                                    info="Comma-separated values",
                                )
                            elif "Union[str, List[str]]" in type_str:
                                filetype_arg_components[arg_name] = gr.Textbox(
                                    label=label,
                                    value=None,
                                    placeholder="value or item1, item2",
                                    info="Single value or comma-separated list",
                                )
                            else:  # Default to textbox for str and other types
                                filetype_arg_components[arg_name] = gr.Textbox(
                                    label=label,
                                    value=None,
                                )

                # Process button - prominently placed outside the accordion
                process_btn = gr.Button(
                    "🚀 Process Document", variant="primary", size="lg"
                )

                # Logs section - collapsed by default, continuously updated
                with gr.Accordion("📋 Logs", open=False):
                    logs_output = gr.Textbox(
                        label="Application Logs (last 5000 lines)",
                        lines=10,
                        max_lines=20,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes="log-output",
                    )
                    # Timer to update logs every 0.2 seconds
                    log_timer = gr.Timer(value=0.2, active=True)

            with gr.Tab("Output", id=1):
                # Output display
                output_md = gr.Markdown(label="Output")

                # Download and copy buttons
                with gr.Row():
                    download_btn = gr.DownloadButton(
                        label="📥 Download as Markdown",
                        variant="secondary",
                    )
                    # Note: Copy to clipboard requires JavaScript and is handled via download for simplicity

                # Hidden textbox to store the output for download
                output_text = gr.Textbox(visible=False)

        # Event handlers
        def update_visibility(
            task_name: str,
        ) -> Tuple[gr.Group, gr.Group, gr.Group, gr.Group]:
            """Update visibility of task-specific input groups based on selected task."""
            # Query text input only for query task
            query_visible = task_name == "query"
            # Parse group (currently empty but kept for future use)
            parse_visible = task_name == "parse"
            # Query/search settings visible for query and search tasks
            query_settings_visible = task_name in ["query", "search"]
            # Summary settings only for summarize task
            summary_settings_visible = task_name == "summarize"

            return (
                gr.Group(visible=query_visible),  # query_group
                gr.Group(visible=parse_visible),  # parse_group
                gr.Group(visible=query_settings_visible),  # query_settings_group
                gr.Group(visible=summary_settings_visible),  # summary_settings_group
            )

        task.change(
            fn=update_visibility,
            inputs=[task],
            outputs=[
                query_group,
                parse_group,
                query_settings_group,
                summary_settings_group,
            ],
        )

        # Refresh vectorstore dropdown
        refresh_vectorstore_btn.click(
            fn=lambda: gr.Dropdown(choices=["None"] + list_vectorstore_dirs()),
            inputs=None,
            outputs=[load_embeds_from],
        )

        # Timer event to continuously update logs
        log_timer.tick(
            fn=read_log_file,
            inputs=None,
            outputs=[logs_output],
        )

        # Process button click - also switches to Output tab
        def process_and_switch(*args):
            """Process document and return results plus tab selection."""
            # The last N arguments are the filetype args, before that are env args
            n_filetype_args = len(filetype_arg_components)
            n_env_args = len(env_arg_components)
            regular_args = args[: -(n_filetype_args + n_env_args)]
            env_arg_values = (
                args[-(n_filetype_args + n_env_args) : -n_filetype_args]
                if n_filetype_args > 0
                else args[-n_env_args:]
            )
            filetype_arg_values = args[-n_filetype_args:] if n_filetype_args > 0 else []

            # Build env_args dict from component values
            env_args_dict = {}
            for (field_name, component), value in zip(
                env_arg_components.items(), env_arg_values
            ):
                if value is not None and value != "":
                    env_args_dict[field_name] = value

            # Build filetype_args dict from component values
            filetype_args_dict = {}
            for (arg_name, component), value in zip(
                filetype_arg_components.items(), filetype_arg_values
            ):
                # Parse special types
                if value is not None and value != "":
                    arg_type = filetype_arg_types[arg_name]
                    type_str = str(arg_type)

                    if arg_type == dict or "dict" in type_str:
                        # Parse JSON string to dict
                        try:
                            import json

                            filetype_args_dict[arg_name] = json.loads(value)
                        except:
                            logger.warning(
                                f"Failed to parse {arg_name} as JSON: {value}"
                            )
                            continue
                    elif "List" in type_str or arg_type == list:
                        # Parse comma-separated string to list
                        filetype_args_dict[arg_name] = [
                            item.strip() for item in value.split(",")
                        ]
                    elif "Union[str, List[str]]" in type_str:
                        # Parse as list if comma-separated, otherwise as string
                        if "," in value:
                            filetype_args_dict[arg_name] = [
                                item.strip() for item in value.split(",")
                            ]
                        else:
                            filetype_args_dict[arg_name] = value
                    else:
                        filetype_args_dict[arg_name] = value

            md_output, text_output = process_document(
                *regular_args, env_args_dict, filetype_args_dict
            )
            # Return results and update to select the Output tab (id=1)
            return md_output, text_output, gr.update(selected=1)

        process_btn.click(
            fn=process_and_switch,
            inputs=[
                task,
                path_text,
                uploaded_file,
                query_text,
                model,
                filetype,
                llm_verbosity,
                disable_llm_cache,
                dollar_limit,
                embed_model,
                top_k,
                query_retrievers,
                query_eval_model,
                query_eval_check_number,
                summary_n_recursion,
                load_embeds_from,
                save_embeds_as,
            ]
            + list(env_arg_components.values())
            + list(filetype_arg_components.values()),
            outputs=[output_md, output_text, main_tabs],
        )

        # Download button - create temporary file with the markdown content
        output_text.change(
            fn=lambda text: gr.DownloadButton(
                label="📥 Download as Markdown",
                value=text,
                visible=bool(text),
            )
            if text
            else gr.DownloadButton(visible=False),
            inputs=[output_text],
            outputs=[download_btn],
        )

        gr.Markdown(
            "\n\n---\n\n*Built for [wdoc](https://github.com/thiswillbeyourgithub/wdoc) with [Gradio](https://gradio.app). "
            "Assisted by [aider.chat](https://github.com/Aider-AI/aider/)*"
        )

    return interface


if __name__ == "__main__":
    # Configure logging to show in Docker logs
    logger.info("Starting wdoc Gradio interface...")

    # Create and launch interface
    # Uses environment variables for host/port configuration
    interface = create_interface()

    # Configure queue with max_size=1 to ensure sequential processing
    # wdoc does not parallelize well, so we limit to one request at a time
    interface.queue(max_size=1)

    interface.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,  # Set to True to create a public link
        pwa=True,
    )
