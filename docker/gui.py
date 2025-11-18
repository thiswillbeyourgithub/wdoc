#!/usr/bin/env python3
"""
Gradio web UI for wdoc document processing.

This provides a browser-based interface to wdoc's query, summarize, and parse
capabilities. Designed to run in a Docker container with volume-mounted vectorstore.

This file was created with assistance from aider.chat.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Any
import tempfile

import gradio as gr
from loguru import logger

from wdoc.wdoc import wdoc


def process_document(
    task: str,
    path_text: str,
    uploaded_file: Optional[Any],
    query_text: str,
    model: str,
    filetype: str,
    parse_format: str,
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
    parse_format : str
        Output format for parse task

    Returns
    -------
    Tuple[str, str]
        (markdown_output, download_content) - the rendered output and downloadable text
    """
    try:
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

        # Create wdoc instance based on task
        if task == "query":
            if not query_text or not query_text.strip():
                return "❌ **Error**: Query text is required for query task.", ""

            logger.info(f"Starting query task with query: {query_text}")
            instance = wdoc(
                task="query",
                path=path,
                filetype=filetype,
                model=model,
                # If vectorstore exists, we could load it here
                # For now, wdoc will create embeddings on the fly
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
            )
            result = instance.summary_task()
            output_md = f"# Summary\n\n{result['summary']}"

            # Add metadata if available
            if "doc_total_cost" in result:
                output_md += f"\n\n---\n*Total cost: ${result['doc_total_cost']:.4f}*"

        elif task == "parse":
            logger.info(f"Starting parse task with format: {parse_format}")
            result = wdoc.parse_doc(
                path=path,
                filetype=filetype,
                format=parse_format,
            )

            # Format output based on parse_format
            if parse_format == "text":
                output_md = f"# Parsed Document\n\n```\n{result}\n```"
            elif parse_format == "xml":
                output_md = f"# Parsed Document (XML)\n\n```xml\n{result}\n```"
            elif parse_format == "langchain":
                # Convert Document objects to readable format
                docs_text = "\n\n---\n\n".join(
                    [
                        f"## Document {i + 1}\n\n{doc.page_content}\n\n**Metadata**: {doc.metadata}"
                        for i, doc in enumerate(result)
                    ]
                )
                output_md = f"# Parsed Documents\n\n{docs_text}"
            elif parse_format == "langchain_dict":
                # Convert dicts to readable format
                docs_text = "\n\n---\n\n".join(
                    [
                        f"## Document {i + 1}\n\n{doc['page_content']}\n\n**Metadata**: {doc['metadata']}"
                        for i, doc in enumerate(result)
                    ]
                )
                output_md = f"# Parsed Documents\n\n{docs_text}"
            else:
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
    with gr.Blocks(title="wdoc Web UI", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            "# 📚 wdoc Web Interface\n\nProcess documents with AI-powered query, summarization, and parsing."
        )

        with gr.Tabs() as main_tabs:
            with gr.Tab("Input", id=0):
                # Task selection
                task = gr.Dropdown(
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

                with gr.Group(visible=False) as parse_group:
                    parse_format = gr.Dropdown(
                        choices=["text", "xml", "langchain", "langchain_dict"],
                        value="text",
                        label="Parse Output Format",
                        info="Format for parsed document output",
                    )

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
                            "txt",
                            "pdf",
                            "word",
                            "youtube",
                            "online_pdf",
                            "html",
                        ],
                        value="auto",
                        label="File Type",
                        info="Auto-detect or specify the file type",
                    )

                    # Process button
                    process_btn = gr.Button(
                        "🚀 Process Document", variant="primary", size="lg"
                    )

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
        def update_visibility(task_name: str) -> Tuple[gr.Group, gr.Group]:
            """Update visibility of task-specific input groups based on selected task."""
            return (
                gr.Group(visible=(task_name == "query")),  # query_group
                gr.Group(visible=(task_name == "parse")),  # parse_group
            )

        task.change(
            fn=update_visibility,
            inputs=[task],
            outputs=[query_group, parse_group],
        )

        # Process button click - also switches to Output tab
        def process_and_switch(*args):
            """Process document and return results plus tab selection."""
            md_output, text_output = process_document(*args)
            return md_output, text_output, gr.Tabs(selected=1)

        process_btn.click(
            fn=process_and_switch,
            inputs=[
                task,
                path_text,
                uploaded_file,
                query_text,
                model,
                filetype,
                parse_format,
            ],
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
            "\n\n---\n\n*Built with [wdoc](https://github.com/yourusername/wdoc) and [Gradio](https://gradio.app). "
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
    )
