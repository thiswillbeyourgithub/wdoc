# wdoc Advanced Summary Algorithm

The wdoc library implements a sophisticated document summarization algorithm designed to preserve the depth, structure, and logical flow of the original document while making it easier to digest.

## Core Philosophy

Unlike traditional summarization approaches that produce high-level takeaways with limited detail, wdoc's algorithm:

- **Preserves depth and nuance** by compressing the reasoning, arguments, and thought processes of the author
- **Maintains logical structure** using hierarchical markdown formatting with proper indentation
- **Ensures continuity** through context-aware chunk processing
- **Adapts to document size** with optional recursive summarization for extremely large documents

## How the Algorithm Works

### 1. Document Chunking
The document is divided into manageable chunks optimized for the LLM's context window.

### 2. Context-Aware Chunk Summarization
Each chunk is processed by a powerful LLM (default: `anthropic/claude-3-7-sonnet-20250219`) with specific instructions to:
- Create detailed summaries that retain important information
- Format output as markdown bullet points
- Use logical indentation to reflect the hierarchical structure of ideas
- Reference the end of the previous chunk's summary to maintain continuity

### 3. Optional Recursive Summarization
For extensive documents like books or research papers, the summary can be recursively processed:
- The first summary serves as input for a second round of summarization
- Controlled via the `--summary_n_recursion` parameter (e.g., `--summary_n_recursion=2`)
- Each recursion level produces a more condensed output while preserving key insights

### 4. Summary Concatenation
The individual chunk summaries are combined into a cohesive, well-structured document that:
- Is easy to skim with clear markdown formatting
- Preserves the logical flow and relationships between ideas
- Can be in the original language or translated during the summarization process

This approach results in summaries that are both comprehensive and accessible, making it easier to extract valuable information from complex documents.
