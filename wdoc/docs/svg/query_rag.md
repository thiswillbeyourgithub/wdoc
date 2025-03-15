The portion of the README related to the query algorithm used in the RAG library is as follows:

### Query Algorithm:

1. **Embedding Creation**: The text is split into chunks, and embeddings are created for each chunk.
2. **Query Expansion**: The user query is embedded, and the default LLM is used to generate alternative queries, which are also embedded.
3. **Embedding Search**: The embeddings of the queries are used to search through all chunks of the text to retrieve the most relevant documents (initially set to 200).
4. **Document Filtering**: Each retrieved document is passed to a smaller LLM (default: anthropic/claude-3-5-haiku-20241022) to evaluate if the document is relevant to the user query.
5. **Iterative Search**: If more than 90% of the documents are relevant, another search is performed with a higher `top_k` value, and the process repeats until documents become irrelevant or the maximum number of documents (500) is reached.
6. **Information Extraction**: Each relevant document is sent to a stronger LLM (default: anthropic/claude-3-7-sonnet-20250219) to extract relevant information and generate an intermediate answer.
7. **Semantic Batching**: The intermediate answers are semantically batched by creating embeddings, performing hierarchical clustering, and grouping semantically similar answers together.
8. **Answer Combination**: Each batch of intermediate answers is combined into a single answer by the strong LLM. This process is repeated until only one final answer remains, which is returned to the user.

### Key Points:

- **High Recall and Specificity**: The system is designed to retrieve a large number of documents using carefully designed embedding searches.
- **Semantic Batching**: Intermediate answers are batched by semantic similarity using hierarchical clustering to ensure the final answer is coherent and logically structured.
- **Document Sourcing**: Each document is identified by a unique hash, and the final answer includes references to the exact portions of the source documents.

### Example Command:

```zsh
wdoc --path="https://example.com/document.pdf" --task=query --filetype="online_pdf" --query="What does it say about topic X?" --query_retrievers='default_multiquery' --top_k=auto_200_500
```

This command will:
1. Parse the PDF from the given URL.
2. Split the text into chunks and create embeddings.
3. Use the user query and alternative queries to retrieve relevant documents.
4. Evaluate, extract, and combine the information to produce a single final answer.
