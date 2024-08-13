# Fixing Lost in the Middle Phenomenon

## Overview

This repository addresses the ```"Lost in the Middle"``` phenomenon, a common issue in sequence modeling tasks where models tend to focus more on the ```beginning``` and ```end tokens```, losing context and performance on tokens in the middle. This can significantly impact the quality of text generation, comprehension, and retrieval tasks.

![App Screenshot](https://github.com/jhaayush2004/Fixing-Lost-In-The-middle-Phenomenon/blob/main/Visulas/LM.png)

### Understanding Figure :
Changing the location of relevant information
(in this case, the position of the passage that answers an
input question) within the language model’s input context results in a U-shaped performance curve—models
are better at using relevant information that occurs at the
very beginning (primacy bias) or end of its input context
(recency bias), and performance degrades significantly
when models must access and use information located
in the middle of its input context.

## The Phenomenon

The "lost in the middle" phenomenon refers to the ```degradation of model performance on tokens located in the middle of long sequences```. Models typically pay more attention to the initial and final parts of a sequence, often neglecting the middle parts, leading to a ```loss of important contextual information```.

![App Screenshot](https://github.com/jhaayush2004/Fixing-Lost-In-The-middle-Phenomenon/blob/main/Visulas/LITM.png)

### understanding figure:
The effect of changing the position of relevant information (document containing the answer) on multidocument question answering performance. Lower positions are closer to the start of the input context. Performance
is highest when relevant information occurs at the very start or end of the context, and rapidly degrades when models
must reason over information in the middle of their input context.

## How Filters and LOTR Help in Reducing the Lost in the Middle Phenomenon

### Overview

To combat the "lost in the middle" phenomenon, we utilize a combination of filters and the ```LOTR (Long-Term Order Retriever)``` framework. These techniques ensure that models maintain context and performance across entire sequences, not just at the beginning and end.

### Key Components and Their Roles

1. **Embeddings Redundant Filter**

   - **Purpose**: Removes redundant documents based on their embeddings.
   - **Benefit**: By ```eliminating duplicate``` or overly similar documents, this filter reduces noise and ensures that the dataset contains unique, valuable information. This focused dataset helps models pay better attention to all parts of the sequence, including the middle.

2. **Embeddings Clustering Filter**

   - **Purpose**: ```Groups similar documents``` together based on their embeddings.
   - **Benefit**: Clustering helps in organizing documents into coherent groups, making it easier for models to understand and retrieve relevant information from the middle of the sequence. This structured approach ensures that mid-sequence tokens are as well-attended as the initial and final tokens.

3. **Document Compression**

   - **Purpose**: ```Reduces the size``` of documents while preserving essential information.
   - **Benefit**: Compression ensures that the model can handle long contexts more efficiently. By keeping critical information intact and reducing unnecessary data, models can better focus on and retain middle parts of the sequence.

4. **Long Context Reorder**

   - **Purpose**: ```Reorders documents``` to manage long contexts more effectively.
   - **Benefit**: This reordering process ensures that important information, especially from the ```middle of long sequences```, is given appropriate emphasis. By rearranging the context, the model maintains a balanced focus across the ```entire sequence```.

### Document Compressor Pipeline

The ```Document Compressor Pipeline``` integrates the above techniques to filter, cluster, compress, and reorder documents. This comprehensive processing enhances the ```quality``` and ```relevance``` of the retrieved information, directly addressing the "lost in the middle" issue.

### Contextual Compression Retriever with LOTR

The ```Contextual Compression Retriever```, powered by the ```Document Compressor Pipeline``` and LOTR, retrieves and processes documents to mitigate the "lost in the middle" phenomenon effectively.

```python
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import LongContextReorder

# Assuming hf_bge_embeddings and lotr (a base retriever) are predefined variables
filter = EmbeddingsRedundantFilter(embeddings=hf_bge_embeddings)
clustering = EmbeddingsClusteringFilter(embeddings=hf_bge_embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[filter, clustering, reordering])
compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, 
    base_retriever=lotr,
    search_kwargs={"k": 3, "include_metadata": True}
)
```
![App Screenshot](https://github.com/jhaayush2004/Fixing-Lost-In-The-middle-Phenomenon/blob/main/Visulas/LOTR%20Blank%20board.png)

## How It Helps Combat the Issue
- **```Noise Reduction```**: The redundancy filter removes unnecessary duplicates, ensuring that only unique and relevant information is retained, which helps the model focus better on all parts of the sequence.
- **```Structured Retrieval```**: Clustering organizes documents into coherent groups, making it easier for the model to understand and process the entire sequence, including the middle.
- **```Efficient Context Management```**: Document compression ensures that essential information is preserved while reducing the overall size, allowing the model to handle long contexts more effectively.
- **```Balanced Attention```**: Long context reordering gives appropriate emphasis to mid-sequence information, ensuring that the model maintains a balanced focus across the entire sequence.
By integrating these filters and the ```LOTR``` framework, we significantly improve the model's ability to retain and process information from the middle of sequences, effectively reducing the ```"lost in the middle"``` phenomenon.

## Benefits
- **```Improved Relevance```**: By filtering out redundant documents and clustering similar ones, we ensure that the most relevant documents are retrieved, improving overall relevance.
- **```Enhanced Context Handling```**: The reordering and compression techniques ensure that long contexts are handled more effectively, preserving important information throughout the sequence.
- **```Noise Reduction```**: The redundancy filter reduces noise, making the retrieval process more efficient and effective

