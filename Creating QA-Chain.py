# Load the model from huggingface
!pip install llama-cpp-python
from langchain.llms import LlamaCpp
llms = LlamaCpp(streaming=True,
                   model_path="/content/drive/MyDrive/zephyr-7b-beta.Q4_K_M.gguf",
                   max_tokens = 1500,
                   temperature=0.75,
                   top_p=1,
                   gpu_layers=0,
                   stream=True,
                   verbose=True,n_threads = int(os.cpu_count()/2),
                   n_ctx=4096)
from langchain.chains import RetrievalQA
     

qa = RetrievalQA.from_chain_type(
      llm=llms,
      chain_type="stuff",
      retriever = compression_retriever_reordered,
      return_source_documents = True
)
     

query ="who is jon snow?"
results = qa(query)
print(results['result'])
#
print(results["source_documents"])

# What is LlamaCpp?
# LlamaCpp is a library for working with LLaMA (Large Language Model Architecture) models in C++. It provides an interface to load and run inference on LLaMA models, which are large-scale pre-trained language models. This library is often used for deploying language models in environments where performance and resource efficiency are critical. It supports features like streaming, tokenization, and context management, making it suitable for building applications that require real-time language processing.
