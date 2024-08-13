retriever_harrypotter = harrypotter_vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5, "include_metadata": True})
retriever_got = got_vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5, "include_metadata": True})

# Let's Merge both Retriever
# It is also called lord of retriever(LOTR)

from langchain.retrievers.merger_retriever import MergerRetriever
lotr = MergerRetriever(retrievers=[retriever_harrypotter, retriever_got])
     

for chunks in lotr.get_relevant_documents("Who was the jon snow?"):
    print(chunks.page_content)


for chunks in lotr.get_relevant_documents("Who is a harry potter?"):
    print(chunks.page_content)

# But the result from these two  is too much messy now lets refine it according to the question and overcome the situation of lost in middle
