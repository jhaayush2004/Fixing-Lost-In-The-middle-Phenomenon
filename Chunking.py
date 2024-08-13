from langchain.text_splitter import RecursiveCharacterTextSplitter
     

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
     

text_harrypotter = text_splitter.split_documents(documnet_harrypotter)
     

text_got = text_splitter.split_documents(documnet_got)
