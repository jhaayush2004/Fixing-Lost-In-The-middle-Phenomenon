from langchain.document_loaders import PyPDFLoader
     
from google.colab import drive
drive.mount('/content/drive')
     

loader_harrypotter  = PyPDFLoader("/content/harry_potter_book.pdf")
documnet_harrypotter = loader_harrypotter.load()
     

loader_got = PyPDFLoader("/content/got_book.pdf")
documnet_got = loader_got.load()   


     



     
