from langchain_community.document_loaders import DirectoryLoader , PDFPlumberLoader
loader = DirectoryLoader(
    path = 'books',
    glob = '*.pdf',
    loader_cls = PDFPlumberLoader
)
docs = loader.lazy_load()

for doc in docs :
    print(doc.metadata)