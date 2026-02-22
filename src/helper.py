from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings


### Extract text from Pdf files

def load_pdf_files(path):
    
    loader = DirectoryLoader(
        path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents




def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    
    minimal_docs : List[Document] = []
    
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {'source' : src}
            )
        )
    return minimal_docs


def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts = text_splitter.split_documents(docs)
    return texts




def download_embedding():
    
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    return embeddings



embedding = download_embedding()