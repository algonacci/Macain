from haystack.nodes import PDFToTextConverter, PreProcessor, EmbeddingRetriever, FARMReader
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
import time


def preprocessing(file_path):
    pdf_converter = PDFToTextConverter(
        remove_numeric_tables=True, valid_languages=["en"])
    converted = pdf_converter.convert(file_path=file_path, meta={
                                      "company": "Company_1", "processed": False})
    preprocessor = PreProcessor(
        split_by="word", split_length=200, split_overlap=10)
    preprocessed = preprocessor.process(converted)
    return preprocessed


def document_store(document_preprocessed):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    document_store = FAISSDocumentStore(
        sql_url='sqlite:///'+timestr+'_document_store.db', faiss_index_factory_str="Flat", return_embedding=True)
    document_store.delete_documents()
    document_store.write_documents(document_preprocessed)
    return document_store


def question_answer_pipeline(document_store):
    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                                   model_format="sentence_transformers")
    reader = FARMReader(
        model_name_or_path='deepset/tinyroberta-squad2', use_gpu=True)
    document_store.update_embeddings(retriever)
    pipeline = ExtractiveQAPipeline(reader, retriever)
    return pipeline
