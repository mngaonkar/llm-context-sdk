from bz2 import compress
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from zmq import Context
from backend import llm_provider
from backend.database import Database
from backend.utils import pretty_print_docs
from  constants import *
from loguru import logger
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from backend.vectorstore import VectorStore
from backend.llm_provider import LLMOllama, LLMLlamaCpp


class Pipeline():
    """Pipeline class for processing data"""
    chat_history = []

    def __init__(self):
        """Set default values."""
        self.llm_provider_name = LLM_PROVIDER_OLLAMA
        self.retriever = None
        
    def setup_session_state(self, session_state = {}):
        """User specific session data"""
        self.session_state = session_state

    def setup_prompt_tepmlate(self, template=None):
        self.prompt = PromptTemplate(
        template="""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {history}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {context}
        {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["history", "context", "question"],
        )

    def setup_large_language_model_provider(self, model = MODEL_NAME, base_url = INFERENCE_URL_OLLAMA):
        logger.debug(f"session state = {self.session_state}")
        logger.info(f"Setting up LLM provider with model {model} and base url {base_url}")
        self.llm_provider = LLMOllama(base_url=base_url, model=model)
        # self.llm = LLMLlamaCpp(base_url=constants.INFERENCE_URL_LLAMA_CPP_LOCAL)

    def setup(self, vector_store: VectorStore):
        """Overall setup."""
        self.setup_prompt_tepmlate()
        self.setup_large_language_model_provider(self.session_state["model"], INFERENCE_URL_OLLAMA)
        self.vector_store = vector_store
        if self.session_state.get("augmented_flag"):
            self.vector_store.init_vectorstore(self.session_state["dataset"])
        self.setup_chain()
        logger.info("Pipeline setup complete.")
    
    def setup_chain(self):
        """Setup the pipeline for the chatbot."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        if self.session_state.get("augmented_flag"):
            self.retriever = self.vector_store.database.vector_db.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def get_context_with_reranked_docs_bge(prompt):
            """Get the context of the conversation with reranked documents (BGE)."""
            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            compressor = CrossEncoderReranker(model=model, top_n=10)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.retriever
            )

            compressed_docs = compression_retriever.invoke("What is the plan for the economy?")
            pretty_print_docs(compressed_docs)

            return format_docs(compressed_docs)

        def get_context_with_reranked_docs(prompt):
            """Get the context of the conversation with reranked documents."""
            compressor = FlashrankRerank(top_n=10)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                                   base_retriever=self.retriever)
            compressed_docs = compression_retriever.invoke(prompt)
            pretty_print_docs(compressed_docs)
            return format_docs(compressed_docs)
        
        def get_no_context(prompt):
            """Return no context."""
            return ""
        
        def get_context(prompt):
            """Get the context of the conversation."""
            doc_list = []
            if self.vector_store is not None:
                docs = self.vector_store.database.query_document(prompt)
                for index, (doc, score) in enumerate(docs):
                    logger.info(f"Document {index} score: {score}")
                    doc_list.append(doc)

                pretty_print_docs(doc_list)
            return format_docs(doc_list)
            
        # build the pipeline
        if "augmented_flag" in self.session_state and self.session_state["augmented_flag"]:
            self.rag_chain = (
                {"history":self.get_session_history, "context": get_context, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm_provider.llm
                | StrOutputParser()
            )
        else:
            self.rag_chain = (
                {"history":self.get_session_history, "context": get_no_context, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm_provider.llm
                | StrOutputParser()
            )

    def setup_vector_store(self, store: VectorStore):
            self.vector_store = store

    def generate_response(self, query):
        """Generate a response from the LLM model."""
        response = self.rag_chain.invoke(query,
                                         config={
                                             "configurable": {'session_id': '1234'}, 
                                         })

        return response
    
    def stream_response(self, prompt):
        """Stream a response from the LLM model."""
        for chunk in self.rag_chain.stream(prompt,
                                           config={'callbacks': [ConsoleCallbackHandler()]}):
            yield chunk

    
    def get_session_history(self, prompt):
        logger.info(f"length of chat history: {len(self.chat_history)}")
        if len(self.chat_history) > MAX_CHAT_HISTORY:
            logger.info("chat history is greater than max chat history, truncating...")
            self.chat_history = self.chat_history[-MAX_CHAT_HISTORY:]
        return " ".join(self.chat_history)
