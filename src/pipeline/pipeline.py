from bz2 import compress
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from zmq import Context
from src.pipeline.utils import pretty_print_docs
from src.pipeline.constants import *
import logging
from langchain_core.callbacks import StdOutCallbackHandler

from src.database.vectorstore import VectorStore
from src.provider.llm_provider import LLMProvider
from src.provider.constants import LLMProviderType
from src.provider.llm_provider_config import LLMProviderConfig
from src.pipeline.pipeline_config import PipelineConfig
from src.database.dataset_config import DatasetConfig
from src.database.loader import DocumentLoader
from src.session.session import Session
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

CONFIG_DB = "deploy/configuration/configdb/config.db"

class Pipeline():
    """Pipeline class for processing data"""
    chat_history = []

    def __init__(self):
        """Set default values."""
        self.llm_provider_name = LLM_PROVIDER_OLLAMA
        self.retriever = None
        self.session_state = None
        self.session_list = {}
        
    def setup_session_state(self, session_state = {}):
        """User specific session data"""
        self.session_state = session_state

    def setup_prompt_tepmlate(self, template=None, input_variables=None):
        """Setup the prompt template."""

        if template is None: # default template
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
        else:
            self.prompt = PromptTemplate(template=template, 
                                         input_variables=input_variables)

    def setup_llm_provider(self, provider_type = LLMProviderType.OLLAMA.value):
        logger.debug(f"session state = {self.session_state}")
        logger.info(f"Setting up LLM provider with provider {provider_type}")

        config = LLMProviderConfig(CONFIG_DB)
        llm_config = config.get_llm_provider_config(provider_type)[0]

        try:
            self.llm_provider = LLMProvider.instantiate(provider_type, llm_config)
        except Exception as e:
            logger.error(f"Error setting up LLM provider: {e}")
            raise e

    def setup(self):
        """Overall setup."""
        logger.info("Setting up pipeline...")
        self.pipeline_config = PipelineConfig(CONFIG_DB).get_pipeline_config("default")[0]
        logger.debug(f"pipeline config = {self.pipeline_config}")
        self.llm_config = LLMProviderConfig(CONFIG_DB).get_llm_provider_config(self.pipeline_config["provider_name"])[0]

        self.setup_llm_provider(self.llm_config["provider_name"])
        self.setup_prompt_tepmlate(self.llm_config["template"], self.llm_config["input_variables"])
        document_loader = DocumentLoader(config=None)
        self.vector_store = VectorStore(document_loader, config=None)
        if self.pipeline_config.get("augmented_flag"):
            self.dataset_config = DatasetConfig(CONFIG_DB).get_dataset_config(self.pipeline_config["dataset"])[0]
            self.vector_store.init_vectorstore(self.dataset_config)
        self.setup_chain()
        logger.info("Pipeline setup complete.")
    
    def setup_chain(self):
        """Setup the pipeline for the chatbot."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        if self.pipeline_config.get("augmented_flag"):
            self.retriever = self.vector_store.database.vector_db.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
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
            
        def prompt_func(data: dict):
            text = data["text"]
            image = data.get("image", None)

            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            }

            context = get_context(data["text"])
            content_parts = []

            text_part = {"type": "text", "text": data["history"] + " " + context + " " + text}

            if image is not None:
                content_parts.append(image_part)
                text_part = {"type": "text", "text": data["history"] + " " + text}
            else:
                if "context" in data["text"]:
                    text_part = {"type": "text", "text": data["history"] + " " + context + " " + text}
                else:
                    text_part = {"type": "text", "text": data["history"] + " " + text}
            
            content_parts.append(text_part)

            return [HumanMessage(content=content_parts)]
    
        # build the pipeline
        if self.pipeline_config.get("augmented_flag"):
            self.rag_chain = (
                prompt_func
                | self.llm_provider.llm
                | StrOutputParser()
            )
        else:
            self.rag_chain = (
                prompt_func
                | self.llm_provider.llm
                | StrOutputParser()
            )

    def setup_vector_store(self, store: VectorStore):
            self.vector_store = store

    def generate_response(self, query:str, images:list[str], session_id:str):
        """Generate a response from the LLM model."""
        if len(images) == 0:
            image_data = ""
        else:
            image_data = images[0]

        # message = {
        #     "prompt": query,
        #     "images": [image_data],
        # }

        # message = HumanMessage(
        # content=[
        #     {"type": "text", "text": "Describe what's in the image."},
        #     {"type": "image", "image": image_data}
        # ]
        # )
        
        # message = f"You are a helpful assistant.  Here's an image: {image_data}.  What's in the image?"

        

        if image_data != "":
            message = {"text": "What's in the image?", "image": image_data, "history": ""}
        else:
            session = self.session_list.get(session_id)
            if session is not None:
                history = session.get_chat_history_as_string()
            else:
                history = ""
            message = {"text": query, "history": history}

        response = self.rag_chain.invoke(input=message,
                                         config={
                                             'callbacks': [StdOutCallbackHandler()],
                                             'configurable': {
                                                 'session_id': session_id
                                             }
                                         })

        if session_id not in self.session_list:
            self.session_list[session_id] = Session()
            logger.info(f"Creating new session with id {session_id}")

        self.session_list[session_id].add_message(role="user", content=query)
        self.session_list[session_id].add_message(role="agent", content=response)

        return response
    
    def stream_response(self, prompt):
        """Stream a response from the LLM model."""
        for chunk in self.rag_chain.stream(prompt,
                                           config={'callbacks': [StdOutCallbackHandler()]}):
            yield chunk

    
    def get_session_history(self, input=None, config=None):
        """Get the session history."""
        session_id = config.get("configurable", {}).get("session_id", None)
        if session_id is None:
            logger.info("Session ID not provided")
            return ""
        
        if self.session_list.get(session_id) is None:
            logger.info(f"Session {session_id} not found")
            return ""
    
        session_history = self.session_list[session_id].get_chat_history_as_string()
        logger.info(f"Session history for {session_id}: {session_history}")
        return session_history
