from ast import List
import asyncio
import hashlib
import os
import time
from typing import Any
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import chromadb
# from chromadb.api.async_client import AsyncClient
# from chromadb.api.client import Client
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import copy

import pandas as pd
from abc import ABC, abstractmethod

from taskgen.base import strict_json
from taskgen.base_async import strict_json_async

from taskgen.ranker import AsyncRanker, Ranker
from taskgen.utils import ensure_awaitable, get_source_code_for_func, top_k_index

###################
## Base Template ##
###################


class MemoryTemplate(ABC):
    """A generic template provided for all memories"""

    @abstractmethod
    def append(self, memory_list, mapper=None):
        """Appends multiple new memories"""
        pass

    # TODO Should this be deleted based on metadata key - value filter
    @abstractmethod
    def remove(self, existing_memory):
        """Removes an existing_memory. existing_memory can be str, or triplet if it is a Knowledge Graph"""
        pass

    @abstractmethod
    def reset(self):
        """Clears all memories"""

    @abstractmethod
    def retrieve(self, task: str):
        """Retrieves some memories according to task"""
        pass

    ## Some utility functions
    def read_file(self, filepath, text_splitter=None):
        if ".xls" in filepath:
            text = pd.read_excel(filepath).to_string()
        elif ".csv" in filepath:
            text = pd.read_csv(filepath).to_string()
        elif ".docx" in filepath:
            text = self.read_docx(filepath)
        # elif ".pdf" in filepath:
        #     text = self.read_pdf(filepath)
        else:
            raise ValueError(
                "File type not spported, supported file types: pdf, docx, csv, xls"
            )

        if not text_splitter:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
                separators=[".\n", "\n"],
            )

        texts = text_splitter.split_text(text)
        memories = [{"content": text, "filepath": filepath} for text in texts]
        return memories

    # def read_pdf(self, filepath):
    #     # Open the PDF file
    #     text_list = []
    #     with open(filepath, "rb") as file:
    #         pdf_reader = pypdf.PdfReader(file)
    #         for page in pdf_reader.pages:
    #             page_text = page.extract_text()
    #             if page_text:  # Ensure there's text on the page
    #                 text_list.append(page_text)
    #             else:
    #                 print("No text found on page")
    #     return "\n".join(text_list)

    def read_docx(self, filepath):
        doc = Document(filepath)
        text_list = []
        for para in doc.paragraphs:
            text_list.append(para.text)
        return "\n".join(text_list)


########################
## In-house vector db ##
########################

### BaseMemory is VectorDB that is natively implemented, but not optimised. This will be used for function-based RAG and other kinds of RAG that are not natively text-based. For more optimised memory, check out ChromaDbMemory


class BaseMemory(MemoryTemplate):
    """Retrieves top k memory items based on task. This is an in-house, unoptimised, vector db
    - Inputs:
        - `memory`: List. Default: None. The list containing the memory items
        - `top_k`: Int. Default: 3. The number of memory list items to retrieve
        - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
            - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
        - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items
            - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for contextual information
        - `llm`: Function. The llm to use for `strict_json` llm retriever
        - `retrieve_fn`: Default: None. Takes in task and outputs top_k similar memories in a list. Does away with the Ranker() altogether
        - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model.
            - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
    """

    def __init__(
        self,
        memory: list = None,
        top_k: int = 3,
        mapper=lambda x: x,
        approach="retrieve_by_ranker",
        llm=None,
        retrieve_fn=None,
        ranker=None,
    ):
        if memory is None:
            self.memory = []
        else:
            self.memory = memory
        self.top_k = top_k
        self.mapper = mapper
        self.approach = approach
        self.ranker = ranker
        self.retrieve_fn = retrieve_fn
        self.llm = llm

    def add_file(self, filepath, text_splitter=None):
        memories = self.read_file(filepath, text_splitter)
        texts = [memory["content"] for memory in memories]
        self.append(texts)

    def append(self, memory_list, mapper=None):
        """Adds a list of memories"""
        if not isinstance(memory_list, list):
            memory_list = [memory_list]
        self.memory.extend(memory_list)

    def remove(self, memory_to_remove):
        """Removes a memory"""
        self.memory.remove(memory_to_remove)

    def reset(self):
        """Clears all memory"""
        self.memory = []

    def isempty(self) -> bool:
        """Returns whether or not the memory is empty"""
        return not self.memory

    def get_python_representation(self, include_memory_elements) -> str:
        """Returns a string representation of the object for debugging"""
        return f"Memory(memory={self.memory if include_memory_elements else []}, top_k={self.top_k}, mapper={get_source_code_for_func(self.mapper)}, approach='{self.approach}', ranker={self.ranker.get_python_representation() if hasattr(self.ranker, 'get_python_representation') else 'None'})"


class Memory(BaseMemory):
    """Retrieves top k memory items based on task
    - Inputs:
        - `memory`: List. Default: Empty List. The list containing the memory items
        - `top_k`: Int. Default: 3. The number of memory list items to retrieve
        - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
            - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
        - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items
            - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for contextual information
        - `llm`: Function. The llm to use for `strict_json` llm retriever
        - `retrieve_fn`: Default: None. Takes in task and outputs top_k similar memories in a list
        - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model.
            - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ranker is None:
            self.ranker = (
                Ranker()
            )  # Assuming Ranker needs to be initialized if not provided

    def retrieve(self, task: str) -> list:
        """Performs retrieval of top_k similar memories according to approach stated"""
        # if you have your own vector search function, implement it in retrieve_fn. Takes in a task and outputs the top-k results
        if self.retrieve_fn is not None:
            return self.retrieve_fn(task)
        else:
            if self.approach == "retrieve_by_ranker":
                return self.retrieve_by_ranker(task)
            else:
                return self.retrieve_by_llm(task)

    def retrieve_by_ranker(self, task: str) -> list:
        """Performs retrieval of top_k similar memories
        Returns the memory list items corresponding to top_k matches"""
        # if there is no need to filter because top_k is already more or equal to memory size, just return memory
        if self.top_k >= len(self.memory):
            return copy.deepcopy(self.memory)

        # otherwise, perform filtering
        else:
            memory_score = [
                self.ranker(self.mapper(memory_chunk), task)
                for memory_chunk in self.memory
            ]
            top_k_indices = top_k_index(memory_score, self.top_k)
            return [self.memory[index] for index in top_k_indices]

    def retrieve_by_llm(self, task: str) -> list:
        """Performs retrieval via LLMs
        Returns the key list as well as the value list"""
        res = strict_json(
            f'You are to output the top {self.top_k} most similar list items in Memory relevant to this: ```{task}```\nMemory: {[f"{i}. {self.mapper(mem)}" for i, mem in enumerate(self.memory)]}',
            "",
            output_format={
                f"top_{self.top_k}_list": f"Indices of top {self.top_k} most similar list items in Memory, type: list[int]"
            },
            llm=self.llm,
        )
        top_k_indices = res[f"top_{self.top_k}_list"]
        return [self.memory[index] for index in top_k_indices]


class AsyncMemory(BaseMemory):
    """Retrieves top k memory items based on task
    - Inputs:
        - `memory`: List. Default: Empty List. The list containing the memory items
        - `top_k`: Int. Default: 3. The number of memory list items to retrieve
        - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
            - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
        - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items
            - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for contextual information
        - `llm`: Function. The llm to use for `strict_json` llm retriever
        - `retrieve_fn`: Default: None. Takes in task and outputs top_k similar memories in a list
        - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model.
            - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ranker is None:
            self.ranker = (
                AsyncRanker()
            )  # Assuming Ranker needs to be initialized if not provided
        if not isinstance(self.ranker, AsyncRanker):
            raise Exception("Sync Ranker not allowed in AsyncMemory")
        ensure_awaitable(self.retrieve_fn, "retrieve_fn")
        ensure_awaitable(self.llm, "llm")

    async def retrieve(self, task: str) -> list:
        """Performs retrieval of top_k similar memories according to approach stated"""
        # if you have your own vector search function, implement it in retrieve_fn. Takes in a task and outputs the top-k results
        if self.retrieve_fn is not None:
            return await self.retrieve_fn(task)
        else:
            if self.approach == "retrieve_by_ranker":
                return await self.retrieve_by_ranker(task)
            else:
                return await self.retrieve_by_llm(task)

    async def retrieve_by_ranker(self, task: str) -> list:
        """Performs retrieval of top_k similar memories
        Returns the memory list items corresponding to top_k matches"""
        # if there is no need to filter because top_k is already more or equal to memory size, just return memory
        if self.top_k >= len(self.memory):
            return copy.copy(self.memory)

        # otherwise, perform filtering
        else:
            tasks = [
                self.ranker(self.mapper(memory_chunk), task)
                for memory_chunk in self.memory
            ]
            memory_score = await asyncio.gather(*tasks)
            top_k_indices = top_k_index(memory_score, self.top_k)
            return [self.memory[index] for index in top_k_indices]

    async def retrieve_by_llm(self, task: str) -> list:
        """Performs retrieval via LLMs
        Returns the key list as well as the value list"""
        res = await strict_json_async(
            f'You are to output the top {self.top_k} most similar list items in Memory relevant to this: ```{task}```\nMemory: {[f"{i}. {self.mapper(mem)}" for i, mem in enumerate(self.memory)]}',
            "",
            output_format={
                f"top_{self.top_k}_list": f"Indices of top {self.top_k} most similar list items in Memory, type: list[int]"
            },
            llm=self.llm,
        )
        top_k_indices = res[f"top_{self.top_k}_list"]
        return [self.memory[index] for index in top_k_indices]


############################DISABLING CHROMA DB FOR NOW############################

######################
## Chroma vector db ##
######################

## TODO: Make this non-OpenAI dependent


# class BaseChromaDbMemory(MemoryTemplate):
#     """Takes in the following parameters:
#     `collection_name`: str. Compulsory. Name of the memory. Need to provide a unique name so that we can disambiguate between collections
#     `client` - Default: None. ChromaDB client to use, if any
#     `embedding_model`: Name of OpenAI's embedding_model to use with ChromaDB. Default OpenAI "text-embedding-3-small"
#     `top_k`: Number of elements to retrieve. Default: 3
#     `mapper`: Function. Maps the memory value to the embedded value. We do not need to embed the whole value, so this will serve as a way to tell us what to embed. Default: lambda x: x
#     `pre_delete`: Bool. Default: False. If set to True, delete collection with all data inside it when initialising
#     """

#     def __init__(
#         self,
#         collection_name,
#         client=None,
#         embedding_model="text-embedding-3-small",
#         top_k=3,
#         mapper=lambda x: x,
#         pre_delete=False,
#     ):
#         # Evaluate async client for chroma db for storage
#         self.client = client or chromadb.PersistentClient()
#         self.embedding_model = embedding_model
#         self.top_k = top_k
#         self.embedding_function = OpenAIEmbeddingFunction(
#             api_key=os.environ.get("OPENAI_API_KEY"), model_name=self.embedding_model
#         )
#         self.salt = os.urandom(16).hex()
#         self.collection_name = collection_name
#         self.mapper = mapper
#         self.collection = None
#         if pre_delete:
#             self.reset()
#         if isinstance(self.client, Client):
#             self.collection = self.client.get_or_create_collection(
#                 self.collection_name, embedding_function=self.embedding_function
#             )

#     @abstractmethod
#     def get_openai_client(self):
#         pass

#     @abstractmethod
#     def create_embedding(self, text):
#         pass

#     @abstractmethod
#     def get_or_create_collection(self):
#         pass

#     def remove(self, memories: list[str]):
#         if not isinstance(memories, list):
#             memories = [memories]
#         max_count = self.collection.count()
#         for memory in memories:
#             # retrieve up to top_k memories to remove
#             retrieved_ = self.collection.query(
#                 query_texts=[memory], n_results=min(self.top_k, max_count)
#             )
#             removed = False
#             for document, retrieved_id in zip(
#                 retrieved_["documents"][0], retrieved_["ids"][0]
#             ):
#                 # only remove on exact match
#                 if document == memory:
#                     self.collection.delete([retrieved_id])
#                     print(f"Removing memory: {memory}")
#                     removed = True
#             if not removed:
#                 print(f"No memory to remove: {memory}")

#     def remove_by_id(self, ids):
#         if not isinstance(ids, list):
#             ids = [ids]
#         return self.collection.delete(ids)

#     def generate_id(self, embedding):
#         # Generate a unique ID based on the embedding with added salt
#         current_time = str(time.time()).encode()
#         salted_embedding = str(embedding).encode() + self.salt.encode() + current_time
#         return hashlib.sha256(salted_embedding).hexdigest()

#     def reset(self):
#         return self.client.delete_collection(name=self.collection.name)

#     def retrieve(self, task: str, filter=[]):
#         max_count = self.collection.count()
#         results = self.collection.query(
#             query_texts=[task], n_results=min(self.top_k, max_count), where=filter
#         )["metadatas"][0]

#         return [
#             mem_item["taskgen_content"] if "taskgen_content" in mem_item else mem_item
#             for mem_item in results
#         ]
#         # return self.collection.query(
#         #     query_texts=[task], n_results=self.top_k, where=filter
#         # )


# class ChromaDbMemory(BaseChromaDbMemory):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.openai_client = self.get_openai_client()
#         self.collection = self.get_or_create_collection()

#     def get_openai_client(self):
#         from openai import OpenAI

#         return OpenAI()

#     def get_or_create_collection(self):
#         return self.client.get_or_create_collection(
#             self.collection_name, embedding_function=self.embedding_function
#         )

#     def add_file(self, filepath, text_splitter=None):
#         new_memories = self.read_file(filepath, text_splitter)
#         return self.append(new_memories, mapper=lambda x: x["content"])

#     def create_embedding(self, text):
#         return (
#             self.openai_client.embeddings.create(
#                 input=[text], model=self.embedding_model
#             )
#             .data[0]
#             .embedding
#         )

#     def append(self, new_memories, mapper=None):
#         if not isinstance(new_memories, list):
#             new_memories = [new_memories]

#         if mapper:
#             memory_strings = [mapper(memory) for memory in new_memories]
#         else:
#             memory_strings = [self.mapper(memory) for memory in new_memories]

#         if all(isinstance(item, dict) for item in new_memories):
#             metadatas = new_memories
#         else:
#             metadatas = [{"taskgen_content": item} for item in new_memories]

#         return self.append_memory_list(memory_strings, metadatas)

#     def append_memory_list(self, new_memories: list[str], metadatas: list[dict] = None):
#         embeddings = [self.create_embedding(text) for text in new_memories]
#         if metadatas is None:
#             metadatas = [{} for _ in new_memories]
#         ids = [
#             metadata.get("id") or self.generate_id(embedding)
#             for metadata, embedding in zip(metadatas, embeddings)
#         ]
#         return self.collection.upsert(
#             ids=ids,
#             embeddings=embeddings,
#             documents=new_memories,
#             metadatas=metadatas,
#         )


# class AsyncChromaDbMemory(BaseChromaDbMemory):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.openai_client = self.get_openai_client()

#     def get_openai_client(self):
#         from openai import AsyncOpenAI

#         return AsyncOpenAI()

#     async def get_or_create_collection(self):
#         if self.collection:
#             return self.collection
#         if isinstance(self.client, Client):
#             return self.client.get_or_create_collection(
#                 self.collection_name, embedding_function=self.embedding_function
#             )
#         elif isinstance(self.client, AsyncClient):
#             return await self.client.get_or_create_collection(
#                 self.collection_name, embedding_function=self.embedding_function
#             )

#     async def add_file(self, filepath, text_splitter=None):
#         new_memories = self.read_file(filepath, text_splitter)
#         return await self.append(new_memories, mapper=lambda x: x["content"])

#     async def create_embedding(self, text):
#         embedding = await self.openai_client.embeddings.create(
#             input=[text], model=self.embedding_model
#         )
#         return embedding.data[0].embedding

#     async def append(self, new_memories=[], mapper=None):
#         if not isinstance(new_memories, list):
#             new_memories = [new_memories]

#         if mapper:
#             memory_strings = [mapper(memory) for memory in new_memories]
#         else:
#             memory_strings = [self.mapper(memory) for memory in new_memories]

#         if all(isinstance(item, dict) for item in new_memories):
#             metadatas = new_memories
#         else:
#             metadatas = [{"taskgen_content": item} for item in new_memories]

#         return await self.append_memory_list(memory_strings, metadatas)

#     async def append_memory_list(
#         self, new_memories: list[str], metadatas: list[dict] = None
#     ):
#         self.collection = await self.get_or_create_collection()
#         embeddings = await asyncio.gather(
#             *[self.create_embedding(text) for text in new_memories]
#         )
#         if metadatas is None:
#             metadatas = [{} for _ in new_memories]
#         ids = [
#             metadata.get("id") or self.generate_id(embedding)
#             for metadata, embedding in zip(metadatas, embeddings)
#         ]
#         if isinstance(self.client, AsyncClient):
#             return await self.collection.upsert(
#                 ids=ids,
#                 embeddings=embeddings,
#                 documents=new_memories,
#                 metadatas=metadatas,
#             )
#         elif isinstance(self.client, Client):
#             return self.collection.upsert(
#                 ids=ids,
#                 embeddings=embeddings,
#                 documents=new_memories,
#                 metadatas=metadatas,
#             )
#         else:
#             raise Exception("Unknown client type")

#     async def remove(self, memories: list[str]):
#         self.collection = await self.get_or_create_collection()
#         if isinstance(self.client, AsyncClient):
#             if not isinstance(memories, list):
#                 memories = [memories]
#             max_count = self.collection.count()
#             n_results_num = min(self.top_k, max_count)
#             for memory in memories:
#                 # retrieve up to top_k memories to remove
#                 retrieved_ = await self.collection.query(
#                     query_texts=[memory], n_results=n_results_num
#                 )
#                 removed = False
#                 for document, retrieved_id in zip(
#                     retrieved_["documents"][0], retrieved_["ids"][0]
#                 ):
#                     # only remove on exact match
#                     if document == memory:
#                         await self.collection.delete([retrieved_id])
#                         print(f"Removing memory: {memory}")
#                         removed = True
#                 if not removed:
#                     print(f"No memory to remove: {memory}")
#         else:
#             return super().remove(memories)

#     async def remove_by_id(self, ids=[]):
#         self.collection = await self.get_or_create_collection()
#         if isinstance(self.client, AsyncClient):
#             return await self.collection.delete(ids)
#         else:
#             return super().remove(ids)

#     async def reset(self):
#         self.collection = await self.get_or_create_collection()
#         if isinstance(self.client, AsyncClient):
#             return await self.client.delete_collection(name=self.collection.name)
#         return super().reset()

#     async def retrieve(self, task: str, filter=[]):
#         self.collection = await self.get_or_create_collection()
#         if isinstance(self.client, AsyncClient):
#             max_count = self.collection.count()
#             n_results_num = min(self.top_k, max_count)
#             results = await self.collection.query(
#                 query_texts=[task], n_results=n_results_num, where=filter
#             )["metadatas"][0]

#             return [
#                 (
#                     mem_item["taskgen_content"]
#                     if "taskgen_content" in mem_item
#                     else mem_item
#                 )
#                 for mem_item in results
#             ]

#         else:
#             return super().retrieve(task, filter)
