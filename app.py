import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import time


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = "pdf-soru-cevap"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY ortam değişkenini tanımlayın.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ortam değişkenini tanımlayın.")


try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    print(f"Pinecone istemcisi başlatılamadı: {e}")
    pc = None


def process_pdf_and_create_index(file_path):
    if pc is None:
        return None

    if not os.path.exists(file_path):
        print(f"PDF dosyası bulunamadı: {file_path}")
        return None

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"PDF başarıyla yüklendi: {len(documents)} sayfa")
    except Exception as e:
        print(f"PDF yüklenirken hata oluştu: {e}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Embeddings modeli yüklenirken hata oluştu: {e}")
        return None

    try:
        index_list = pc.list_indexes()
        index_exists = False
        for index_item in index_list:
            if getattr(index_item, "name", None) == INDEX_NAME or index_item.get("name", None) == INDEX_NAME:
                index_exists = True
                break

        if index_exists:
            try:
                existing_index = pc.Index(INDEX_NAME)
                index_stats = existing_index.describe_index_stats()
                current_dimension = index_stats.get('dimension', 0)

                if current_dimension != 1536:
                    pc.delete_index(INDEX_NAME)
                    index_exists = False
                    import time
                    time.sleep(2)
            except Exception as e:
                try:
                    pc.delete_index(INDEX_NAME)
                    index_exists = False
                    import time
                    time.sleep(2)
                except:
                    pass

        if not index_exists:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)

        index = pc.Index(INDEX_NAME)

        docsearch = PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=INDEX_NAME
        )
        return docsearch

    except Exception as e:
        print(f"Pinecone ile ilgili bir hata oluştu: {e}")
        return None


PROMPT = PromptTemplate(
    template=(
        "Aşağıdaki bağlam parçalarına dayanarak soruya kısa ve spesifik bir cevap ver.\n"
        "Yalnızca bağlamdaki bilgilere dayan; bağlamda yoksa 'Bilmiyorum' de.\n"
        "Gereksiz açıklama yapma.\n\n"
        "Soru: {question}\n\n"
        "Bağlam:\n{context}"
    ),
    input_variables=["question", "context"]
)

def build_qa_chain(docsearch):
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=512
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 6, "score_threshold": 0.3}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa


def ask_question(query, docsearch):
    if not docsearch:
        print("Pinecone indeksi yüklenemediği için soru-cevap yapılamıyor.")
        return None

    try:
        related_docs = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 6, "score_threshold": 0.3}
        ).invoke(query)

        if not related_docs:
            print("Bu soruyu yanıtlayacak bilgi PDF içinde bulunamadı.")
            return None

        qa = build_qa_chain(docsearch)
        result = qa.invoke({"query": query})

        print(f"\nCevap: {result['result']}")
        return result

    except Exception as e:
        print(f"Soru-cevap sırasında hata oluştu: {e}")
        return None


if __name__ == "__main__":
    pdf_file_path = "örnek_belge.pdf"

    try:
        docsearch = process_pdf_and_create_index(pdf_file_path)

        if docsearch:
            while True:
                question = input("\nPDF hakkında bir soru sorun (Çıkmak için 'exit' yazın): ")
                if question.lower() == 'exit':
                    break
                result = ask_question(question, docsearch)
                if result is None:
                    print("Soru yanıtlanamadı.")

    except Exception as e:
        print(f"Programın ana bölümünde bir hata oluştu: {e}")
