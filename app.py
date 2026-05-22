import streamlit as st
import os
import json
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from transformers import AutoTokenizer
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Игнорируем конкретное предупреждение про Accessing __path__
# warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")

st.set_page_config(page_title="AI Библиотекарь", page_icon="📚")

DB_PATH = "./db_index"

# Функция для подсчёта токенов, отправляемых в qwen2.5
def count_and_log_qwen_tokens(prompt_value):
    # Корректное извлечение текста для любого типа промпта
    try:
        prompt_string = prompt_value.to_string()
    except AttributeError:
        prompt_string = str(prompt_value.to_messages())
    
    tokens = tokenizer.encode(prompt_string)
    token_count = len(tokens)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("rag_metrics.log", "a", encoding="utf-8") as f:
        f.write(f"[{current_time}] ---------- NEW REQUEST ----------\n")
        f.write(f"📈 Total Prompt Tokens: {token_count}\n")
        f.write(f"📝 Prompt : {prompt_string}")
        f.write("="*50 + "\n\n")
    
    return prompt_value

# --- ФУНКЦИИ ДЛЯ РАБОТЫ СО СПИСКОМ КНИГ ---
LIB_FILE = "library_list.json"

def save_to_library(filename):
    """Добавляет имя файла в список библиотеки"""
    library = load_library()
    if filename not in library:
        library.append(filename)
        with open(LIB_FILE, "w", encoding="utf-8") as f:
            json.dump(library, f, ensure_ascii=False)

def load_library():
    """Загружает список имен файлов из библиотеки"""
    if not os.path.exists(LIB_FILE):
        return []
    with open(LIB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def clear_library():
    # 1. Очищаем векторную базу
    clear_inside_db()
    
    # 2. Удаляем файл со списком названий книг
    if os.path.exists(LIB_FILE):
        os.remove(LIB_FILE)
    
    # 3. Очищаем историю чата в текущей сессии
    st.session_state.messages = []

def clear_inside_db():
    embeddings = get_embeddings()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    # Получаем ID всех документов и удаляем их
    all_ids = db.get()['ids']
    if all_ids:
        db.delete(ids=all_ids)

# Инициализируем модель векторизации один раз
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = get_embeddings()

# ---ФУНКЦИЯ ОБРАБОТКИ файла ---
def process_new_file(uploaded_file):
    temp_path = os.path.join("temp_" + uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = Docx2txtLoader(temp_path)
    data = loader.load()
    chunk_size = 800
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=(chunk_size/10))
    chunks = text_splitter.split_documents(data)
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    save_to_library(uploaded_file.name)
    
    os.remove(temp_path)
    return len(chunks)

# --- Инициализация RAG системы ---
# Инициализация базы и LLM
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
llm = OllamaLLM(model="qwen2.5:3b")

# Промпт для модели
template = """
ПРАВИЛА:
1. НЕ используй свои внешние знания.

КОНТЕКСТ ДЛЯ АНАЛИЗА:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

ВАШ ОТВЕТ:"""

prompt = ChatPromptTemplate.from_template(template)

base_retriever = db.as_retriever(search_kwargs={"k": 8})

compressor = FlashrankRerank(top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Tokenizer для предпосчёта токенов
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

rag_chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(count_and_log_qwen_tokens)
    | llm
    | StrOutputParser()
)

# --- ИНТЕРФЕЙС БОКОВОЙ ПАНЕЛИ ---
with st.sidebar:
    st.title("📚 Библиотека")
    
    # 1. Показать список уже загруженных книг
    library = load_library()
    if library:
        st.subheader("Загруженные книги:")
        for book in library:
            st.info(f"📖 {book}")
    else:
        st.write("Пока нет загруженных книг.")
    
    st.divider() # Линия-разделитель
    
    # 2. Форма для новой загрузки
    uploaded_file = st.file_uploader("Добавить книгу (.docx)", type="docx")
    if uploaded_file is not None:
        if st.button("Проиндексировать"):
            with st.spinner("Добавление в базу..."):
                process_new_file(uploaded_file)
                st.success("Книга добавлена!")
                st.rerun()
               
    
    st.divider() # Линия-разделитель
    
    if st.button("🗑️ Полная очистка базы"):
        with st.spinner("Удаление данных..."):
            clear_library()
            st.success("Все данные удалены!")
            st.rerun()

# --- ОСНОВНОЙ ЧАТ ---
st.title("📚 Чат с твоими книгами")

# История сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Задайте вопрос по книгам..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        # Добавить визуализацию ожидания
        with st.status("Ищу информацию в книгах...", expanded=False):
            response = rag_chain.invoke(prompt_input)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})