import streamlit as st
import os
from dotenv import load_dotenv
import tempfile

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Configuration Streamlit
st.set_page_config(page_title="RAG Assistant", layout="wide")

# Variables d'environnement
load_dotenv()

# Vérification des clés API
if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("❌ Clés API manquantes. Vérifiez votre fichier .env")
    st.stop()

# Configuration Pinecone (corrigée pour le plan gratuit)
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "simple-rag"
    
    # Créer l'index s'il n'existe pas (région compatible avec le plan gratuit)
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Région compatible plan gratuit
        )
        st.success("✅ Index Pinecone créé avec succès!")
    
    return pc.Index(index_name)

# Configuration du vector store
@st.cache_resource
def init_vector_store():
    index = init_pinecone()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index=index, embedding=embeddings)

# Configuration du LLM avec GPT-4o Mini (version équilibrée)
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",  # GPT-4o Mini - équilibre vitesse/qualité
        temperature=0.1,      # Légère créativité contrôlée
        max_tokens=2000,      # Limite généreuse pour des réponses complètes
        request_timeout=30    # Timeout standard
    )

# Outils de recherche avancés
vector_store = init_vector_store()

# Configuration Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000))

@tool
def search_wikipedia(query: str) -> str:
    """Recherche des informations sur Wikipedia."""
    try:
        result = wikipedia.run(query)
        if result:
            response = f"📚 **Résultats Wikipedia pour '{query}':**\n\n{result}\n\nSource: Wikipedia"
            response += "\n\n📚 **Sources utilisées:**\n• Wikipedia"
            return response
        else:
            return f"Aucun résultat trouvé sur Wikipedia pour '{query}'."
    except Exception as e:
        return f"Erreur lors de la recherche Wikipedia: {str(e)}"

@tool
def retrieve(query: str) -> str:
    """Outil principal de retrieve - recherche dans tous les documents."""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "Aucune information trouvée dans les documents."
    
    results = []
    sources = set()
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get('filename', 'Document inconnu')
        sources.add(filename)
        source = f"Source: {filename}"
        results.append(f"📄 {filename} - Résultat {i}:\n{doc.page_content}\n{source}\n")
    
    # Add sources summary
    results.append("\n📚 **Sources utilisées:**")
    for source in sorted(sources):
        results.append(f"• {source}")
    
    return "\n".join(results)

@tool
def search_documents(query: str) -> str:
    """Recherche générale dans les documents uploadés."""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "Aucune information trouvée dans les documents."
    
    results = []
    sources = set()
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get('filename', 'Document inconnu')
        sources.add(filename)
        source = f"Source: {filename}"
        results.append(f"📄 {filename} - Extrait {i}:\n{doc.page_content}\n{source}\n")
    
    # Add sources summary
    results.append("\n📚 **Sources utilisées:**")
    for source in sorted(sources):
        results.append(f"• {source}")
    
    return "\n".join(results)

@tool
def search_specific_document(filename: str, query: str) -> str:
    """Recherche dans un document spécifique par nom de fichier."""
    docs = vector_store.similarity_search(
        query, 
        k=5,
        filter={"filename": filename}
    )
    if not docs:
        return f"Aucune information trouvée dans le document '{filename}' pour cette requête."
    
    results = []
    sources = set()
    for i, doc in enumerate(docs, 1):
        sources.add(filename)
        source = f"Source: {filename}"
        results.append(f"📄 {filename} - Section {i}:\n{doc.page_content}\n{source}\n")
    
    # Add sources summary
    results.append("\n📚 **Sources utilisées:**")
    for source in sorted(sources):
        results.append(f"• {source}")
    
    return "\n".join(results)

@tool
def get_document_summary(filename: str) -> str:
    """Obtient un résumé d'un document spécifique."""
    docs = vector_store.similarity_search(
        "", 
        k=10,  # Plus de chunks pour un meilleur résumé
        filter={"filename": filename}
    )
    if not docs:
        return f"Document '{filename}' non trouvé."
    
    content = "\n".join([doc.page_content for doc in docs[:5]])  # Premiers 5 chunks
    return f"📄 Résumé de '{filename}':\n{content[:1000]}..."

@tool
def get_database_stats() -> str:
    """Obtient des statistiques sur la base de données documentaire."""
    try:
        # Échantillonnage pour obtenir des stats
        sample_docs = vector_store.similarity_search("", k=100)
        
        if not sample_docs:
            return "La base de données est vide."
        
        # Compter les documents uniques
        filenames = [doc.metadata.get('filename', 'Inconnu') for doc in sample_docs]
        unique_files = set(filenames)
        
        # Compter les chunks par document
        file_chunks = {}
        for filename in filenames:
            file_chunks[filename] = file_chunks.get(filename, 0) + 1
        
        # Créer le rapport
        stats = f"📊 **Statistiques de la base de données:**\n\n"
        stats += f"• Nombre total de chunks: {len(sample_docs)}\n"
        stats += f"• Nombre de documents: {len(unique_files)}\n\n"
        
        stats += "📄 **Détails par document:**\n"
        for filename in sorted(unique_files):
            chunk_count = file_chunks.get(filename, 0)
            stats += f"• {filename}: {chunk_count} chunks\n"
        
        return stats
    except Exception as e:
        return f"Erreur lors de la récupération des statistiques: {str(e)}"

@tool
def list_available_documents() -> str:
    """Liste tous les documents disponibles dans la base avec détails."""
    try:
        # Récupérer un échantillon de documents pour obtenir les noms de fichiers
        docs = vector_store.similarity_search("", k=50)
        
        if not docs:
            return "Aucun document n'a été uploadé."
        
        # Organiser par nom de fichier avec métadonnées
        files_info = {}
        for doc in docs:
            filename = doc.metadata.get('filename', 'Inconnu')
            if filename not in files_info:
                files_info[filename] = {
                    'chunks': 0,
                    'upload_time': doc.metadata.get('upload_time', 'N/A'),
                    'file_size': doc.metadata.get('file_size', 'N/A')
                }
            files_info[filename]['chunks'] += 1
        
        # Formater la liste
        result = "📚 **Documents disponibles:**\n\n"
        for filename in sorted(files_info.keys()):
            info = files_info[filename]
            result += f"📄 **{filename}**\n"
            result += f"   • Chunks: {info['chunks']}\n"
            if info['upload_time'] != 'N/A':
                # Formatage simplifié de la date
                try:
                    date_part = info['upload_time'].split('T')[0]
                    result += f"   • Ajouté le: {date_part}\n"
                except:
                    pass
            result += "\n"
        
        return result
    except Exception as e:
        return f"Erreur: {str(e)}"

@tool
def search_with_context(query: str, context_size: int = 5) -> str:
    """Recherche avec plus de contexte (plus de chunks)."""
    docs = vector_store.similarity_search(query, k=context_size)
    if not docs:
        return "Aucune information trouvée dans les documents."
    
    results = []
    sources = set()
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get('filename', 'Document inconnu')
        sources.add(filename)
        source = f"Source: {filename}"
        results.append(f"📄 {filename} - Contexte {i}:\n{doc.page_content}\n{source}\n")
    
    # Add sources summary
    results.append("\n📚 **Sources utilisées:**")
    for source in sorted(sources):
        results.append(f"• {source}")
    
    return "\n".join(results)

@tool
def find_related_content(query: str) -> str:
    """Trouve du contenu lié/similaire dans les documents."""
    # Première recherche
    initial_docs = vector_store.similarity_search(query, k=2)
    if not initial_docs:
        return "Aucun contenu lié trouvé."
    
    # Utiliser le contenu trouvé pour chercher du contenu similaire
    related_results = []
    sources = set()
    for doc in initial_docs:
        # Recherche basée sur le contenu trouvé
        related_docs = vector_store.similarity_search(doc.page_content[:200], k=2)
        for related_doc in related_docs:
            if related_doc.page_content != doc.page_content:  # Éviter les doublons
                filename = related_doc.metadata.get('filename', 'Document inconnu')
                sources.add(filename)
                source = f"Source: {filename}"
                related_results.append(f"📄 {filename} - Contenu lié:\n{related_doc.page_content}\n{source}\n")
    
    if not related_results:
        return "Aucun contenu lié supplémentaire trouvé."
    
    # Add sources summary
    related_results.append("\n📚 **Sources utilisées:**")
    for source in sorted(sources):
        related_results.append(f"• {source}")
    
    return "\n".join(related_results[:3])  # Limiter à 3 résultats

# Configuration de l'agent avec tous les outils
def create_agent(search_mode="both"):
    llm = init_llm()
    
    # Define tools based on search mode
    tools = []
    if search_mode in ["both", "internal"]:
        tools.extend([
            retrieve,  # Outil principal de retrieve
            search_documents,
            search_specific_document,
            get_document_summary,
            list_available_documents,
            search_with_context,
            find_related_content
        ])
    if search_mode in ["both", "external"]:
        tools.append(search_wikipedia)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant expert en recherche documentaire. Tu as accès à plusieurs outils puissants :

🔍 **OUTILS DISPONIBLES** :
1. **retrieve(query)** : Outil principal pour récupérer des informations
2. **search_documents(query)** : Recherche générale dans tous les documents
3. **search_specific_document(filename, query)** : Recherche dans un document spécifique
4. **get_document_summary(filename)** : Résumé d'un document
5. **list_available_documents()** : Liste tous les documents disponibles
6. **search_with_context(query, context_size)** : Recherche avec plus de contexte
7. **find_related_content(query)** : Trouve du contenu connexe

📋 **INSTRUCTIONS** :
- Utilise TOUJOURS au moins un outil avant de répondre
- Pour les questions générales → retrieve() ou search_documents()
- Pour chercher dans un fichier spécifique → search_specific_document()
- Pour lister les fichiers → list_available_documents()
- Pour plus de détails → search_with_context()
- Réponds UNIQUEMENT avec les informations trouvées

💡 **EXEMPLES D'USAGE** :
- "Dis-moi ce que contiennent les documents" → retrieve("contenu")
- "Résume le document rapport.pdf" → get_document_summary("rapport.pdf")
- "Cherche dans document.pdf les infos sur X" → search_specific_document("document.pdf", "X")

🚫 **RÈGLES STRICTES** :
- Jamais de connaissances générales
- Toujours utiliser les outils
- Si rien trouvé : "Aucune information sur ce sujet dans les documents."
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )

# Fonction pour vérifier si un document existe déjà
def check_document_exists(filename):
    """Vérifie si un document est déjà dans la base vectorielle."""
    try:
        # Rechercher des documents avec ce nom de fichier
        docs = vector_store.similarity_search(
            "", 
            k=10,
            filter={"filename": filename}
        )
        return len(docs) > 0
    except Exception as e:
        return False

# Fonction pour traiter les fichiers (avec gestion des gros fichiers)
def process_file(uploaded_file):
    """Traite et indexe un fichier uploadé (évite les doublons et gère les gros fichiers)."""
    
    # Vérifier si le document existe déjà
    if check_document_exists(uploaded_file.name):
        return "exists"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Charger le document
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith('.txt'):
        loader = TextLoader(tmp_path)
    elif uploaded_file.name.endswith('.docx'):
        loader = Docx2txtLoader(tmp_path)
    else:
        os.unlink(tmp_path)
        return False
    
    docs = loader.load()
    os.unlink(tmp_path)
    
    # Diviser en chunks équilibrés
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Taille équilibrée
        chunk_overlap=100,   # Bon overlap pour la cohérence
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    
    # Ajouter metadata avec timestamp pour traçabilité
    import datetime
    current_time = datetime.datetime.now().isoformat()
    
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'filename': uploaded_file.name,
            'chunk_index': i,
            'total_chunks': len(chunks),
            'upload_time': current_time,
            'file_size': len(uploaded_file.getvalue())
        })
    
    # Indexer par batches pour éviter les erreurs de taille
    batch_size = 50  # Traiter 50 chunks à la fois
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            # Mettre à jour l'interface
            progress_text.text(f"⏳ Indexation batch {current_batch}/{total_batches} ({len(batch)} chunks)")
            progress_bar.progress(current_batch / total_batches)
            
            # Ajouter le batch à Pinecone
            vector_store.add_documents(batch)
            
            # Pause courte pour éviter la surcharge
            import time
            time.sleep(0.5)
        
        # Nettoyer l'interface
        progress_text.empty()
        progress_bar.empty()
        
        return len(chunks)
        
    except Exception as e:
        st.error(f"Erreur lors de l'indexation: {str(e)}")
        return False

# Interface utilisateur améliorée
st.title("🤖 Assistant RAG Multilingue (GPT-4o Mini)")
st.caption("🌐 Posez vos questions en français, anglais, espagnol ou toute autre langue")
st.caption("⚡ Powered by GPT-4o Mini - Rapide et efficace")

# Sidebar pour upload avec gestion des doublons
with st.sidebar:
    st.header("📁 Upload de Documents")
    
    # Afficher le statut de la base de données avec gestion des erreurs
    try:
        # Obtenir quelques stats sur les documents existants
        sample_docs = vector_store.similarity_search("", k=20)
        existing_files = set([doc.metadata.get('filename', 'Inconnu') for doc in sample_docs])
        
        if existing_files:
            st.success(f"✅ Connexion Pinecone active")
            st.info(f"📚 {len(existing_files)} document(s) déjà dans la base")
            with st.expander("Voir les documents existants"):
                for filename in sorted(existing_files):
                    st.write(f"📄 {filename}")
        else:
            st.info("📂 Base de données vide - Prête pour le premier document")
    except Exception as e:
        st.error(f"⚠️ Problème de connexion Pinecone: {str(e)}")
        st.info("💡 Vérifiez votre clé API et la configuration de l'index")
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier",
        type=['pdf', 'txt', 'docx']
    )
    
    if uploaded_file:
        # Afficher des informations sur le fichier
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.write(f"📁 Fichier: {uploaded_file.name}")
        st.write(f"📏 Taille: {file_size_mb:.2f} MB")
        
        if file_size_mb > 5:
            st.warning("⚠️ Fichier volumineux - Le traitement peut prendre du temps")
        
        with st.spinner("⏳ Vérification et traitement..."):
            result = process_file(uploaded_file)
            
            if result == "exists":
                st.warning(f"⚠️ Le document '{uploaded_file.name}' existe déjà dans la base !")
                st.info("💡 Pas besoin de le reprocesser. Vous pouvez directement poser des questions.")
            elif result and isinstance(result, int):
                st.success(f"✅ Nouveau document ajouté ! {result} chunks indexés pour '{uploaded_file.name}'")
                st.balloons()  # Animation de célébration
            elif result == False:
                st.error("❌ Type de fichier non supporté ou erreur de traitement")
            else:
                st.error("❌ Erreur lors du traitement")
    
    # Bouton pour effacer la base (optionnel)
    st.divider()
    if st.button("🗑️ Effacer la base de données", type="secondary"):
        if st.checkbox("Confirmer la suppression"):
            try:
                # Ici on peut supprimer l'index et le recréer
                # Ou utiliser d'autres méthodes selon votre configuration Pinecone
                st.warning("⚠️ Fonctionnalité de suppression à implémenter si nécessaire")
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    # Section d'aide
    st.header("💡 Exemples de Questions")
    st.markdown("""
    **🌐 L'assistant répond dans votre langue :**
    
    **📄 Recherche Interne (Documents):**
    - "Quels documents sont disponibles ?"
    - "What documents are available?"
    - "Résume le document rapport.pdf"
    - "Search for information about [topic] in my documents"
    
    **🌐 Recherche Externe (Wikipedia):**
    - "Recherche sur Wikipedia l'intelligence artificielle"
    - "Search Wikipedia for machine learning"
    - "¿Qué dice Wikipedia sobre blockchain?"
    - "Find information about quantum computing on Wikipedia"
    
    **🔄 Recherche Combinée (Les Deux):**
    - "Compare mes documents avec les infos Wikipedia sur [sujet]"
    - "What do my documents say about AI vs Wikipedia?"
    - "Información interna y externa sobre [tema]"
    
    **🔍 Recherche avancée :**
    - "Search with more context on [topic]"
    - "Cherche du contenu lié à [sujet]"
    - "Statistiques de la base de données"
    """)
    
    # Indicateur du mode actuel
    current_mode = st.session_state.get("search_mode", "both")
    if current_mode == "both":
        st.success("🔄 Mode Hybride: Documents + Wikipedia")
    elif current_mode == "internal":
        st.info("📄 Mode Interne: Documents uniquement")
    else:
        st.info("🌐 Mode Externe: Wikipedia uniquement")

# Zone principale avec sélecteur de mode
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("🔍 Mode de Recherche")
    # Initialiser search_mode dans session_state s'il n'existe pas
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "both"
    
    search_mode = st.radio(
        "Choisissez le type de recherche:",
        ["both", "internal", "external"],
        format_func=lambda x: {
            "both": "🔄 Les Deux (Documents + Wikipedia)",
            "internal": "📄 Documents Uploadés Seulement", 
            "external": "🌐 Wikipedia Seulement"
        }[x],
        index=0,
        key="search_mode_radio"
    )
    
    # Mettre à jour session_state
    st.session_state.search_mode = search_mode
    
    st.subheader("🛠️ Outils Disponibles")
    current_mode = st.session_state.get("search_mode", "both")
    if current_mode in ["internal", "both"]:
        st.write("""
        **📄 Outils Internes:**
        - 🔍 Recherche générale
        - 📄 Recherche spécifique
        - 📋 Résumé de document
        - 📚 Liste des documents
        - 🔎 Recherche contextuelle
        - 🔗 Contenu connexe
        - 📊 Statistiques
        """)
    
    if current_mode in ["external", "both"]:
        st.write("""
        **🌐 Outils Externes:**
        - 📚 Recherche Wikipedia
        """)
    
    st.info(f"Mode actuel: {current_mode.upper()}")
    st.caption("🌐 L'assistant s'adapte automatiquement à la langue de votre question")

with col1:
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input utilisateur avec traitement direct de la langue
    if prompt := st.chat_input("Posez votre question... | Ask your question..."):
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Détecter la langue et créer une instruction explicite
        def is_english(text):
            english_words = ['what', 'how', 'where', 'when', 'why', 'can', 'could', 'would', 'should', 'list', 'show', 'tell', 'explain', 'describe', 'the', 'for', 'tools', 'models', 'used']
            return any(word.lower() in text.lower() for word in english_words)
        
        def is_french(text):
            french_words = ['que', 'quoi', 'comment', 'où', 'quand', 'pourquoi', 'peux', 'pourrais', 'devrait', 'liste', 'montre', 'dis', 'explique', 'décris', 'les', 'pour', 'outils', 'modèles', 'utilisés']
            return any(word.lower() in text.lower() for word in french_words)
        
        # Créer un prompt avec instruction de langue explicite
        if is_english(prompt):
            enhanced_prompt = f"INSTRUCTION: Respond in English only.\n\nUser question: {prompt}"
        elif is_french(prompt):
            enhanced_prompt = f"INSTRUCTION: Répondre en français uniquement.\n\nQuestion utilisateur: {prompt}"
        else:
            enhanced_prompt = f"INSTRUCTION: Respond in the same language as this question.\n\nUser question: {prompt}"
        
        # Générer la réponse avec le mode sélectionné
        current_search_mode = st.session_state.get("search_mode", "both")
        agent = create_agent(search_mode=current_search_mode)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyse en cours... | Analysis in progress..."):
                try:
                    # Préparer l'historique pour l'agent
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclure le dernier message
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    # Ajouter l'information du mode de recherche au prompt
                    mode_info = {
                        "both": "You can search in uploaded documents AND Wikipedia. Use both sources when relevant.",
                        "internal": "You can ONLY search in uploaded documents. Do not use external knowledge.",
                        "external": "You can ONLY search on Wikipedia. Do not refer to uploaded documents."
                    }
                    
                    enhanced_prompt = f"{enhanced_prompt}\n\nSEARCH MODE: {current_search_mode.upper()} - {mode_info[current_search_mode]}"
                    
                    # Exécuter l'agent avec l'instruction de langue et mode
                    response = agent.invoke({
                        "input": enhanced_prompt,
                        "chat_history": chat_history
                    })
                    
                    answer = response['output']
                    
                    # Post-traitement pour vérifier la langue (en cas d'échec)
                    if is_english(prompt) and not any(word in answer.lower() for word in ['the', 'are', 'is', 'and', 'tools', 'documents']):
                        # Si la réponse semble être en français alors que la question était en anglais
                        st.warning("🔄 Correction automatique de la langue...")
                        
                        # Nouvelle tentative avec instruction plus forte
                        second_prompt = f"You MUST respond in ENGLISH. Translate and reformat this answer to English: {answer}"
                        second_response = agent.invoke({
                            "input": second_prompt,
                            "chat_history": []
                        })
                        answer = second_response['output']
                    
                    st.write(answer)
                    
                    # Afficher les outils utilisés
                    if 'intermediate_steps' in response:
                        with st.expander("🔧 Outils utilisés | Tools used"):
                            for step in response['intermediate_steps']:
                                tool_name = step[0].tool
                                tool_input = step[0].tool_input
                                st.write(f"**{tool_name}** : {tool_input}")
                    
                    # Ajouter à l'historique
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Erreur | Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Bouton pour vider l'historique
if st.sidebar.button("🗑️ Vider l'historique"):
    st.session_state.messages = []
    st.rerun()