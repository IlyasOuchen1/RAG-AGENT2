# RAG AGENT - Assistant IA avec RAG

Un AGENT IA puissant basé sur la technologie RAG (Retrieval-Augmented Generation) qui permet d'interagir avec vos documents interne de manière intelligente et aussi de faire des recherche externe a travers wikipedia search .

## 🚀 Fonctionnalités

- **Interface Web Interactive** : Interface utilisateur intuitive basée sur Streamlit
- **Support Multi-Documents** : Chargez et interrogez plusieurs types de documents (PDF, TXT, DOCX)
- **Recherche Sémantique** : Recherche intelligente dans vos documents avec contexte
- **Base de Connaissances Vectorielle** : Utilisation de Pinecone pour le stockage vectoriel
- **Outils de Recherche Avancés** :
  - Recherche générale dans tous les documents
  - Recherche dans un document spécifique
  - Résumé de documents
  - Statistiques de la base de données
  - Recherche avec contexte étendu
  - Découverte de contenu lié
- **Source Summary** : Chaque résultat de recherche inclut un résumé des sources utilisées, facilitant la traçabilité des informations.
- **Recherche Wikipedia** : Intégration de la recherche Wikipedia pour enrichir les résultats avec des informations externes.

## 📋 Prérequis

- Python 3.8+
- Clé API OpenAI
- Clé API Pinecone
- Compte Pinecone (plan gratuit disponible)

## 🛠️ Installation

1. Clonez le repository :
```bash
git clone https://github.com/IlyasOuchen1/RAG-AGENT2.git
cd RAG-AGENT2
```

2. Créez un environnement virtuel et activez-le :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Créez un fichier `.env` à la racine du projet avec vos clés API :
```
OPENAI_API_KEY=votre_clé_openai
PINECONE_API_KEY=votre_clé_pinecone
```

## 🚀 Utilisation

1. Lancez l'application :
```bash
streamlit run simple_rag_chat2.py
```

2. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8501)

3. Utilisez l'interface pour :
   - Télécharger des documents
   - Poser des questions sur vos documents
   - Explorer le contenu de votre base de connaissances

4. **Source Summary** : Chaque résultat de recherche inclut un résumé des sources utilisées, facilitant la traçabilité des informations.

5. **Recherche Wikipedia** : Utilisez la recherche Wikipedia pour obtenir des informations supplémentaires.

## 🛠️ Technologies Utilisées

- **Streamlit** : Interface utilisateur web
- **LangChain** : Framework pour les applications LLM
- **OpenAI** : Modèles de langage et embeddings
- **Pinecone** : Base de données vectorielle
- **Python-dotenv** : Gestion des variables d'environnement

## 📚 Types de Documents Supportés

- PDF (.pdf)
- Texte (.txt)
- Word (.docx)

## 🔒 Sécurité

- Les clés API sont stockées de manière sécurisée dans un fichier .env
- Les documents sont traités localement avant d'être envoyés à l'API
- Utilisation du plan gratuit de Pinecone pour la démonstration

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails. 
