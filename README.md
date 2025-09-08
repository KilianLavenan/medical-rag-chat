# 🩺 Medical RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** designed to answer medical questions about the management of **community-acquired pneumonia (PAC)**.  
It leverages **SentenceTransformers** for embeddings, **ChromaDB** for vector storage, and **OpenAI GPT-4o-mini** for natural language generation.  

The chatbot can handle **text, tables, and images** extracted from the official document provided (`Prise en charge des Pneumopathies aigues communautaires V2.docx`).

---

## 📖 Overview

This repository demonstrates a complete RAG implementation:


- **Data Handling:** Cleans, chunks, and embeds the docx data with Sentence Transformers.  
- **Retrieval & Query Processing:** Stores embeddings in ChromaDB and retrieves context for user queries.  
- **Response Generation:** Constructs a contextual prompt for OpenAI’s GPT and generates answers. 
- **Chatbot Interface:** Provides a user-friendly interface using streamlit.    

---

## ✨ Features

- 🔍 **Document ingestion:** Extracts paragraphs, tables, and images from DOCX.  
- 📑 **Table summarization:** Converts tables into readable narrative text.  
- 🖼 **Image interpretation:** Summarizes diagrams using GPT-4o-mini vision capabilities.  
- 📚 **Vector database (ChromaDB):** Stores embeddings for semantic search.  
- 💬 **Chat interface:** Interactive QA system with contextual memory (streamlit interface).  
- 🚀 **Deployment-ready:** Runs locally or deployable on Hugging Face Spaces.  

---

## 📂 Project Structure

medical-rag-chat/
├── app/
│ ├── init.py
│ ├── app.py # streamlit interface (main entry point)
│ ├── indexing.py # DOCX ingestion, chunking, embeddings, ChromaDB storage
│ └── query.py # Querying ChromaDB + building prompts
├── data/
│ ├── Prise en charge des Pneumopathies aigues communautaires V2.docx # Medical guideline (tracked via Git LFS)
│ ├── extracted_image.png # Extracted organigram image from DOCX
│ └── chroma_db/ # Persistent ChromaDB vector store
├── .env # Environment variables (e.g., OpenAI API key)
├── .gitattributes # Git LFS tracking configuration
├── .gitignore # Ignored files/folders
├── pyproject.toml # Project dependencies and configuration
├── README.md # Project documentation
└── requirements.txt # Optional: alternative dependency list
## ⚙️ Installation

1. Clone the repository:
   git clone https://github.com/your-username/medical-rag-chat.git
   cd medical-rag-chat
2. Install dependencies using uv:
    uv sync
3. Add your OpenAI API key in a .env file:
    OPENAI_API_KEY=sk-xxxxxx

## ▶️ Usage

You can use the chatbot in **two different ways**:  
either **locally on your computer** or directly on **Hugging Face Spaces**.

---

### 🔹 Option 1: Run locally

1. Make sure you installed the dependencies and set your `OPENAI_API_KEY` in a `.env` file (see [Installation](#installation)).

2. launch the streamlit interface:
   ```bash
   uv run python app/app.py

Then open http://localhost:7860 in your browser.

### 🔹 Option 2: Use Hugging Face Spaces

Go to the project’s Space on Hugging Face:
👉 https://huggingface.co/spaces/<username>/medical-rag-chat

The chatbot will launch directly in your browser with the streamlit interface.
No installation required, just type your medical question and get an answer.

On Spaces, the embeddings are precomputed so you can start chatting immediately.


## ❓ Demonstration Questions
These 3 question-answer pairs were selected in collaboration with doctors to cover different types of data from the document (text, tables, figures).
They represent the required deliverables.

### Q1

Question :
Un patient sans domicile fixe se presente en consultation, PAC typique à l'examen clinique. Comment donner suite à sa prise en charge ?

Pour la prise en charge d'un patient sans domicile fixe présentant une pneumopathie aigüe communautaire (PAC) typique, il est recommandé d'évaluer la gravité de sa situation. Étant donné que le patient appartient à une situation socio-économique défavorable, l'hospitalisation est fortement conseillée. Au cours de l'hospitalisation, il faudra effectuer des examens complémentaires, tels que des hémocultures et éventuellement une PCR virale, ainsi qu'une imagerie pulmonaire par radiographie ou échographie pulmonaire. La prise en charge doit également inclure l'instauration d'une antibiothérapie probabiliste adaptée aux PAC supposées d'origine bactérienne, en tenant compte de ses antécédents médicaux et de ses facteurs de risque.

Source :

- "Lieu de PEC : La hospitalisation est nécessaire en cas de signes de gravité et recommandée dans certaines situations (précarité, risque d’inobservance, FDR/comorbidités Cf tableau 2) (Annexe 1)"
- "Recherche de facteurs de risque de mortalité : [...] conditions socio-économiques défavorables [...] → Hospitalisation recommandée."
- "Imagerie pulmonaire de première intention : Radiographie du thorax ou Echographie pulmonaire"
- "Antibiothérapie probabiliste d’une PAC suspectée bactérienne :"

### Q2

Question :
Une patiente de 60 ans connue pour une BPCO, se présente aux urgences avec une dyspnée sévère nécessitant rapidement une ventilation mécanique pour maintenir une saturation en oxygène supérieure à 88%. La patiente est fébrile à 38,7°C et présente un état confusionnel aigu. Quels examens sont à envisager dans cette situation ?

Réponse :
Le diagnostic probable pour cette patiente est une pneumopathie aigue communautaire (PAC) grave, compte tenu des éléments cliniques présentés : détresse respiratoire nécessitant ventilation mécanique, fièvre à 38,7°C, état confusionnel aigu et antécédent de BPCO.
Examens à envisager :

- ECBC et hémocultures

- Antigénurie légionelle et pneumocoque

- Biologie sanguine

- Imagerie pulmonaire (radiographie thoracique ou TDM si doute ou absence d’amélioration)

Source :

- Tableau 1. Critères diagnostiques de pneumopathie aigue grave

- "PAC hospitalisée grave et A faire: Antigénurie Légionelle et pneumocoque, ECBC + Hémoculture + biologie sanguine"

- "Imagerie pulmonaire de première intention : Radiographie du thorax"

### Q3

Question (suite de la question 2) :
A l'interrogatoire, le patient mentionne un épisode d'EABPCO à Pseudomonas aeruginosa dans le même hopital il y à 3 mois. Quel antibiotique prescrire ?

Réponse :
Étant donné que le patient a un antécédent d'infection à Pseudomonas aeruginosa dans l'hôpital il y a 3 mois, il faut prescrire de l'antibiothérapie spécifiquement adaptée pour couvrir ce germe. Le traitement recommandé dans ce contexte serait de la Pipéracilline-tazobactam à une posologie de 12 à 16 g/24 h en intraveineux en perfusion continue, ou bien du Céfépime à 6 g/24 h. Il est également possible d'associer un macrolide (comme la spiramycine) en cas de pneumopathie grave.

Source :
"Facteur de risque de Pseudomonas aeruginosa... Antibiothérapie 1ère ligne: PIPERACILLINE/TAZOBACTAM 12 à 16g/24h IV continu OU CEFEPIME 6 g/24h" (Extrait de la section « Antibactéries pour PAC grave suspectées »).

