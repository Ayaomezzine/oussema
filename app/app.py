import streamlit as st
import os
import PyPDF2
from pdf2image import convert_from_path
import io
from PIL import Image
import re
import base64
import numpy as np
import pytesseract
from io import BytesIO
import sys

# Code pour éviter les erreurs de surveillance PyTorch avec Python 3.12
import streamlit.watcher.local_sources_watcher
original_get_module_paths = streamlit.watcher.local_sources_watcher.get_module_paths

def patched_get_module_paths(module):
    try:
        return original_get_module_paths(module)
    except RuntimeError as e:
        if 'torch' in str(module):
            return []
        raise e

streamlit.watcher.local_sources_watcher.get_module_paths = patched_get_module_paths

# Configuration de la page Streamlit en PREMIER
st.set_page_config(
    page_title="Chatbot Fondations Superficielles",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importer sentence_transformers et faiss avec gestion d'erreur
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:
    st.error(f"Erreur lors de l'importation des modules PyTorch: {str(e)}")
    st.info("Solution: Utilisez Python 3.10 ou 3.11, ou exécutez l'application via run_app.py")
    st.stop()

# Initialisation de la session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'question_counter' not in st.session_state:
    st.session_state.question_counter = 1

# Classe pour stocker les extraits de texte et les images associées
class DocumentChunk:
    def __init__(self, text, page_num, file_path, images=None, has_formulas=False):
        self.text = text
        self.page_num = page_num
        self.file_path = file_path
        self.images = images or []
        self.has_formulas = has_formulas
        
    def __str__(self):
        return f"Page {self.page_num+1} de {os.path.basename(self.file_path)}: {self.text[:100]}..."

# Fonction pour détecter si un texte contient probablement des formules mathématiques
def detect_formulas(text):
    # Motifs communs dans les formules mathématiques
    patterns = [
        r'\b[A-Za-z]\s*=\s*[0-9]+',  # Variables simples (ex: q = 5)
        r'\\sigma',                  # Symbole sigma
        r'\\alpha',                  # Symbole alpha
        r'\\beta',                   # Symbole beta
        r'\\gamma',                  # Symbole gamma
        r'\\delta',                  # Symbole delta
        r'\\sum',                    # Somme
        r'\\int',                    # Intégrale
        r'\\frac',                   # Fraction
        r'\\sqrt',                   # Racine carrée
        r'[A-Za-z]\^[0-9]',          # Exposants (ex: x^2)
        r'\\over',                   # Division
        r'\\pi',                     # Pi
        r'\\infty',                  # Infini
        r'_{',                       # Indices
        r'\$.*\$',                   # LaTeX délimité
        r'[0-9]\\times[0-9]',        # Multiplication
        r'\\leq',                    # Inférieur ou égal
        r'\\geq',                    # Supérieur ou égal
        r'\\approx',                 # Approximativement égal
        r'[0-9]+,[0-9]+ ?[a-zA-Z]*', # Nombres avec unités (ex: 2,5 kN)
        r'[0-9]+\.[0-9]+ ?[a-zA-Z]*' # Nombres avec unités (ex: 2.5 kN)
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

# Fonction pour extraire le texte des images - modifiée pour ne pas afficher de message
def extract_text_from_image(image):
    try:
        # Essayer avec l'anglais par défaut
        text = pytesseract.image_to_string(image)
        
        if not text or len(text.strip()) < 5:
            return None
        return text
    except Exception as e:
        return None

# Fonction pour extraire le texte et les images des PDF
@st.cache_resource(show_spinner=False)
def process_pdfs(pdf_files):
    chunks = []
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    processed_files = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        
        try:
            # Extraction du texte
            pdf_reader = PyPDF2.PdfReader(pdf_path)
            
            # Conversion en images
            try:
                images = convert_from_path(pdf_path)
                processed_files.append(f"Images extraites avec succès de {filename}")
            except Exception as e:
                processed_files.append(f"Erreur lors de la conversion du PDF en images: {e}")
                images = []
            
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Diviser le texte en paragraphes
                paragraphs = re.split(r'\n\s*\n', text)
                
                # Créer des chunks de texte significatifs
                for j, para in enumerate(paragraphs):
                    if len(para.strip()) > 50:  # Ignorer les paragraphes trop courts
                        page_images = []
                        if i < len(images):
                            page_images.append(images[i])
                        
                        has_formulas = detect_formulas(para)
                        chunks.append(DocumentChunk(para.strip(), i, pdf_path, page_images, has_formulas))
        except Exception as e:
            processed_files.append(f"Erreur lors du traitement du fichier {filename}: {e}")
    
    # Créer l'index FAISS pour la recherche sémantique
    if chunks:
        texts = [chunk.text for chunk in chunks]
        embeddings = model.encode(texts)
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        return chunks, model, index, processed_files
    else:
        return [], None, None, ["Aucun contenu n'a pu être extrait des PDF."]

# Fonction pour rechercher les réponses pertinentes
def search_documents(query, chunks, model, index, k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector).astype('float32'), k)
    
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    
    return results

# Fonction pour identifier et formater les équations mathématiques
def format_math_equations(text):
    # Motifs pour détecter les équations mathématiques
    equation_patterns = [
        # Équations avec des symboles d'égalité
        r'([A-Za-z0-9σγφτ_]+\s*(?:=|<|>|≤|≥)\s*[^.\n;:,]{3,}(?:\n|$))',
        # Équations avec des formules complexes
        r'((?:[A-Za-z0-9σγφτ_]+\s*[+\-*/^]\s*)+[A-Za-z0-9σγφτ_\(\)]+)',
        # Équations avec des fractions
        r'([A-Za-z0-9σγφτ_]+\s*/\s*[A-Za-z0-9σγφτ_\(\)]+)',
        # Formules et équations numériques
        r'(\b[A-Za-z]\s*=\s*[0-9]+(?:[.,][0-9]+)?\s*(?:[a-zA-Z²³]*)\b)',
        # Formules avec unités
        r'(\b[0-9]+(?:[.,][0-9]+)?\s*(?:kN|MPa|m|cm|kPa|kg|N)\b)'
    ]
    
    # Identifier et remplacer les équations par du HTML formaté
    for pattern in equation_patterns:
        text = re.sub(
            pattern,
            r'<div class="math-equation">\1</div>',
            text,
            flags=re.MULTILINE
        )
    
    return text

# Fonction pour formater la réponse avec correction d'exercice et mise en évidence des équations
def format_answer(results, query, is_exercise_question=False):
    if not results:
        return "Je n'ai pas trouvé d'informations pertinentes sur ce sujet dans les documents fournis."
    
    # Déterminer si c'est une question d'exercice
    exercise_keywords = ["exercice", "calculer", "déterminer", "trouver", "résoudre", "valeur", 
                         "dimension", "semelle", "charge", "capacité", "portante", "résistance", 
                         "contrainte", "sol", "fondation", "largeur", "profondeur"]
    
    is_exercise = is_exercise_question or any(keyword.lower() in query.lower() for keyword in exercise_keywords)
    
    # Créer une introduction claire
    answer = f"<h3>Question :</h3>\n<p>{query}</p>\n\n<h3>Réponse :</h3>\n\n"
    
    # Organiser les résultats pour former une réponse cohérente et éviter les répétitions
    seen_text = set()  # Pour suivre les textes déjà inclus
    content_by_source = {}
    
    for chunk in results:
        filename = os.path.basename(chunk.file_path)
        if filename not in content_by_source:
            content_by_source[filename] = []
        
        # Vérifier si ce texte (ou un texte très similaire) a déjà été inclus
        # Simplification du texte pour la comparaison
        simplified_text = re.sub(r'\s+', ' ', chunk.text).strip()
        
        # Si ce texte est nouveau (pas encore vu), l'ajouter
        if simplified_text not in seen_text:
            # Formater le texte pour mettre en évidence les équations mathématiques
            formatted_text = format_math_equations(chunk.text)
            content_by_source[filename].append({
                "text": formatted_text,
                "page": chunk.page_num + 1,
                "has_formulas": chunk.has_formulas
            })
            
            # Ajouter à l'ensemble des textes vus
            seen_text.add(simplified_text)
    
    # Construction de la réponse finale
    if is_exercise:
        answer += "<div class='section-title'>Démarche de résolution :</div>\n"
    
    # Intégrer le contenu des sources dans la réponse sans répétitions
    for filename, contents in content_by_source.items():
        if contents:  # Vérifier s'il y a du contenu à afficher pour cette source
            answer += f"<div class='source-text'>Information extraite de : {filename}</div>\n\n"
            
            # Grouper le contenu par type (formules vs autre contenu)
            formulas = []
            other_content = []
            
            for content in contents:
                if is_exercise and content["has_formulas"]:
                    formulas.append(content)
                else:
                    other_content.append(content)
            
            # Afficher d'abord le contenu non-formule
            for content in other_content:
                answer += f"<div class='content-section'><p>Page {content['page']}</p>{content['text']}</div>\n\n"
            
            # Afficher ensuite les formules, si présentes
            if formulas:
                answer += "<div class='section-title'>Formules applicables :</div>\n"
                for formula in formulas:
                    answer += f"<div class='formula-section'><p>Page {formula['page']}</p>{formula['text']}</div>\n\n"
    
    # Ajouter une conclusion
    if is_exercise:
        answer += "<div class='section-title'>Conseils pour la résolution :</div>\n"
        answer += "<p>1. Identifiez correctement toutes les variables du problème.<br>" \
                 "2. Appliquez les formules appropriées extraites ci-dessus.<br>" \
                 "3. Vérifiez que vos unités sont cohérentes tout au long du calcul.<br>" \
                 "4. N'oubliez pas de vérifier que votre résultat est dans un ordre de grandeur raisonnable.</p>\n"
    
    return answer

# Ajouter une fonction pour traiter les questions soumises
def process_question(question_text):
    if question_text and chunks and model and index:
        # Affecter un numéro à la question
        current_question = st.session_state.question_counter
        st.session_state.question_counter += 1
        
        # Ajouter la question à l'historique
        st.session_state.chat_history.append({
            "role": "user", 
            "content": question_text,
            "number": current_question
        })
        
        # Rechercher les informations pertinentes
        with st.spinner("Recherche en cours..."):
            results = search_documents(question_text, chunks, model, index, k=5)
            
            # Formater la réponse
            is_exercise = 'is_exercise' in st.session_state and st.session_state.is_exercise
            answer = format_answer(results, question_text, is_exercise)
            
            # Réinitialiser le flag d'exercice
            if 'is_exercise' in st.session_state:
                st.session_state.is_exercise = False
            
            # Ajouter les images à la réponse
            images = []
            for chunk in results:
                if chunk.images:
                    images.extend(chunk.images)
            
            # Ajouter la réponse à l'historique
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "images": images[:5],
                "number": current_question
            })
        
        # Force la page à se recharger (au lieu de modifier st.session_state.user_input)
        st.rerun()

# Ajout d'une fonction pour traiter l'image téléchargée
def process_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        try:
            # Convertir le fichier en image PIL
            image = Image.open(uploaded_file)
            
            # Extraire le texte de l'image
            text = extract_text_from_image(image)
            
            if text and len(text) > 5:
                # Définir qu'il s'agit d'un exercice si cela provient d'une image
                st.session_state.is_exercise = True
                
                # Traiter la question extraite
                process_question(text)
                return True
            else:
                st.error("Aucun texte n'a pu être extrait de cette image ou le texte est trop court.")
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image: {str(e)}")
    return False

# Chemin vers les fichiers PDF
pdf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_files = [
    os.path.join(pdf_dir, "Chapitre 4 _ Fondations superficielles.pdf"),
    os.path.join(pdf_dir, "TD 4 _ EX 1 (semelle isolée _ charge centrée).pdf"),
    os.path.join(pdf_dir, "TD 4 _ EX 3 _ Semelles filantes _ charge centrée.pdf"),
    os.path.join(pdf_dir, "TD 4 _ EX 5 (semelle isolée _ charge excentrée).pdf")
]

# Styles CSS pour interface similaire à ChatGPT
st.markdown("""
<style>
    /* Style global */
    .stApp {
        background-color: #343541;
        color: #ECECF1;
    }
    
    /* Style pour la barre latérale */
    .sidebar .sidebar-content {
        background-color: #202123;
        color: #ECECF1;
    }
    
    /* Style pour le conteneur principal */
    .main .block-container {
        max-width: 1000px;
        padding-bottom: 100px; /* Espace pour la zone de saisie */
    }
    
    /* Style pour les messages de l'utilisateur */
    .user-message {
        background-color: #343541;
        padding: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        align-items: flex-start;
    }
    
    .user-icon {
        background-color: #5436DA;
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 16px;
        flex-shrink: 0;
    }
    
    /* Style pour les messages de l'assistant */
    .assistant-message {
        background-color: #444654;
        padding: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        align-items: flex-start;
    }
    
    .assistant-icon {
        background-color: #10A37F;
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 16px;
        flex-shrink: 0;
    }
    
    /* Style pour le contenu des messages */
    .message-content {
        width: 100%;
        overflow-wrap: break-word;
    }
    
    /* Style pour les images */
    .image-gallery {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 15px;
    }
    
    .gallery-image {
        max-width: 100%;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 10px;
    }
    
    /* Style pour la zone de saisie fixe en bas */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #343541;
        padding: 15px;
        display: flex;
        justify-content: center;
        border-top: 1px solid rgba(255,255,255,0.1);
        z-index: 1000;
    }
    
    .input-box {
        max-width: 900px;
        width: 100%;
        border-radius: 5px;
        background-color: #40414F;
        padding: 10px;
        display: flex;
        align-items: center;
    }
    
    /* Style pour les équations mathématiques */
    .math-equation {
        background-color: rgba(255, 100, 100, 0.1);
        color: #ff9e9e;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff5555;
        font-family: monospace;
        font-weight: bold;
        margin: 10px 0;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    
    /* Style pour la source */
    .source-text {
        color: #999;
        font-size: 0.9em;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px dashed rgba(255,255,255,0.2);
    }
    
    /* Style pour les titres de section */
    .section-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #10A37F;
    }
    
    /* Cache le bouton d'upload par défaut */
    .stFileUploader {
        display: none;
    }
    
    /* Style pour le bouton d'upload personnalisé */
    .upload-btn {
        background: none;
        border: none;
        color: #ECECF1;
        cursor: pointer;
        font-size: 24px;
        padding: 0 10px;
    }
    
    /* Nouveau style pour le bouton d'upload plus visible */
    .upload-button {
        background-color: #40414F;
        color: #ECECF1;
        border: 1px solid #555;
        border-radius: 5px;
        padding: 8px 12px;
        margin: 5px;
        cursor: pointer;
        font-size: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background-color 0.3s;
    }
    
    .upload-button:hover {
        background-color: #555;
    }
    
    .upload-button svg {
        margin-right: 5px;
        fill: #ECECF1;
    }
    
    /* Rendre le file uploader visible mais stylisé */
    .stFileUploader {
        display: block;
    }
    
    .stFileUploader > div {
        background-color: #40414F !important;
        border-radius: 5px;
    }
    
    .stFileUploader > div > div {
        color: #ECECF1 !important;
    }
    
    /* Style pour la barre de défilement */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #343541;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #10A37F;
    }
    
    /* Correction pour les labels - assurer que les labels sont correctement liés */
    label {
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Assurer que les labels et inputs sont correctement associés dans les composants générés */
    .element-container .stTextInput input,
    .element-container .stNumberInput input,
    .element-container .stTextArea textarea,
    .element-container .stSelectbox select,
    .element-container .stMultiselect select {
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Traitement des PDFs au démarrage
with st.spinner("Chargement des documents, veuillez patienter..."):
    chunks, model, index, processing_logs = process_pdfs(pdf_files)
    if not chunks or not model or not index:
        st.error("Erreur lors du chargement des documents. Vérifiez les logs ci-dessous.")
        for log in processing_logs:
            st.write(log)
        st.stop()

# Affichage du titre et de la description
st.title("Chatbot Fondations Superficielles 🏗️")
st.markdown("""
Ce chatbot vous aide à étudier les fondations superficielles. Il peut :
- Répondre à vos questions sur le cours
- Vous aider à résoudre des exercices
- Expliquer les concepts clés
""")

# Zone pour les messages
st.markdown("<div style='height: 400px; overflow-y: auto; padding-right: 10px;'>", unsafe_allow_html=True)

# Afficher l'historique des messages
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'''
        <div class="user-message">
            <div class="user-icon">U</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="assistant-message">
            <div class="assistant-icon">A</div>
            <div class="message-content">{message["content"]}
        ''', unsafe_allow_html=True)
        
        # Afficher les images si présentes
        if "images" in message and message["images"]:
            st.markdown('<div class="image-gallery">', unsafe_allow_html=True)
            for img in message["images"]:
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                st.markdown(f'''
                <img src="data:image/png;base64,{img_str}" class="gallery-image" alt="Image extraite du document">
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Zone de saisie fixe en bas
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<div class="input-box">', unsafe_allow_html=True)

# Créer des colonnes pour la zone de saisie et le bouton d'envoi
col1, col2, col3 = st.columns([8, 1, 1])

with col1:
    # Ajouter un attribut 'id' unique au champ de texte pour résoudre l'erreur de label
    user_input = st.text_input("Posez votre question ici", key="user_input", label_visibility="collapsed", 
                              placeholder="Posez votre question ici...", value="", 
                              help="Entrez votre question sur les fondations superficielles")

with col2:
    # Uploader une image avec un ID unique pour résoudre l'erreur de label
    uploaded_file = st.file_uploader("Télécharger une image d'un exercice", 
                                    type=["jpg", "jpeg", "png"], 
                                    key="file_uploader",
                                    label_visibility="collapsed",
                                    help="Téléchargez une image contenant un exercice")

with col3:
    # Utiliser le bouton pour soumettre la question avec un ID unique pour résoudre l'erreur de label
    submit_button = st.button("Envoyer", key="submit_button", help="Cliquer pour envoyer votre question")

st.markdown('</div></div>', unsafe_allow_html=True)

# Traitement des entrées
if uploaded_file:
    # Seulement traiter l'image si elle n'a pas déjà été traitée
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        process_uploaded_image(uploaded_file)

if user_input and submit_button:
    # Seulement traiter la question si elle n'est pas vide et n'a pas déjà été traitée
    if user_input.strip() and ('last_question' not in st.session_state or st.session_state.last_question != user_input):
        st.session_state.last_question = user_input
        process_question(user_input)
