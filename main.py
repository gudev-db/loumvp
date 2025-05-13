import os
import requests
import streamlit as st
from dotenv import load_dotenv
import openai
from typing import List, Dict

# Load environment variables
load_dotenv()

# Configurations
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"  # Changed from gpt-4o-mini which doesn't exist
COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION")
NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE", "default_keyspace")
EMBEDDING_DIMENSION = 1536
ASTRA_DB_API_BASE = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

class AstraDBClient:
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_BASE}/api/json/v1/{NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_TOKEN,
            "Accept": "application/json"
        }
    
    def vector_search(self, collection: str, vector: List[float], limit: int = 3) -> List[Dict]:
        """Perform vector similarity search"""
        url = f"{self.base_url}/{collection}"
        payload = {
            "find": {
                "sort": {"$vector": vector},
                "options": {"limit": limit}
            }
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()["data"]["documents"]
        except Exception as e:
            st.error(f"Vector search error: {str(e)}")
            st.error(f"API response: {response.text if 'response' in locals() else 'N/A'}")
            return []

def get_embedding(text: str) -> List[float]:
    """Get text embedding using OpenAI"""
    try:
        response = openai.Embedding.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return []

def generate_response(query: str, context: str, mental_state: dict = None) -> str:
    """Generate response using OpenAI chat model"""
    if not context and not mental_state:
        return "Não encontrei informações relevantes para responder sua pergunta."
    
    # Base system prompt
    system_prompt = """
    Você é o co-piloto da agência de marketing Macfor e está aqui para prover informações sobre a empresa Holambra Cooperativa. 
    Seja atencioso e ajude o usuário a encontrar o que quer.
    """
    
    # Add mental state information if available
    if mental_state:
        system_prompt += f"""
        \n\nINFORMAÇÕES SOBRE O ESTADO MENTAL DO USUÁRIO:
        - Nível de Depressão: {mental_state['depression']}
        - Nível de Estresse: {mental_state['stress']}
        - Nível de Ansiedade: {mental_state['anxiety']}
        
        Adapte sua comunicação conforme necessário para ser mais empático e compreensivo.
        """
    
    prompt = f"""Responda baseado no contexto abaixo:
    
    Contexto:
    {context}
    
    Pergunta: {query}
    Resposta:"""
    
    try:
        response = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

def dsm5_assessment():
    """DSM-5 Depression, Anxiety, and Stress Assessment"""
    st.header("Avaliação DSM-5")
    st.write("Por favor, responda às perguntas abaixo para avaliar seus níveis de depressão, estresse e ansiedade.")
    
    questions = [
        "1. Me sinto triste ou deprimido",
        "2. Sinto falta de interesse ou prazer nas coisas",
        "3. Me sinto tenso ou nervoso",
        "4. Preocupo-me excessivamente com coisas",
        "5. Sinto dificuldade em relaxar",
        "6. Me sinto irritado ou agitado",
        "7. Sinto que tudo exige grande esforço",
        "8. Tenho dificuldade em me concentrar",
        "9. Me sinto cansado ou sem energia",
        "10. Tenho pensamentos ruins sobre mim mesmo"
    ]
    
    scores = {"depression": 0, "anxiety": 0, "stress": 0}
    
    for i, question in enumerate(questions):
        score = st.radio(
            question,
            options=["Nenhum", "Leve", "Moderado", "Severo", "Muito severo"],
            key=f"q_{i}",
            horizontal=True
        )
        
        # Map answer to score (0-4)
        value = ["Nenhum", "Leve", "Moderado", "Severo", "Muito severo"].index(score)
        
        # Distribute scores to different categories based on question type
        if i in [0, 1, 6, 8, 9]:  # Depression-related questions
            scores["depression"] += value
        elif i in [2, 3, 4, 5]:   # Anxiety-related questions
            scores["anxiety"] += value
        else:                       # Stress-related questions
            scores["stress"] += value
    
    if st.button("Calcular resultados"):
        # Calculate severity levels (max score per category is 16)
        depression_level = "Normal"
        if scores["depression"] > 12:
            depression_level = "Severo"
        elif scores["depression"] > 8:
            depression_level = "Moderado"
        elif scores["depression"] > 4:
            depression_level = "Leve"
        
        anxiety_level = "Normal"
        if scores["anxiety"] > 12:
            anxiety_level = "Severo"
        elif scores["anxiety"] > 8:
            anxiety_level = "Moderado"
        elif scores["anxiety"] > 4:
            anxiety_level = "Leve"
            
        stress_level = "Normal"
        if scores["stress"] > 12:
            stress_level = "Severo"
        elif scores["stress"] > 8:
            stress_level = "Moderado"
        elif scores["stress"] > 4:
            stress_level = "Leve"
        
        st.session_state.mental_state = {
            "depression": depression_level,
            "anxiety": anxiety_level,
            "stress": stress_level
        }
        
        st.write(f"**Nível de Depressão:** {depression_level}")
        st.write(f"**Nível de Ansiedade:** {anxiety_level}")
        st.write(f"**Nível de Estresse:** {stress_level}")
        
        if any(level != "Normal" for level in [depression_level, anxiety_level, stress_level]):
            st.info("Com base nos seus resultados, você pode querer conversar com nosso assistente sobre como está se sentindo.")

def email_generator():
    """Email generator tab"""
    st.header("Gerador de Emails")
    
    with st.form("email_form"):
        recipient = st.text_input("Destinatário:")
        subject = st.text_input("Assunto:")
        notes = st.text_area("Observações/Conteúdo desejado:", height=200)
        
        submitted = st.form_submit_button("Gerar Email")
        
        if submitted:
            try:
                response = openai.ChatCompletion.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "Você é um assistente que ajuda a escrever emails profissionais."},
                        {"role": "user", "content": f"""
                        Escreva um email com as seguintes especificações:
                        - Destinatário: {recipient}
                        - Assunto: {subject}
                        - Observações: {notes}
                        
                        Por favor, formate o email adequadamente com saudações e despedida.
                        """}
                    ],
                    temperature=0.7
                )
                
                generated_email = response["choices"][0]["message"]["content"]
                st.text_area("Email gerado:", value=generated_email, height=300)
                
                # Add download button
                st.download_button(
                    label="Baixar Email",
                    data=generated_email,
                    file_name=f"email_{subject.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Erro ao gerar email: {str(e)}")

def document_generator():
    """Document generator tab"""
    st.header("Gerador de Documentos")
    
    with st.form("document_form"):
        doc_type = st.selectbox(
            "Tipo de Documento:",
            ["Relatório", "Proposta", "Contrato", "Memorando", "Outro"]
        )
        guidelines = st.text_area("Diretrizes/Conteúdo desejado:", height=300)
        
        submitted = st.form_submit_button("Gerar Documento")
        
        if submitted:
            try:
                response = openai.ChatCompletion.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": f"Você é um assistente que ajuda a escrever documentos do tipo {doc_type}."},
                        {"role": "user", "content": f"""
                        Escreva um documento com as seguintes especificações:
                        - Tipo: {doc_type}
                        - Diretrizes: {guidelines}
                        
                        Por favor, formate o documento adequadamente com títulos e seções quando necessário.
                        """}
                    ],
                    temperature=0.7
                )
                
                generated_doc = response["choices"][0]["message"]["content"]
                st.text_area("Documento gerado:", value=generated_doc, height=400)
                
                # Add download button
                st.download_button(
                    label="Baixar Documento",
                    data=generated_doc,
                    file_name=f"{doc_type.lower()}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Erro ao gerar documento: {str(e)}")

def rag_chat():
    """RAG Chat tab"""
    st.title("🤖 NeIA - Chat")
    st.write("Conectado à base de dados")
    
    # Initialize Astra DB client
    astra_client = AstraDBClient()
    
    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Process new input
    if prompt := st.chat_input("Digite sua mensagem..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get embedding and search Astra DB
        embedding = get_embedding(prompt)
        if embedding:
            results = astra_client.vector_search(COLLECTION_NAME, embedding)
            context = "\n".join([str(doc) for doc in results])
            
            # Get mental state if available
            mental_state = st.session_state.get("mental_state")
            
            # Generate response
            response = generate_response(prompt, context, mental_state)
            
            # Add response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

def main():
    st.sidebar.title("NeIA - Navegação")
    tabs = {
        "Chat RAG": rag_chat,
        "Avaliação DSM-5": dsm5_assessment,
        "Gerador de Emails": email_generator,
        "Gerador de Documentos": document_generator
    }
    
    selected_tab = st.sidebar.radio("Selecione uma aba:", list(tabs.keys()))
    
    # Display the selected tab
    tabs[selected_tab]()

if __name__ == "__main__":
    main()
