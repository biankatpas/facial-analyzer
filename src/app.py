"""
Streamlit Interface - Facial Expression Analysis
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import requests
import os
from io import BytesIO
import tempfile
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Facial Expression Analysis",
    page_icon="🎯",
    layout="wide"
)

# URL da API
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Inicializar session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.logs = []
    st.session_state.all_sessions = []
    st.session_state.current_session_id = None
    st.session_state.video_analyzed = False

# Perguntas
QUESTIONS = [
    "Tell me about your previous professional experience",
    "Why do you want to work at our company?",
    "What was your biggest professional challenge?"
]

# Funções para comunicação com API
def start_session(question: str):
    """Inicia nova sessão na API"""
    try:
        response = requests.post(
            f"{API_URL}/session/start",
            params={"question": question}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error starting session: {e}")
        return None


def analyze_frame(image_bytes, session_id=None):
    """Envia frame para análise na API"""
    try:
        files = {'file': ('frame.jpg', BytesIO(image_bytes), 'image/jpeg')}
        
        params = {}
        if session_id:
            params['session_id'] = session_id
        
        response = requests.post(
            f"{API_URL}/analyze/frame",
            files=files,
            params=params
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def get_session_summary(session_id: str):
    """Obtém resumo da sessão"""
    try:
        response = requests.get(f"{API_URL}/session/{session_id}/summary")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting summary: {e}")
        return None


def generate_insights(session_id: str):
    """Gera insights com IA"""
    try:
        response = requests.post(f"{API_URL}/session/{session_id}/insights")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return None


def process_video(video_file, session_id):
    """Processa vídeo frame a frame"""
    # Salvar vídeo temporariamente
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    tfile.close()
    
    # Abrir vídeo
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    st.info(f"📹 Video info: {total_frames} frames, {fps} FPS, ~{total_frames//fps}s duration")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    emotions_timeline = []
    
    # Processar frames (1 a cada 10 frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analisar a cada 10 frames
        if frame_count % 10 == 0:
            # Converter frame para JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()
            
            # Analisar
            emotion_data = analyze_frame(img_bytes, session_id)
            
            if emotion_data:
                # Adicionar timestamp
                timestamp = frame_count / fps
                emotion_data['timestamp'] = timestamp
                emotion_data['frame'] = frame_count
                emotions_timeline.append(emotion_data)
                
                # Log
                log_msg = f"[{int(timestamp)}s] {emotion_data['dominant']} ({emotion_data['emotions'][emotion_data['dominant']]:.1f}%)"
                st.session_state.logs.append(log_msg)
        
        # Atualizar progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}...")
    
    cap.release()
    os.unlink(video_path)
    
    progress_bar.empty()
    status_text.empty()
    
    return emotions_timeline


# Título
st.title("🎯 Facial Expression Analysis for HR Recruiters")
st.markdown("---")

# Layout em colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Candidate Video Upload")
    
    # Mostrar pergunta atual
    if st.session_state.current_question < len(QUESTIONS):
        st.info(f"**Question {st.session_state.current_question + 1}/{len(QUESTIONS)}**")
        st.markdown(f"### {QUESTIONS[st.session_state.current_question]}")
        
        # Upload de vídeo
        uploaded_video = st.file_uploader(
            "Upload candidate's video response to this question",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key=f"upload_{st.session_state.current_question}"
        )
        
        if uploaded_video is not None:
            # Mostrar preview do vídeo
            st.video(uploaded_video)
            
            # Botão para analisar
            if st.button("🔍 Analyze Candidate's Expression", key=f"analyze_{st.session_state.current_question}"):
                # Iniciar sessão
                if not st.session_state.current_session_id:
                    question = QUESTIONS[st.session_state.current_question]
                    session_data = start_session(question)
                    
                    if session_data:
                        st.session_state.current_session_id = session_data['session_id']
                        st.session_state.logs.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Started analysis for question {st.session_state.current_question + 1}"
                        )
                
                # Processar vídeo
                with st.spinner("Analyzing video frames..."):
                    emotions_timeline = process_video(uploaded_video, st.session_state.current_session_id)
                    
                    if emotions_timeline:
                        st.success(f"✅ Analyzed {len(emotions_timeline)} frames successfully!")
                        st.session_state.video_analyzed = True
                        
                        # Mostrar timeline de emoções
                        st.markdown("#### Emotion Timeline")
                        
                        # Criar DataFrame para visualização
                        df_emotions = []
                        for entry in emotions_timeline:
                            row = {'Timestamp (s)': entry['timestamp']}
                            row.update(entry['emotions'])
                            df_emotions.append(row)
                        
                        df = pd.DataFrame(df_emotions)
                        
                        # Gráfico de linha
                        st.line_chart(df.set_index('Timestamp (s)'))
                        
                        # Estatísticas
                        st.markdown("#### Emotion Statistics")
                        emotions_only = df.drop('Timestamp (s)', axis=1)
                        st.dataframe(emotions_only.describe())
                        
                        # TOP 3 EMOÇÕES COM MEDALHAS
                        st.markdown("#### 🏆 Top 3 Emotions Detected")
                        
                        # Calcular média de cada emoção
                        emotion_averages = emotions_only.mean().sort_values(ascending=False)
                        medals = ['🥇', '🥈', '🥉']
                        
                        top3_cols = st.columns(3)
                        for idx, (emotion, avg_score) in enumerate(emotion_averages.head(3).items()):
                            with top3_cols[idx]:
                                st.markdown(f"### {medals[idx]} {emotion.upper()}")
                                st.metric(
                                    label="Average Score",
                                    value=f"{avg_score:.1f}%",
                                    delta=f"Peak: {emotions_only[emotion].max():.1f}%"
                                )
            
            # Botão para próxima pergunta
            if st.session_state.video_analyzed:
                if st.button("➡️ Next Question", key=f"next_{st.session_state.current_question}"):
                    # Salvar session_id
                    st.session_state.all_sessions.append(st.session_state.current_session_id)
                    st.session_state.current_session_id = None
                    st.session_state.video_analyzed = False
                    
                    # Próxima pergunta
                    st.session_state.current_question += 1
                    st.session_state.logs.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Moving to question {st.session_state.current_question + 1}"
                    )
                    st.rerun()
        
        else:
            # Placeholder
            st.info("📹 Please upload the candidate's video response to analyze their facial expressions")
    
    else:
        st.success("✅ All questions completed!")
        st.balloons()

with col2:
    st.subheader("📋 Interview Progress")
    
    # Progress bar
    progress = st.session_state.current_question / len(QUESTIONS)
    st.progress(progress)
    st.caption(f"Question {st.session_state.current_question}/{len(QUESTIONS)}")
    
    st.markdown("---")
    
    # Botão reset
    if st.button("🔄 Reset Interview"):
        st.session_state.current_question = 0
        st.session_state.logs = []
        st.session_state.all_sessions = []
        st.session_state.current_session_id = None
        st.session_state.video_analyzed = False
        st.rerun()
    
    st.markdown("---")
    
    # Logs em tempo real
    st.subheader("📊 Candidate Analysis Logs")
    log_container = st.container()
    
    with log_container:
        if st.session_state.logs:
            # Mostrar logs em ordem reversa (mais recente primeiro)
            for log in reversed(st.session_state.logs[-20:]):
                st.text(log)
        else:
            st.text("No analysis yet. Upload candidate's video to begin.")

# Resultados finais
if st.session_state.current_question >= len(QUESTIONS) and st.session_state.all_sessions:
    st.markdown("---")
    st.subheader("📈 Candidate Evaluation Results")
    
    # Mostrar resultados de cada sessão
    for i, session_id in enumerate(st.session_state.all_sessions, 1):
        with st.expander(f"Question {i}: {QUESTIONS[i-1]}", expanded=(i==1)):
            summary_data = get_session_summary(session_id)
            
            if summary_data and summary_data.get('summary'):
                summary = summary_data['summary']
                
                # TOP 3 EMOÇÕES COM MEDALHAS
                st.markdown("#### 🏆 Top 3 Candidate Emotions")
                
                # Mostrar top 3 emoções com medalhas
                sorted_emotions = sorted(
                    summary.items(), 
                    key=lambda x: x[1]['media'], 
                    reverse=True
                )
                
                medals = ['🥇', '🥈', '🥉']
                cols = st.columns(3)
                
                for idx, (emotion, stats) in enumerate(sorted_emotions[:3]):
                    with cols[idx]:
                        st.markdown(f"### {medals[idx]} {emotion.upper()}")
                        st.metric(
                            label="Average Score",
                            value=f"{stats['media']:.1f}%",
                            delta=f"Range: {stats['min']:.1f}% - {stats['max']:.1f}%"
                        )
                
                # Botão para gerar insights
                if st.button(f"🤖 Generate AI Insights", key=f"insights_{i}"):
                    with st.spinner("Generating insights..."):
                        insights_data = generate_insights(session_id)
                        
                        if insights_data:
                            st.markdown("### AI Insights")
                            st.markdown(insights_data.get('insights', 'No insights available'))
            else:
                st.warning("No data available for this question")
    
    # Botão para exportar relatório completo
    st.markdown("---")
    if st.button("💾 Export Complete Report"):
        try:
            response = requests.post(f"{API_URL}/export/report")
            response.raise_for_status()
            st.success("✅ Report exported successfully!")
            st.json(response.json())
        except Exception as e:
            st.error(f"❌ Error exporting report: {e}")

# Footer com status da API
st.markdown("---")
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    status = "🟢 Online" if health['status'] == 'ok' else "🔴 Offline"
    st.caption(f"API Status: {status} | Gemini AI: {'✅ Configured' if health.get('gemini_configured') else '⚠️ Not configured'}")
except:
    st.caption("API Status: 🔴 Offline")