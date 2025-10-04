"""
Módulo de análise de expressões faciais
"""

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import google.generativeai as genai
from datetime import datetime
import json
from collections import deque
import time


class FacialExpressionAnalyzer:
    """Analisador de expressões faciais em tempo real"""
    
    def __init__(self, gemini_api_key=None):
        """
        Inicializa o analisador
        
        Args:
            gemini_api_key: Chave API do Google Gemini (opcional)
        """
        # MediaPipe para detecção de face
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        
        # Configuração Gemini
        self.model = None
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Gemini AI configurado")
            except Exception as e:
                print(f"⚠️  Gemini não configurado: {e}")
        
        # Armazenamento
        self.emotion_history = deque(maxlen=100)
        self.analysis_results = []
        
    def detect_face(self, frame):
        """
        Detecta faces no frame
        
        Returns:
            list: Coordenadas das faces [(x, y, w, h), ...]
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        
        return faces
    
    def analyze_emotion(self, frame):
        """
        Analisa emoções usando DeepFace
        
        Returns:
            dict: {'emotions': {...}, 'dominant': str, 'timestamp': str}
        """
        try:
            analysis = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            return {
                'emotions': analysis.get('emotion', {}),
                'dominant': analysis.get('dominant_emotion', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Erro na análise: {e}")
            return None
    
    def process_video_stream(self, duration=30, question=""):
        """
        Processa stream de vídeo da webcam
        
        Args:
            duration: Duração em segundos
            question: Pergunta sendo respondida
            
        Returns:
            list: Lista de análises de emoções
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("❌ Não foi possível acessar a webcam")
        
        start_time = time.time()
        frame_count = 0
        session_emotions = []
        
        print(f"\n🎥 Gravando: '{question}'")
        print(f"⏱️  Duração: {duration}s | Pressione 'q' para parar\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            # Detecta faces
            faces = self.detect_face(frame)
            
            # Analisa emoções (a cada 10 frames)
            if frame_count % 10 == 0 and faces:
                emotion_data = self.analyze_emotion(frame)
                if emotion_data:
                    session_emotions.append(emotion_data)
                    self.emotion_history.append(emotion_data)
            
            # Desenha retângulos
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Exibe emoção atual
            if session_emotions:
                emotion = session_emotions[-1]['dominant']
                cv2.putText(frame, f"Emocao: {emotion}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2)
            
            # Timer
            remaining = int(duration - elapsed)
            cv2.putText(frame, f"Tempo: {remaining}s", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2)
            
            cv2.imshow('Analise Facial - RH', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Armazena resultado
        if session_emotions:
            self.analysis_results.append({
                'question': question,
                'emotions': session_emotions,
                'duration': elapsed,
                'timestamp': datetime.now().isoformat()
            })
        
        return session_emotions
    
    def generate_emotion_summary(self, emotions_list):
        """Gera resumo estatístico das emoções"""
        if not emotions_list:
            return {}
        
        emotion_counts = {}
        for entry in emotions_list:
            for emotion, score in entry['emotions'].items():
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = []
                emotion_counts[emotion].append(score)
        
        summary = {}
        for emotion, scores in emotion_counts.items():
            summary[emotion] = {
                'media': round(np.mean(scores), 2),
                'max': round(np.max(scores), 2),
                'min': round(np.min(scores), 2)
            }
        
        return summary
    
    def generate_ai_insights(self, question, emotions_summary):
        """Gera insights usando Gemini AI"""
        if not self.model:
            return "⚠️  API Gemini não configurada"
        
        prompt = f"""
        Analise o resultado de reconhecimento facial durante entrevista:
        
        Pergunta: {question}
        
        Emoções detectadas (% média):
        {json.dumps(emotions_summary, indent=2, ensure_ascii=False)}
        
        Como especialista em RH, forneça:
        1. Interpretação das emoções dominantes
        2. Insights sobre o estado emocional
        3. Pontos de atenção para o recrutador
        4. Sugestões de perguntas de follow-up
        
        Seja profissional e evite conclusões definitivas.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Erro ao gerar insights: {e}"
    
    def export_report(self, filename="/app/reports/analise_candidato.json"):
        """Exporta relatório completo em JSON"""
        report = {
            'data_analise': datetime.now().isoformat(),
            'total_perguntas': len(self.analysis_results),
            'sessoes': []
        }
        
        for session in self.analysis_results:
            emotions_summary = self.generate_emotion_summary(session['emotions'])
            
            session_report = {
                'pergunta': session['question'],
                'duracao_segundos': round(session['duration'], 2),
                'timestamp': session['timestamp'],
                'resumo_emocoes': emotions_summary,
                'total_frames_analisados': len(session['emotions'])
            }
            
            # Gera insights com IA
            if self.model:
                session_report['insights_ia'] = self.generate_ai_insights(
                    session['question'], 
                    emotions_summary
                )
            
            report['sessoes'].append(session_report)
        
        # Salva arquivo
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Relatório exportado: {filename}")
        return report
