import json
import os
import logging
from datetime import datetime
from typing import Dict, List

import boto3
import pymysql
import openai

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수
DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# OpenAI 클라이언트 초기화
openai.api_key = OPENAI_API_KEY

# 감정 라벨
EMOTION_LABELS = [
    "기쁨", "슬픔", "분노", "불안", "설렘",
    "지루함", "외로움", "만족", "실망", "무기력"
]


def get_db_connection():
    """데이터베이스 연결"""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def get_today_diaries(connection, user_id: int, target_date: str) -> List[Dict]:
    """오늘 하루 작성된 일기들 조회"""
    with connection.cursor() as cursor:
        query = """
        SELECT id, content, emotion_vector, summary, tags, primary_emotion, created_at
        FROM diaries
        WHERE user_id = %s
            AND DATE(created_at) = %s
            AND analysis_status = 'completed'
        ORDER BY created_at DESC
        """
        cursor.execute(query, (user_id, target_date))
        return cursor.fetchall()


def get_or_create_session(connection, user_id: int, session_date: str) -> int:
    """대화 세션 조회 또는 생성"""
    with connection.cursor() as cursor:
        # 기존 세션 조회
        cursor.execute(
            "SELECT id FROM conversation_sessions WHERE user_id = %s AND session_date = %s",
            (user_id, session_date)
        )
        session = cursor.fetchone()

        if session:
            return session['id']

        # 새 세션 생성
        cursor.execute(
            "INSERT INTO conversation_sessions (user_id, session_date) VALUES (%s, %s)",
            (user_id, session_date)
        )
        connection.commit()
        return cursor.lastrowid


def get_conversation_history(connection, session_id: int, limit: int = 10) -> List[Dict]:
    """대화 히스토리 조회"""
    with connection.cursor() as cursor:
        query = """
        SELECT message_type, content, created_at
        FROM conversation_messages
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        cursor.execute(query, (session_id, limit))
        messages = cursor.fetchall()
        return list(reversed(messages))  # 시간순 정렬


def save_message(connection, session_id: int, user_id: int, message_type: str, content: str):
    """메시지 저장"""
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO conversation_messages
               (session_id, user_id, message_type, content)
               VALUES (%s, %s, %s, %s)""",
            (session_id, user_id, message_type, content)
        )
        connection.commit()


def format_emotion_vector(emotion_vector: List[float]) -> str:
    """감정 벡터를 텍스트로 포매팅"""
    if not emotion_vector or len(emotion_vector) != 10:
        return "감정 데이터 없음"

    emotions = []
    for i, value in enumerate(emotion_vector):
        if value > 0.1:  # 10% 이상인 감정만 표시
            emotions.append(f"{EMOTION_LABELS[i]}: {value:.1f}")

    return ", ".join(emotions) if emotions else "중성적 감정"


def format_diaries_for_prompt(diaries: List[Dict]) -> str:
    """일기 데이터를 프롬프트용으로 포매팅"""
    if not diaries:
        return "오늘 작성된 일기가 없습니다."

    diary_texts = []
    total_emotion_vector = [0.0] * 10

    for diary in diaries:
        # 일기 내용
        content = diary['content']
        content_preview = content[:200] + "..." if len(content) > 200 else content
        diary_texts.append(f"- {content_preview}")

        # 감정 벡터 합계
        if diary['emotion_vector']:
            emotion_vector = diary['emotion_vector']
            if isinstance(emotion_vector, str):
                emotion_vector = json.loads(emotion_vector)
            for i, value in enumerate(emotion_vector):
                total_emotion_vector[i] += float(value)

    # 평균 감정 벡터 계산
    if len(diaries) > 0:
        avg_emotion_vector = [v / len(diaries) for v in total_emotion_vector]
    else:
        avg_emotion_vector = [0.0] * 10

    diary_text = "\n".join(diary_texts)
    emotion_text = format_emotion_vector(avg_emotion_vector)

    return f"일기 내용:\n{diary_text}\n\n감정 분석:\n{emotion_text}"


def format_conversation_history(history: List[Dict]) -> str:
    """대화 히스토리를 프롬프트용으로 포매팅"""
    if not history:
        return ""

    history_text = []
    for msg in history[-6:]:  # 최근 6개 메시지만
        role = "사용자" if msg['message_type'] == 'user' else "AI"
        content = msg['content']
        content_preview = content[:150] + "..." if len(content) > 150 else content
        history_text.append(f"{role}: {content_preview}")

    return "\n".join(history_text)


def create_conversation_prompt(user_input: str, diaries_info: str, conversation_history: str) -> str:
    """대화용 프롬프트 생성"""
    base_prompt = """
[역할]
당신은 사용자의 감정에 귀 기울이는 정서적 동행자이자, 공감과 위로, 격려를 제공하는 AI입니다. \
사용자가 하루 동안 느낀 다양한 감정을 음성으로 털어놓았고, 그 내용을 텍스트로 변환하여 당신에게 전달되었습니다. \
또한, 사용자가 기록한 오늘 하루의 일기와 감정 분석 결과(10가지 감정에 대한 정량적 벡터)도 함께 제공됩니다.

[목표]
* 사용자의 **음성 기반 감정 표현**을 분석하고,
* 해당 사용자가 기록한 **오늘 하루의 일기 전체 내용**과
* AI 감정 분석 시스템이 계산한 **감정 벡터 (10차원)** 를 함께 고려하여
* 사용자에게 **정서적 대화, 위로, 공감, 혹은 감정의 원인 탐색**을 제안하는 피드백을 제공하세요.

[대답 형식]
* 대화체 2~3문장으로 구성
* 반드시 사용자가 표현한 감정을 먼저 "들어주는" 태도를 취하세요.
* 이후, 그 감정의 원인을 유추하거나, 사용자의 상황을 다시 되짚어주며 위로하거나, \
스스로 감정을 이해할 수 있도록 도와주세요.

[제약 사항]
* 사용자의 감정 벡터를 참고하여 가장 강하게 느껴졌던 감정을 **명시적으로 반영**하세요.
* 일기 내용에서 발생한 주요 사건을 **구체적으로 언급**하세요.
* 지나친 낙관이나 가벼운 말투는 지양하고, 진정성 있는 상담사 같은 톤을 유지하세요.
* 문장은 자연스러운 구어체로 작성하며, 단답이나 형식적 말투를 피하세요.
"""

    history_section = f"\n[이전 대화 내역]\n{conversation_history}\n" if conversation_history else ""

    prompt = f"""{base_prompt}

{history_section}
[오늘의 일기 및 감정 데이터]
{diaries_info}

[사용자 입력]
voice_input: "{user_input}"

위 정보를 바탕으로 사용자에게 정서적 대화를 제공해주세요. JSON 형식이 아닌 자연스러운 대화로 응답해주세요.
"""

    return prompt


def call_openai_api(prompt: str) -> str:
    """OpenAI API 호출"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 전문적이고 공감적인 심리 상담사입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("OpenAI API 호출 오류: %s", str(e))
        return "죄송합니다. 현재 대화 서비스에 일시적인 문제가 있습니다. 잠시 후 다시 시도해주세요."


def lambda_handler(event, _context):
    """Lambda 메인 핸들러"""
    try:
        # CORS 헤더
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }

        # OPTIONS 요청 처리
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({'message': 'OK'})
            }

        # 요청 본문 파싱
        body = json.loads(event.get('body', '{}'))
        user_id = body.get('userId')
        user_input = body.get('text', '').strip()
        target_date = body.get('date', datetime.now().strftime('%Y-%m-%d'))

        # 입력 검증
        if not user_id:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'userId는 필수입니다.'}, ensure_ascii=False)
            }

        if not user_input:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'text는 필수입니다.'}, ensure_ascii=False)
            }

        logger.info("감정 대화 요청: user_id=%s, date=%s", user_id, target_date)

        # DB 연결
        connection = get_db_connection()

        try:
            # 1. 오늘의 일기 조회
            today_diaries = get_today_diaries(connection, user_id, target_date)
            diaries_info = format_diaries_for_prompt(today_diaries)

            # 2. 대화 세션 조회/생성
            session_id = get_or_create_session(connection, user_id, target_date)

            # 3. 대화 히스토리 조회
            conversation_history = get_conversation_history(connection, session_id)
            history_text = format_conversation_history(conversation_history)

            # 4. 사용자 메시지 저장
            save_message(connection, session_id, user_id, 'user', user_input)

            # 5. 프롬프트 생성
            prompt = create_conversation_prompt(user_input, diaries_info, history_text)

            # 6. OpenAI API 호출
            ai_response = call_openai_api(prompt)

            # 7. AI 응답 저장
            save_message(connection, session_id, user_id, 'assistant', ai_response)

            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'success': True,
                    'response': ai_response,
                    'session_id': session_id,
                    'diaries_count': len(today_diaries)
                }, ensure_ascii=False)
            }

        finally:
            connection.close()

    except Exception as e:
        logger.error("Lambda 실행 오류: %s", str(e))
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json; charset=utf-8',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': '서버 내부 오류가 발생했습니다.'
            }, ensure_ascii=False)
        }