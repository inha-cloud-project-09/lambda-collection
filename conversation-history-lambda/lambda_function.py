import json
import os
import logging
from datetime import datetime, timedelta
import pymysql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수
DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']

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

def get_conversation_history(connection, user_id: int, session_date: str = None, limit: int = 50):
    """대화 히스토리 조회"""
    with connection.cursor() as cursor:
        if session_date:
            # 특정 날짜의 대화
            query = """
            SELECT cm.message_type, cm.content, cm.created_at, cs.session_date
            FROM conversation_messages cm
            JOIN conversation_sessions cs ON cm.session_id = cs.id
            WHERE cm.user_id = %s AND cs.session_date = %s
            ORDER BY cm.created_at ASC
            LIMIT %s
            """
            cursor.execute(query, (user_id, session_date, limit))
        else:
            # 최근 대화 (모든 날짜)
            query = """
            SELECT cm.message_type, cm.content, cm.created_at, cs.session_date
            FROM conversation_messages cm
            JOIN conversation_sessions cs ON cm.session_id = cs.id
            WHERE cm.user_id = %s
            ORDER BY cm.created_at DESC
            LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
        
        return cursor.fetchall()

def get_session_dates(connection, user_id: int, days: int = 30):
    """사용자의 대화 세션 날짜 목록 조회"""
    with connection.cursor() as cursor:
        query = """
        SELECT DISTINCT cs.session_date, COUNT(cm.id) as message_count
        FROM conversation_sessions cs
        LEFT JOIN conversation_messages cm ON cs.id = cm.session_id
        WHERE cs.user_id = %s 
            AND cs.session_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        GROUP BY cs.session_date
        ORDER BY cs.session_date DESC
        """
        cursor.execute(query, (user_id, days))
        return cursor.fetchall()

def lambda_handler(event, _context):
    """Lambda 메인 핸들러"""
    try:
        # CORS 헤더
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        
        # OPTIONS 요청 처리
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({'message': 'OK'})
            }
        
        # 경로 파라미터 및 쿼리 파라미터 추출
        path_params = event.get('pathParameters', {}) or {}
        query_params = event.get('queryStringParameters', {}) or {}
        
        user_id = path_params.get('userId')
        session_date = query_params.get('date')
        limit = int(query_params.get('limit', 50))
        action = query_params.get('action', 'messages')  # 'messages' or 'sessions'
        
        if not user_id:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'userId는 필수입니다.'}, ensure_ascii=False)
            }
        
        logger.info("대화 히스토리 조회 요청: user_id=%s, date=%s, action=%s", user_id, session_date, action)
        
        # DB 연결
        connection = get_db_connection()
        
        try:
            if action == 'sessions':
                # 세션 날짜 목록 조회
                days = int(query_params.get('days', 30))
                sessions = get_session_dates(connection, int(user_id), days)
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'success': True,
                        'sessions': [
                            {
                                'date': session['session_date'].strftime('%Y-%m-%d'),
                                'message_count': session['message_count']
                            }
                            for session in sessions
                        ]
                    }, ensure_ascii=False, default=str)
                }
            
            else:
                # 대화 메시지 조회
                messages = get_conversation_history(connection, int(user_id), session_date, limit)
                
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append({
                        'type': msg['message_type'],
                        'content': msg['content'],
                        'timestamp': msg['created_at'].isoformat(),
                        'date': msg['session_date'].strftime('%Y-%m-%d')
                    })
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'success': True,
                        'messages': formatted_messages
                    }, ensure_ascii=False, default=str)
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