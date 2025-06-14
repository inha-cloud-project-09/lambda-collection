import json
import boto3
import pymysql
import os
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS 클라이언트
sns_client = boto3.client('sns', region_name='us-east-1')

# 환경 변수
DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']

def get_db_connection():
    """데이터베이스 연결"""
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10
        )
        logger.info(" DB 연결 성공")
        return connection
    except Exception as e:
        logger.error(f" DB 연결 실패: {str(e)}")
        raise

def validate_request(body):
    """요청 데이터 검증 - title 제거"""
    errors = []
    
    if not body.get('content') or not body.get('content').strip():
        errors.append('content는 필수 항목입니다')
    
    if 'isPublic' not in body:
        errors.append('isPublic은 필수 항목입니다')
    
    if not body.get('userId'):
        errors.append('userId는 필수 항목입니다')
    
    # 길이 제한 검증
    if body.get('content') and len(body['content']) > 10000:
        errors.append('content는 10000자를 초과할 수 없습니다')
    
    return errors

def save_diary_to_db(connection, content, is_public, user_id):
    """일기를 DB에 저장 - title 필드 제거"""
    try:
        with connection.cursor() as cursor:
            # title 필드 제거된 쿼리
            insert_query = """
            INSERT INTO diaries (
                user_id, 
                content, 
                is_public, 
                analysis_status, 
                created_at
            ) VALUES (%s, %s, %s, 'pending', NOW())
            """
            
            cursor.execute(insert_query, (
                user_id,
                content.strip(),
                is_public
            ))
            
            diary_id = cursor.lastrowid
            connection.commit()
            
            logger.info(f"일기 저장 성공: diary_id={diary_id}, user_id={user_id}")
            return diary_id
            
    except Exception as e:
        connection.rollback()
        logger.error(f"일기 저장 실패: {str(e)}")
        raise

def publish_diary_created_event(diary_id, user_id):
    """SNS에 diary_created 이벤트 발행"""
    try:
        # SNS 메시지 구성
        message = {
            "diaryId": diary_id,
            "userId": user_id,
            "event": "diary_created"
        }
        
        # SNS 발행
        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(message),
            Subject='diary_created',
            MessageAttributes={
                'event_type': {
                    'DataType': 'String',
                    'StringValue': 'diary_created'
                }
            }
        )
        
        logger.info(f" SNS 발행 성공: diary_id={diary_id}, MessageId={response['MessageId']}")
        return True
        
    except Exception as e:
        logger.error(f" SNS 발행 실패: diary_id={diary_id}, error={str(e)}")
        # SNS 실패해도 일기 저장은 성공으로 처리
        return False

def get_saved_diary(connection, diary_id):
    """저장된 일기 정보 조회 - title 필드 제거"""
    try:
        with connection.cursor() as cursor:
            select_query = """
            SELECT 
                id, user_id, content, is_public,
                analysis_status, created_at
            FROM diaries 
            WHERE id = %s
            """
            
            cursor.execute(select_query, (diary_id,))
            result = cursor.fetchone()
            
            if result:
                result['created_at'] = result['created_at'].isoformat()
                
            return result
            
    except Exception as e:
        logger.error(f"일기 조회 실패: diary_id={diary_id}, error={str(e)}")
        return None

def lambda_handler(event, context):
    """Lambda 메인 핸들러 - title 처리 제거"""
    logger.info(f"일기 POST Lambda 시작: {json.dumps(event, ensure_ascii=False)}")
    
    connection = None
    
    try:
        # 1. 요청 파싱
        if 'body' not in event:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                },
                'body': json.dumps({
                    'success': False,
                    'message': '요청 본문이 없습니다'
                }, ensure_ascii=False)
            }
        
        body = json.loads(event['body'])
        logger.info(f"요청 데이터: {json.dumps(body, ensure_ascii=False)}")
        
        # 2. 요청 데이터 검증
        validation_errors = validate_request(body)
        if validation_errors:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'message': '요청 데이터가 올바르지 않습니다',
                    'errors': validation_errors
                }, ensure_ascii=False)
            }
        
        # 3. 데이터 추출 - title 제거
        content = body['content']
        is_public = body['isPublic']
        user_id = body['userId']
        
        # 4. DB 연결
        connection = get_db_connection()
        
        # 5. 일기 저장 - title 파라미터 제거
        diary_id = save_diary_to_db(connection, content, is_public, user_id)
        
        # 6. SNS 이벤트 발행
        sns_success = publish_diary_created_event(diary_id, user_id)
        
        # 7. 저장된 일기 정보 조회
        saved_diary = get_saved_diary(connection, diary_id)
        
        # 8. 성공 응답
        response_body = {
            'success': True,
            'message': '일기가 성공적으로 저장되었습니다',
            'data': {
                'diary': saved_diary,
                'sns_published': sns_success
            }
        }
        
        logger.info(f"일기 POST 완료: diary_id={diary_id}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': json.dumps(response_body, ensure_ascii=False)
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {str(e)}")
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'success': False,
                'message': 'JSON 형식이 올바르지 않습니다'
            }, ensure_ascii=False)
        }
        
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'success': False,
                'message': f'서버 오류가 발생했습니다: {str(e)}'
            }, ensure_ascii=False)
        }
        
    finally:
        if connection:
            connection.close()
            logger.info("DB 연결 닫음")