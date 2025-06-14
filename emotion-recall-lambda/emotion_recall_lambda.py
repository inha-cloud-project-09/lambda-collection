import json
import boto3
import pymysql
import os
from datetime import datetime
import math
import re
import urllib3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def cosine_similarity(vector1, vector2):
    try:
        if len(vector1) != len(vector2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(a * a for a in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    except Exception as e:
        logger.error(f"코사인 유사도 계산 오류: {str(e)}")
        return 0.0

def get_past_diaries(connection, user_id):
    with connection.cursor() as cursor:
        query = """
        SELECT id, content, created_at, emotion_vector 
        FROM diaries 
        WHERE user_id = %s 
            AND analysis_status = 'completed' 
            AND emotion_vector IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 100
        """
        cursor.execute(query, (user_id,))
        return cursor.fetchall()

def format_date_for_bedrock(date_obj):
    try:
        if isinstance(date_obj, str):
            try:
                date_obj = datetime.strptime(date_obj, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
                except:
                    return "알 수 없는 날짜"
        
        return date_obj.strftime('%Y년 %m월 %d일')
    
    except Exception as e:
        logger.error(f"날짜 형식 변환 오류: {str(e)}")
        return "알 수 없는 날짜"

def find_most_similar_diary(current_emotion_vector, past_diaries):
    best_similarity = -1
    best_diary = None
    
    for diary in past_diaries:
        try:
            if isinstance(diary['emotion_vector'], str):
                emotion_vector = json.loads(diary['emotion_vector'])
            else:
                emotion_vector = diary['emotion_vector']
            
            similarity = cosine_similarity(current_emotion_vector, emotion_vector)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_diary = {
                    'diary': diary,
                    'similarity': similarity
                }
        
        except Exception as e:
            logger.error(f"일기 처리 오류: {str(e)}")
            continue
    
    return best_diary

def create_bedrock_prompt(current_text, similar_diary_data):
    if not similar_diary_data:
        return None
    
    diary = similar_diary_data['diary']
    past_content = diary['content']
    past_date = diary['created_at']
    
    past_date_formatted = format_date_for_bedrock(past_date)
    current_date = datetime.now().strftime('%Y년 %m월 %d일')
    
    prompt = f"""너는 지금 친한 친구나 가족이 되어서, 상대방이 털어놓는 이야기를 듣고 과거 비슷한 경험을 회상하며 자연스럽게 대화하는 역할이야.

오늘 날짜: {current_date}

상대방이 지금 말한 내용:
\"\"\"{current_text}\"\"\"

과거에 비슷한 일이 있었는데, 그때는 {past_date_formatted}이었고 내용은:
\"\"\"{past_content}\"\"\"

이제 너는 이 두 상황을 비교하면서 친한 사람처럼 자연스럽게 회고해줘야 해. 

조건:
1. 무조건 반말로 말해 (반드시!)
2. "와..", "진짜", "그땐 그랬네", "아 맞다" 같은 자연스러운 감탄사 사용
3. 과거 날짜({past_date_formatted})와 오늘 날짜({current_date})를 비교해서 자연스럽게 "며칠 전에", "몇 주 전에", "몇 달 전에", "몇 년 전에" 등으로 표현
4. 단순 회고가 아니라 추론이나 조언도 포함해 ("이건 악의적인 거 같아", "경찰에 신고하는 게 어때?" 등)
5. 친한 사람이 옆에서 진심으로 걱정하고 조언하는 말투
6. 구체적인 인물이나 상황이 있으면 그대로 언급하기
7. 응답은 1-2문장으로 간결하게
8. 날짜 차이를 직접 계산해서 자연스럽게 말해 (예: "2개월 전에도", "일주일 전에도", "어제도" 등)

예시:
- "와.. 2개월 전에도 철수가 너를 때렸는데 이건 진짜 악의적인 거 같아. 경찰에 신고하는 게 어때?"
- "아 맞다, 한 달 전에도 비슷하게 우울했었네. 그땐 산책하고 나서 좀 나아졌잖아?"
- "진짜? 몇 주 전에도 그 사람 때문에 스트레스 받았는데 아직도 그러는구나. 좀 거리 두는 게 나을 거 같아."

이제 위 조건에 맞게 자연스럽게 응답해줘:"""
    
    return prompt

def call_bedrock_claude(prompt):
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.8
        }
        
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType='application/json',
            body=json.dumps(request_body)
        )
        
        result = json.loads(response['body'].read())
        response_text = result['content'][0]['text'].strip()
        
        return response_text
            
    except Exception as e:
        logger.error(f"Bedrock 호출 오류: {str(e)}")
        return "예전에 비슷한 일이 있었는데 지금 생각해보니까 그때랑 비슷한 패턴인 것 같아."

def send_slack_notification(user_id, input_text, recall_summary):
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL 환경변수가 설정되지 않았습니다.")
        return
    
    try:
        input_preview = input_text[:100] + "..." if len(input_text) > 100 else input_text
        recall_preview = recall_summary[:200] + "..." if len(recall_summary) > 200 else recall_summary
        
        slack_message = {
            "channel": "#2025-인하대-클컴-9팀-logging",
            "username": "Bedrock",
            "text": "감정 회고 분석이 완료되었습니다!",
            "attachments": [{
                "color": "#8e44ad",
                "fields": [
                    {
                        "title": "사용자 ID",
                        "value": f"#{user_id}",
                        "short": True
                    },
                    {
                        "title": "처리 시간",
                        "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "short": True
                    },
                    {
                        "title": "입력 텍스트",
                        "value": f"```{input_preview}```",
                        "short": False
                    },
                    {
                        "title": "생성된 회고",
                        "value": f"```{recall_preview}```",
                        "short": False
                    }
                ],
                "footer": "감정 일기 시스템 - 회고 분석"
            }]
        }
        
        http = urllib3.PoolManager()
        response = http.request(
            'POST',
            webhook_url,
            body=json.dumps(slack_message, ensure_ascii=False),
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status == 200:
            logger.info(f"회고 완료 슬랙 알림 전송 성공: user_id={user_id}")
        else:
            logger.error(f"슬랙 전송 실패: {response.status}")
            
    except Exception as e:
        logger.error(f"슬랙 알림 전송 오류: {str(e)}")

def lambda_handler(event, context):
    connection = None
    
    try:
        logger.info(f"수신된 이벤트: {json.dumps(event, ensure_ascii=False)}")
        
        if 'body' in event and event['body']:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
            
            user_id = body.get('userId')
            text = body.get('text')
            emotion_vector = body.get('emotionVector')
        else:
            user_id = event.get('userId')
            text = event.get('text')
            emotion_vector = event.get('emotionVector')
        
        if not user_id or not text or not emotion_vector:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': '필수 입력값이 누락되었습니다',
                    'required': ['userId', 'text', 'emotionVector']
                }, ensure_ascii=False)
            }
        
        if len(emotion_vector) != 10:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'emotion_vector는 10개 요소를 가져야 합니다'
                }, ensure_ascii=False)
            }
        
        logger.info(f"회고 분석 시작: user_id={user_id}")
        
        connection = get_db_connection()
        past_diaries = get_past_diaries(connection, user_id)
        
        if not past_diaries:
            recall_summary = "아직 충분한 일기 데이터가 없어서 회고할 게 없네. 더 많이 써봐!"
            send_slack_notification(user_id, text, recall_summary)
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'recallSummary': recall_summary
                }, ensure_ascii=False)
            }
        
        most_similar = find_most_similar_diary(emotion_vector, past_diaries)
        
        if not most_similar:
            recall_summary = "비슷한 감정의 과거 일기를 찾을 수 없네. 이런 감정은 처음인가봐?"
            send_slack_notification(user_id, text, recall_summary)
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'recallSummary': recall_summary
                }, ensure_ascii=False)
            }
        
        prompt = create_bedrock_prompt(text, most_similar)
        
        if not prompt:
            recall_summary = "예전 기억이 잘 안 나네.. 다음에 다시 생각해볼게."
            send_slack_notification(user_id, text, recall_summary)
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'recallSummary': recall_summary
                }, ensure_ascii=False)
            }
        
        recall_summary = call_bedrock_claude(prompt)
        
        send_slack_notification(user_id, text, recall_summary)
        
        logger.info(f"회고 분석 완료: user_id={user_id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'recallSummary': recall_summary
            }, ensure_ascii=False)
        }
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': '회고 생성 중 오류가 발생했습니다',
                'details': str(e)
            }, ensure_ascii=False)
        }
        
    finally:
        if connection:
            connection.close()