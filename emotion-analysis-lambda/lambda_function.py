import json
import boto3
import pymysql
import os
from datetime import datetime, timedelta
import base64
from typing import Dict, List, Optional, Tuple
import re

# AWS 클라이언트 초기화
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
comprehend_client = boto3.client('comprehend', region_name='us-east-1')
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')
lambda_client = boto3.client('lambda', region_name='us-east-1')

# 환경 변수
DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']
S3_BUCKET = os.environ['S3_BUCKET']
EMOTION_RECALL_LAMBDA_NAME = os.environ.get('EMOTION_RECALL_LAMBDA_NAME', 'emotion_recall_lambda')

# 감정 카테고리 정의 (10개)
EMOTION_LABELS = [
    "기쁨",
    "슬픔",
    "분노",
    "불안",
    "설렘",
    "지루함",
    "외로움",
    "만족",
    "실망",
    "무기력"
]

# Comprehend 감정과 우리 감정 라벨 매핑
COMPREHEND_EMOTION_MAPPING = {
    'POSITIVE': ['기쁨', '설렘', '만족'],
    'NEGATIVE': ['슬픔', '분노', '실망', '무기력'],
    'NEUTRAL': ['지루함'],
    'MIXED': ['불안', '외로움']
}

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

def get_user_history(connection, user_id: int, current_date: str) -> List[Dict]:
    """사용자의 과거 일기 히스토리 조회 (최근 30일)"""
    with connection.cursor() as cursor:
        query = """
        SELECT 
            id as diary_id,
            created_at AS date,
            content,
            summary,
            tags,
            emotion_vector
        FROM diaries
        WHERE user_id = %s 
            AND created_at < %s
            AND created_at >= DATE_SUB(%s, INTERVAL 30 DAY)
            AND analysis_status = 'completed'
        ORDER BY created_at DESC
        LIMIT 10
        """
        cursor.execute(query, (user_id, current_date, current_date))
        return cursor.fetchall()

def analyze_image_with_bedrock(image_url: str) -> str:
    """S3 이미지를 Bedrock으로 분석"""
    try:
        # S3에서 이미지 다운로드
        bucket_name = S3_BUCKET
        key = image_url.replace(f"https://{bucket_name}.s3.amazonaws.com/", "")
        
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        
        # Base64 인코딩
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Bedrock 프롬프트
        prompt = """이미지를 분석하고 다음 정보를 추출해주세요:
1. 이미지의 전반적인 분위기
2. 표현된 감정
3. 주요 요소들
4. 일기와의 연관성

간단하고 명확하게 2-3문장으로 요약해주세요."""

        # Bedrock 호출
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.7
        }
        
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType='application/json',
            body=json.dumps(request_body)
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
        
    except Exception as e:
        print(f"이미지 분석 오류: {str(e)}")
        return "이미지 분석 실패"

def analyze_with_comprehend(text: str) -> Dict:
    """AWS Comprehend를 사용한 감정 및 키워드 분석"""
    try:
        # 감정 분석
        sentiment_response = comprehend_client.detect_sentiment(
            Text=text,
            LanguageCode='ko'
        )
        
        # 키 구문 추출
        key_phrases_response = comprehend_client.detect_key_phrases(
            Text=text,
            LanguageCode='ko'
        )
        
        # 결과 정리
        sentiment = sentiment_response['Sentiment']
        sentiment_scores = sentiment_response['SentimentScore']
        
        key_phrases = [
            phrase['Text'] for phrase in key_phrases_response['KeyPhrases']
            if phrase['Score'] > 0.8  # 높은 신뢰도만 선택
        ]
        
        return {
            'sentiment': sentiment,
            'sentiment_scores': sentiment_scores,
            'key_phrases': key_phrases[:5]  # 상위 5개만
        }
        
    except Exception as e:
        print(f"Comprehend 분석 오류: {str(e)}")
        return {
            'sentiment': 'NEUTRAL',
            'sentiment_scores': {'Positive': 0.25, 'Negative': 0.25, 'Neutral': 0.5, 'Mixed': 0.0},
            'key_phrases': []
        }

def create_emotion_vector_with_comprehend(bedrock_vector: List[float], comprehend_result: Dict) -> List[float]:
    """Bedrock 결과와 Comprehend 결과를 결합하여 감정 벡터 생성"""
    try:
        # 기본적으로 Bedrock 벡터 사용
        emotion_vector = bedrock_vector.copy()
        
        # Comprehend 감정을 보완적으로 활용
        sentiment = comprehend_result['sentiment']
        sentiment_scores = comprehend_result['sentiment_scores']
        
        # Comprehend 감정에 따른 벡터 조정
        if sentiment in COMPREHEND_EMOTION_MAPPING:
            related_emotions = COMPREHEND_EMOTION_MAPPING[sentiment]
            
            for emotion in related_emotions:
                if emotion in EMOTION_LABELS:
                    idx = EMOTION_LABELS.index(emotion)
                    # Comprehend 점수를 반영하여 기존 값에 가중치 적용
                    comprehend_weight = sentiment_scores.get(sentiment.title(), 0) * 0.3  # 30% 가중치
                    emotion_vector[idx] = min(1.0, emotion_vector[idx] + comprehend_weight)
        
        # 벡터 정규화 (합이 1이 되도록)
        vector_sum = sum(emotion_vector)
        if vector_sum > 0:
            emotion_vector = [v / vector_sum for v in emotion_vector]
        
        return emotion_vector
        
    except Exception as e:
        print(f"감정 벡터 결합 오류: {str(e)}")
        return bedrock_vector

def create_analysis_prompt(content: str, past_diaries: List[Dict], image_analysis: str = "") -> str:
    """Bedrock 분석을 위한 개선된 프롬프트 생성 (기존 방식 유지)"""
    
    # 과거 일기 정보 간단히 포맷팅
    history_text = ""
    if past_diaries:
        history_text = "\n\n### 과거 일기 히스토리:\n"
        for diary in past_diaries[:5]:  # 최근 5개만
            summary = diary.get('summary', '')[:50] if diary.get('summary') else '요약 없음'
            history_text += f"- {diary['date']}: {summary}...\n"
    
    # 이미지 분석 결과
    image_text = ""
    if image_analysis:
        image_text = f"\n\n### 이미지 분석 결과:\n{image_analysis}\n"
    
    prompt = f"""당신은 전문 심리 상담사이자 데이터 분석가입니다. 
사용자가 입력한 오늘의 일기와 과거 일기, 이미지 분석 결과를 바탕으로 반드시 아래 '출력 예시(JSON)'와 똑같은 형태의 JSON 객체만 반환해주세요. 
다른 설명이나 추가 텍스트는 일절 출력하지 마세요.

### 오늘의 일기:
{content}
{history_text}
{image_text}

### 분석 요청
1. 감정 카테고리 10개에 대해 0~1 사이 값 할당 → emotion_vector
   (순서: 기쁨, 슬픔, 분노, 불안, 설렘, 지루함, 외로움, 만족, 실망, 무기력)
2. 주요 감정에 대한 간단 설명 → emotion_analysis
3. 과거 일기와 비교한 피드백 → past_feedback
4. 2~3문장 요약 → summary
5. 주요 키워드 태그 5개 이하 → tags
6. 가장 강한 감정 하나 → primary_emotion (EMOTION_LABELS 중 하나)

### 출력 예시 (정확히 이 형식을 따라야 함)
```json
{{
  "emotion_vector": [0.1, 0.7, 0.0, 0.3, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0],
  "emotion_analysis": "슬픔과 외로움이 주된 감정으로, 친구와의 갈등 후 아픔이 느껴집니다.",
  "past_feedback": "지난주에는 비슷한 슬픔을 느꼈으나 당시에 산책으로 기분을 전환했었습니다.",
  "summary": "친구와 다툰 후 슬픔과 외로움을 느꼈습니다.",
  "tags": ["슬픔", "외로움", "친구갈등"],
  "primary_emotion": "슬픔"
}}
```

"```json ... ```" 블록은 예시이므로 출력에 포함하지 말고, 오직 중괄호로 시작해 중괄호로 끝나는 JSON만 응답해주세요."""
    
    return prompt

def analyze_with_bedrock(prompt: str) -> Dict:
    """Bedrock을 사용한 일기 분석 (기존 방식 유지)"""
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7
        }
        
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType='application/json',
            body=json.dumps(request_body)
        )
        
        result = json.loads(response['body'].read())
        analysis_text = result['content'][0]['text']
        
        # JSON 추출
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            analysis_result = json.loads(json_match.group())
            
            # emotion_vector 검증
            if 'emotion_vector' in analysis_result:
                emotion_vector = analysis_result['emotion_vector']
                if len(emotion_vector) != 10:
                    print(f"경고: emotion_vector 길이가 {len(emotion_vector)}입니다. 10개로 조정합니다.")
                    # 부족하면 0으로 채우고, 많으면 잘라냄
                    if len(emotion_vector) < 10:
                        emotion_vector.extend([0.0] * (10 - len(emotion_vector)))
                    else:
                        emotion_vector = emotion_vector[:10]
                    analysis_result['emotion_vector'] = emotion_vector
                
                # 값이 0-1 범위인지 확인
                analysis_result['emotion_vector'] = [
                    max(0.0, min(1.0, float(v))) for v in emotion_vector
                ]
            
            # primary_emotion 검증
            if 'primary_emotion' not in analysis_result or analysis_result['primary_emotion'] not in EMOTION_LABELS:
                # emotion_vector에서 가장 높은 값의 인덱스 찾기
                emotion_vector = analysis_result.get('emotion_vector', [0.0] * 10)
                max_index = emotion_vector.index(max(emotion_vector))
                analysis_result['primary_emotion'] = EMOTION_LABELS[max_index]
            
            return analysis_result
        else:
            raise ValueError("JSON 형식을 찾을 수 없습니다")
            
    except Exception as e:
        print(f"Bedrock 분석 오류: {str(e)}")
        # 오류 시 기본값 반환
        return {
            "emotion_vector": [0.0] * 10,
            "emotion_analysis": "분석 중 오류가 발생했습니다.",
            "past_feedback": "",
            "summary": "일기 분석 실패",
            "tags": [],
            "primary_emotion": "무기력"
        }

def generate_emotion_vector_for_recall(text: str) -> List[float]:
    """회고 분석을 위한 감정 벡터 생성 (Comprehend + Bedrock)"""
    try:
        # Comprehend 분석
        comprehend_result = analyze_with_comprehend(text)
        
        # 간단한 Bedrock 프롬프트로 감정 벡터 추출
        simple_prompt = f"""다음 일기 내용을 분석하여 감정 벡터만 생성해주세요.

일기 내용: {text}

감정 카테고리 (순서대로): 기쁨, 슬픔, 분노, 불안, 설렘, 지루함, 외로움, 만족, 실망, 무기력

각 감정에 대해 0~1 사이의 값을 할당하여 10개 숫자의 배열만 반환해주세요.
형식: [0.1, 0.2, 0.0, 0.3, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0]

오직 숫자 배열만 응답하세요."""

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": simple_prompt
                }
            ],
            "temperature": 0.5
        }
        
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType='application/json',
            body=json.dumps(request_body)
        )
        
        result = json.loads(response['body'].read())
        analysis_text = result['content'][0]['text']
        
        # 배열 추출
        import ast
        vector_match = re.search(r'\[[\d\.,\s]+\]', analysis_text)
        if vector_match:
            bedrock_vector = ast.literal_eval(vector_match.group())
            if len(bedrock_vector) == 10:
                # Comprehend 결과와 결합
                final_vector = create_emotion_vector_with_comprehend(bedrock_vector, comprehend_result)
                return final_vector
        
        # 실패 시 Comprehend 기반 기본 벡터 생성
        return create_default_emotion_vector(comprehend_result)
        
    except Exception as e:
        print(f"회고용 감정 벡터 생성 오류: {str(e)}")
        return [0.1] * 10  # 기본 균등 분포

def create_default_emotion_vector(comprehend_result: Dict) -> List[float]:
    """Comprehend 결과를 기반으로 기본 감정 벡터 생성"""
    vector = [0.0] * 10
    
    sentiment = comprehend_result.get('sentiment', 'NEUTRAL')
    sentiment_scores = comprehend_result.get('sentiment_scores', {})
    
    if sentiment in COMPREHEND_EMOTION_MAPPING:
        emotions = COMPREHEND_EMOTION_MAPPING[sentiment]
        base_score = sentiment_scores.get(sentiment.title(), 0.5)
        
        for emotion in emotions:
            if emotion in EMOTION_LABELS:
                idx = EMOTION_LABELS.index(emotion)
                vector[idx] = base_score / len(emotions)
    
    # 정규화
    vector_sum = sum(vector)
    if vector_sum > 0:
        vector = [v / vector_sum for v in vector]
    else:
        vector = [0.1] * 10  # 균등 분포
    
    return vector

def call_emotion_recall_lambda(user_id: int, text: str, emotion_vector: List[float], is_emotional: bool) -> Dict:
    """감정 회고 Lambda 호출"""
    try:
        payload = {
            'userId': user_id,
            'text': text,
            'emotionVector': emotion_vector,
            'isEmotional': is_emotional
        }
        
        print(f"감정 회고 Lambda 호출: {EMOTION_RECALL_LAMBDA_NAME}")
        print(f"Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        response = lambda_client.invoke(
            FunctionName=EMOTION_RECALL_LAMBDA_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        result = json.loads(response['Payload'].read())
        print(f"회고 Lambda 응답: {json.dumps(result, ensure_ascii=False)}")
        
        if result.get('statusCode') == 200:
            body = json.loads(result['body'])
            return body.get('recallSummaries', {})
        else:
            print(f"회고 Lambda 오류: {result}")
            return {
                "20240101": "회고 생성 중 오류가 발생했습니다.",
                "20240102": "다시 시도해주세요."
            }
            
    except Exception as e:
        print(f"감정 회고 Lambda 호출 오류: {str(e)}")
        return {
            "20240101": "회고 생성 중 오류가 발생했습니다.",
            "20240102": "다시 시도해주세요."
        }

def update_diary_analysis(connection, diary_id: int, analysis: Dict):
    """분석 결과를 데이터베이스에 저장 (기존 방식 유지)"""
    with connection.cursor() as cursor:
        # emotion_vector와 tags를 JSON 문자열로 변환
        emotion_vector_json = json.dumps(analysis['emotion_vector'])
        tags_json = json.dumps(analysis['tags'], ensure_ascii=False)
        
        query = """
        UPDATE diaries
        SET 
            summary = %s,
            feedback = %s,
            tags = %s,
            emotion_vector = %s,
            primary_emotion = %s,
            analysis_status = 'completed',
            analyzed_at = NOW()
        WHERE id = %s
        """
        
        cursor.execute(query, (
            analysis['summary'],
            analysis.get('past_feedback', ''),
            tags_json,
            emotion_vector_json,
            analysis['primary_emotion'],
            diary_id
        ))
        connection.commit()

def publish_to_sns(diary_id: int, user_id: int, emotion_vector: List[float]):
    """분석 완료 메시지를 SNS에 발행 (기존 방식 유지)"""
    message = {
        "diaryId": diary_id,
        "userId": user_id,
        "emotionVector": emotion_vector,
        "event": "diary_analyzed"
    }
    
    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=json.dumps(message),
        Subject='diary_analyzed'
    )

def handle_emotion_recall_request(event, context):
    """감정 회고 요청 처리 (새로운 기능)"""
    try:
        # 요청 본문 파싱
        body = json.loads(event.get('body', '{}'))
        user_id = body.get('userId')
        text = body.get('text')
        is_emotional = body.get('isEmotional', True)
        
        if not user_id or not text:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'userId와 text는 필수입니다'}, ensure_ascii=False)
            }
        
        print(f"감정 회고 요청: user_id={user_id}, isEmotional={is_emotional}")
        
        # 1. 감정 벡터 생성 (Comprehend + Bedrock)
        emotion_vector = generate_emotion_vector_for_recall(text)
        
        # 2. 감정 회고 Lambda 호출
        recall_summaries = call_emotion_recall_lambda(user_id, text, emotion_vector, is_emotional)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'recallSummaries': recall_summaries
            }, ensure_ascii=False)
        }
                
    except Exception as e:
        print(f"감정 회고 요청 처리 오류: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)}, ensure_ascii=False)
        }

def handle_sqs_request(event, context):
    """SQS 요청 처리 (기존 로직 유지)"""
    connection = None
    
    try:
        # SQS 메시지 파싱
        for record in event['Records']:
            try:
                # SQS 메시지 본문 파싱
                message_body = json.loads(record['body'])
                print(f"SQS 메시지 본문: {json.dumps(message_body, ensure_ascii=False)}")
                
                # SNS 메시지인 경우 Message 필드 파싱
                if 'Message' in message_body:
                    try:
                        message_data = json.loads(message_body['Message'])
                    except json.JSONDecodeError as e:
                        print(f"SNS 메시지 파싱 오류: {str(e)}")
                        print(f"원본 메시지: {message_body['Message']}")
                        continue
                else:
                    message_data = message_body
                
                print(f"파싱된 메시지 데이터: {json.dumps(message_data, ensure_ascii=False)}")
                
                # 메시지에서 필요한 정보 추출
                diary_id = message_data.get('diaryId')
                user_id = message_data.get('userId')
                event_type = message_data.get('event')
                
                # 유효성 검사
                if event_type != 'diary_created':
                    print(f"잘못된 이벤트 타입: {event_type}")
                    continue
                    
                if not diary_id or not user_id:
                    print(f"필수 정보 누락: diary_id={diary_id}, user_id={user_id}")
                    continue
                
                print(f"일기 분석 시작: diary_id={diary_id}, user_id={user_id}")
                
                # DB 연결
                connection = get_db_connection()
                
                # 현재 일기 정보 조회
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT content, image_url, created_at AS date FROM diaries WHERE id = %s AND user_id = %s",
                        (diary_id, user_id)
                    )
                    diary = cursor.fetchone()
                
                if not diary:
                    raise ValueError(f"일기를 찾을 수 없습니다: diary_id={diary_id}, user_id={user_id}")
                
                # 과거 일기 히스토리 조회
                past_diaries = get_user_history(connection, user_id, str(diary['date']))
                
                # 이미지 분석 (image_url이 있는 경우)
                image_analysis = ""
                if diary.get('image_url'):
                    try:
                        image_analysis = analyze_image_with_bedrock(diary['image_url'])
                    except Exception as e:
                        print(f"이미지 분석 실패: {diary['image_url']}, 오류: {str(e)}")
                        image_analysis = "이미지 분석 실패"
                
                # Bedrock 프롬프트 생성 (기존 방식)
                prompt = create_analysis_prompt(
                    diary['content'], 
                    past_diaries, 
                    image_analysis
                )
                
                # Bedrock으로 분석 (기존 방식)
                analysis_result = analyze_with_bedrock(prompt)
                
                # 분석 결과 저장 (기존 방식)
                update_diary_analysis(connection, diary_id, analysis_result)
                
                # SNS에 분석 완료 메시지 발행 (기존 방식)
                publish_to_sns(diary_id, user_id, analysis_result['emotion_vector'])
                
                print(f"일기 분석 완료: diary_id={diary_id}")
                
            except Exception as record_error:
                print(f"레코드 처리 오류: {str(record_error)}")
                import traceback
                print(traceback.format_exc())
                continue
                
        return {
            'statusCode': 200,
            'body': json.dumps({'message': '분석 완료', 'processed': len(event['Records'])})
        }
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # 분석 실패 상태 업데이트
        if connection and 'diary_id' in locals():
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE diaries SET analysis_status = 'failed' WHERE id = %s",
                        (diary_id,)
                    )
                    connection.commit()
            except Exception as update_error:
                print(f"상태 업데이트 실패: {str(update_error)}")
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
        
    finally:
        if connection:
            connection.close()

def lambda_handler(event, context):
    """Lambda 메인 핸들러"""
    print(f"이벤트 수신: {json.dumps(event, ensure_ascii=False)}")
    
    # 요청 타입 판단
    if 'Records' in event:
        # SQS 요청 - 기존 전체 분석 로직
        print("SQS 요청: 전체 일기 분석을 수행합니다")
        return handle_sqs_request(event, context)
    else:
        # API Gateway 요청 - 감정 회고 기능
        print("API Gateway 요청: 감정 회고 분석을 수행합니다")
        return handle_emotion_recall_request(event, context)