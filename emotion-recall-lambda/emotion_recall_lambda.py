import json
import boto3
import pymysql
import os
from datetime import datetime
import math
import re

# AWS 클라이언트 초기화
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# 환경 변수
DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']

def get_db_connection():
    """MySQL RDS 데이터베이스 연결"""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def cosine_similarity(vector1, vector2):
    """두 벡터 간 코사인 유사도 계산 (numpy 없이)"""
    try:
        # 벡터 크기 확인
        if len(vector1) != len(vector2):
            return 0.0
        
        # 내적 계산
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        # 벡터 크기 계산
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(a * a for a in vector2))
        
        # 0으로 나누기 방지
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    except Exception as e:
        print(f"코사인 유사도 계산 오류: {str(e)}")
        return 0.0

def get_past_diaries(connection, user_id):
    """사용자의 과거 일기 조회 (분석 완료된 것만)"""
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

def format_date_to_yyyymmdd(date_obj):
    """날짜를 YYYYMMDD 형식으로 변환"""
    if isinstance(date_obj, str):
        # 문자열인 경우 파싱 시도
        try:
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d %H:%M:%S')
        except:
            try:
                date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
            except:
                return "20240101"  # 기본값
    
    return date_obj.strftime('%Y%m%d')

def find_similar_diaries(current_emotion_vector, past_diaries):
    """현재 감정 벡터와 유사한 과거 일기 찾기"""
    similarities = []
    
    for diary in past_diaries:
        try:
            # emotion_vector 파싱
            if isinstance(diary['emotion_vector'], str):
                emotion_vector = json.loads(diary['emotion_vector'])
            else:
                emotion_vector = diary['emotion_vector']
            
            # 유사도 계산
            similarity = cosine_similarity(current_emotion_vector, emotion_vector)
            
            similarities.append({
                'diary': diary,
                'similarity': similarity,
                'date': format_date_to_yyyymmdd(diary['created_at'])
            })
        
        except Exception as e:
            print(f"일기 처리 오류: {str(e)}")
            continue
    
    # 유사도 기준으로 정렬하고 상위 5개 선택
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:5]

def create_bedrock_prompt(current_text, similar_diaries, is_emotional):
    """Bedrock Claude 호출을 위한 프롬프트 생성"""
    
    # 과거 유사한 일기들 텍스트 형식으로 구성
    past_diaries_text = ""
    for i, item in enumerate(similar_diaries, 1):
        date_formatted = item['date']
        content = item['diary']['content']  # 전체 내용 사용
        
        # 날짜를 YYYY년 MM월 DD일 형식으로 변환
        year = date_formatted[:4]
        month = date_formatted[4:6]
        day = date_formatted[6:8]
        formatted_date = f"{year}년 {month}월 {day}일"
        
        past_diaries_text += f"{i}. [{formatted_date}]: \"{content}\"\n"
    
    # 감정 스타일별 설명
    emotional_instruction = ""
    if is_emotional:
        emotional_instruction = """✔️ `isEmotional = true`:
   * 따뜻하고 감정적인 문장
   * 사용자의 감정을 공감하고 보듬는 말투
   * 상담자나 조언자처럼 표현"""
    else:
        emotional_instruction = """✔️ `isEmotional = false`:
   * 분석적이고 객관적인 문장
   * 상황의 반복, 패턴, 인물의 행동을 근거로 한 추측이나 경고
   * 감정 대신 **팩트와 통찰에 집중**"""
    
    prompt = f"""[역할]
당신은 사용자의 현재 감정 상황과 과거 유사한 감정 기록을 분석하여, 날짜별로 의미 있는 회고를 제공하는 정밀 분석 기반의 AI입니다.

[목표]
아래에 제공된 사용자의 "현재 상황"과 "과거 유사한 일기들"을 바탕으로, 가장 밀접한 연관이 있는 3개의 과거 일기를 선택하고, 각 일기에 대해 구체적이고 통찰력 있는 문장을 하나씩 생성해주세요.

[조건 및 출력 형식]
- 반드시 3개의 회고 문장을 작성하세요
- 출력은 아래 JSON 형식으로 작성합니다:
```json
{{
  "YYYYMMDD": "이 날짜에 해당하는 회고 문장"
}}
```
* 날짜는 실제 제공된 5개의 일기 날짜 중에서 선택합니다.
* 각 문장은 해당 날짜의 일기 내용을 바탕으로 구성합니다.

[추가 제약 조건]
* 각 문장은 "상황과 인물"을 가능한 한 **구체적으로** 표현해주세요.
   * 예: "철수는 그날도 너를 무시했고, 넌 결국 울음을 참지 못했지."
   * "그 친구", "누군가" 같은 일반화된 표현은 피하고, 일기에 등장하는 **이름(예: 철수, 짱구)** 등을 최대한 그대로 활용하세요.
   * 대화체, 비유 대신 **사실을 정리하는 형식의 문장**을 선호합니다.
* isEmotional 값에 따라 문체와 어조를 조절합니다:
{emotional_instruction}

[입력 데이터]
현재 상황: \"\"\"
{current_text}
\"\"\"

과거 유사한 일기들 (총 {len(similar_diaries)}개): \"\"\"
{past_diaries_text}\"\"\"

isEmotional: {is_emotional}

반드시 JSON 형식으로만 응답해주세요:"""
    
    return prompt

def call_bedrock_claude(prompt):
    """Bedrock Claude 3.5 호출"""
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
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
        response_text = result['content'][0]['text']
        
        # JSON 추출
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # JSON을 찾지 못한 경우 기본 응답
            return {
                "20240101": "과거의 경험을 되돌아보며 현재 상황을 이해해보세요.",
                "20240102": "비슷한 상황에서 어떻게 대처했는지 생각해보세요.",
                "20240103": "감정의 패턴을 파악하여 더 나은 선택을 할 수 있습니다."
            }
            
    except Exception as e:
        print(f"Bedrock 호출 오류: {str(e)}")
        # 오류 시 기본 응답
        return {
            "20240101": "과거의 경험을 통해 현재를 이해하려 노력하고 있습니다.",
            "20240102": "비슷한 감정을 느꼈던 순간들을 떠올려보세요.",
            "20240103": "감정의 흐름을 파악하면 더 나은 판단을 할 수 있습니다."
        }

def lambda_handler(event, context):
    """Lambda 메인 핸들러"""
    connection = None
    
    try:
        # 입력값 검증
        user_id = event.get('userId')
        current_text = event.get('text')
        emotion_vector = event.get('emotionVector')
        is_emotional = event.get('isEmotional', True)
        
        if not user_id or not current_text or not emotion_vector:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': '필수 입력값이 누락되었습니다',
                    'required': ['userId', 'text', 'emotionVector']
                }, ensure_ascii=False)
            }
        
        # emotion_vector 검증
        if len(emotion_vector) != 10:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'emotion_vector는 10개 요소를 가져야 합니다'
                }, ensure_ascii=False)
            }
        
        print(f"회고 분석 시작: user_id={user_id}")
        
        # DB 연결
        connection = get_db_connection()
        
        # 과거 일기 조회
        past_diaries = get_past_diaries(connection, user_id)
        
        if not past_diaries:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'recallSummaries': {
                        "20240101": "아직 충분한 일기 데이터가 없어 회고를 생성할 수 없습니다.",
                        "20240102": "더 많은 일기를 작성하신 후 다시 시도해주세요.",
                        "20240103": "감정 패턴 분석을 위해 지속적인 기록이 필요합니다."
                    }
                }, ensure_ascii=False)
            }
        
        # 유사한 일기 찾기
        similar_diaries = find_similar_diaries(emotion_vector, past_diaries)
        
        if not similar_diaries:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'recallSummaries': {
                        "20240101": "유사한 감정의 과거 일기를 찾을 수 없습니다.",
                        "20240102": "현재 감정 상태는 새로운 경험인 것 같습니다.",
                        "20240103": "이 감정을 기록하여 향후 회고 자료로 활용해보세요."
                    }
                }, ensure_ascii=False)
            }
        
        # Bedrock 프롬프트 생성
        prompt = create_bedrock_prompt(current_text, similar_diaries, is_emotional)
        
        # Claude 호출하여 회고 생성
        recall_summaries = call_bedrock_claude(prompt)
        
        print(f"회고 분석 완료: user_id={user_id}, 생성된 회고 수={len(recall_summaries)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'recallSummaries': recall_summaries
            }, ensure_ascii=False)
        }
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
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