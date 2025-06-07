import json
import pymysql
import math
import os

# DB 연결 설정 (환경변수에서만 가져옴)
DB_CONFIG = {
    "host": os.environ['DB_HOST'],
    "user": os.environ['DB_USER'],
    "password": os.environ['DB_PASSWORD'],
    "database": os.environ['DB_NAME'],
    "charset": 'utf8mb4'
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)

def calculate_cosine_similarity(vec1, vec2):
    """numpy 없이 코사인 유사도 계산"""
    try:
        # 벡터 길이 확인
        if len(vec1) != len(vec2):
            return 0.0
        
        # 내적 계산
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # 벡터 크기 계산
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        # 0으로 나누기 방지
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    except Exception as e:
        print(f"코사인 유사도 계산 오류: {str(e)}")
        return 0.0

def get_user_recent_emotion_vector(user_id):
    """사용자의 가장 최근 감정벡터 1개 조회"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
            SELECT emotion_vector 
            FROM diaries 
            WHERE user_id = %s 
              AND emotion_vector IS NOT NULL 
              AND analysis_status = 'completed'
            ORDER BY created_at DESC 
            LIMIT 1
            """, (user_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            # JSON에서 벡터 추출
            try:
                vector = json.loads(result['emotion_vector'])
                if isinstance(vector, list) and len(vector) == 10:
                    return vector
            except (json.JSONDecodeError, TypeError) as e:
                print(f"벡터 파싱 오류: {str(e)}")
                return None
            
            return None
            
    except Exception as e:
        print(f"사용자 감정벡터 조회 오류: {str(e)}")
        return None
    finally:
        conn.close()

def find_similar_users(user_id, user_vector, top_k=3):
    """유사한 감정의 사용자들 찾기"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 다른 사용자들의 최근 감정벡터 조회
            cursor.execute("""
            SELECT DISTINCT d.user_id, d.emotion_vector, d.id as diary_id, d.created_at, d.summary
            FROM diaries d
            WHERE d.user_id != %s 
              AND d.emotion_vector IS NOT NULL 
              AND d.analysis_status = 'completed'
              AND d.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            ORDER BY d.created_at DESC
            LIMIT 50
            """, (user_id,))
            
            results = cursor.fetchall()
            similarities = []
            
            for row in results:
                try:
                    other_vector = json.loads(row['emotion_vector'])
                    if isinstance(other_vector, list) and len(other_vector) == 10:
                        similarity = calculate_cosine_similarity(user_vector, other_vector)
                        
                        similarities.append({
                            'user_id': row['user_id'],
                            'diary_id': row['diary_id'],
                            'summary': row['summary'] or '',
                            'created_at': str(row['created_at']),
                            'similarity': float(similarity)  # 정렬용
                        })
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"벡터 파싱 오류: {str(e)}")
                    continue
            
            # 유사도 높은 순으로 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # similarity 키 제거하고 리턴
            result = []
            for item in similarities[:top_k]:
                result.append({
                    'user_id': item['user_id'],
                    'diary_id': item['diary_id'],
                    'summary': item['summary'],
                    'created_at': item['created_at']
                })
            
            return result
            
    except Exception as e:
        print(f"유사한 사용자 찾기 오류: {str(e)}")
        return []
    finally:
        conn.close()

def find_opposite_users(user_id, user_vector, top_k=3):
    """반대 감정의 사용자들 찾기"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
            SELECT DISTINCT d.user_id, d.emotion_vector, d.id as diary_id, d.created_at, d.summary
            FROM diaries d
            WHERE d.user_id != %s 
              AND d.emotion_vector IS NOT NULL 
              AND d.analysis_status = 'completed'
              AND d.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            ORDER BY d.created_at DESC
            LIMIT 50
            """, (user_id,))
            
            results = cursor.fetchall()
            similarities = []
            
            for row in results:
                try:
                    other_vector = json.loads(row['emotion_vector'])
                    if isinstance(other_vector, list) and len(other_vector) == 10:
                        similarity = calculate_cosine_similarity(user_vector, other_vector)
                        
                        similarities.append({
                            'user_id': row['user_id'],
                            'diary_id': row['diary_id'],
                            'summary': row['summary'] or '',
                            'created_at': str(row['created_at']),
                            'similarity': float(similarity)  # 정렬용
                        })
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"벡터 파싱 오류: {str(e)}")
                    continue
            
            # 유사도 낮은 순으로 정렬 (반대 감정)
            similarities.sort(key=lambda x: x['similarity'])
            
            # similarity 키 제거하고 리턴
            result = []
            for item in similarities[:top_k]:
                result.append({
                    'user_id': item['user_id'],
                    'diary_id': item['diary_id'],
                    'summary': item['summary'],
                    'created_at': item['created_at']
                })
            
            return result
            
    except Exception as e:
        print(f"반대 감정 사용자 찾기 오류: {str(e)}")
        return []
    finally:
        conn.close()

def lambda_handler(event, context):
    """추천 Lambda 핸들러"""
    
    # CORS 헤더
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    }
    
    try:
        print(f"이벤트 수신: {json.dumps(event, ensure_ascii=False)}")
        
        # OPTIONS 요청 처리
        if event.get('httpMethod') == 'OPTIONS':
            return {'statusCode': 200, 'headers': headers, 'body': ''}
        
        # 경로에서 userId 추출: /api/recommend/{userId}
        path = event.get('path', '')
        path_parameters = event.get('pathParameters') or {}
        
        # Path parameter에서 userId 추출
        user_id = None
        if 'userId' in path_parameters:
            try:
                user_id = int(path_parameters['userId'])
            except (ValueError, TypeError):
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'error': 'userId는 숫자여야 합니다'}, ensure_ascii=False)
                }
        
        if not user_id:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'URL에서 userId를 찾을 수 없습니다. /api/recommend/{userId} 형식으로 요청하세요'}, ensure_ascii=False)
            }
        
        print(f"사용자 추천 요청: user_id={user_id}")
        
        # 1. 사용자의 가장 최근 감정벡터 조회 (1개)
        user_vector = get_user_recent_emotion_vector(user_id)
        if not user_vector:
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({'error': '사용자의 감정 데이터를 찾을 수 없습니다'}, ensure_ascii=False)
            }
        
        print(f"사용자 최신 감정벡터: {user_vector}")
        
        # 2. 유사한 감정과 반대 감정 사용자 찾기
        result = {}
        
        # 유사한 감정 사용자 TOP 3
        similar_users = find_similar_users(user_id, user_vector, top_k=3)
        result['similar_users'] = similar_users
        print(f"유사한 사용자 {len(similar_users)}명 찾음")
        
        # 반대 감정 사용자 TOP 3
        opposite_users = find_opposite_users(user_id, user_vector, top_k=3)
        result['opposite_users'] = opposite_users
        print(f"반대 감정 사용자 {len(opposite_users)}명 찾음")
        
        print(f"추천 결과 생성 완료")
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(result, ensure_ascii=False)
        }
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': f'서버 오류: {str(e)}'}, ensure_ascii=False)
        }