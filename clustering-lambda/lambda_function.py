import json
import boto3
import pymysql
import os
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns_client = boto3.client("sns")

# 환경 변수
DB_HOST = os.environ["DB_HOST"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_NAME = os.environ["DB_NAME"]
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')

def get_db_connection():
    """데이터베이스 연결"""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def publish_clustering_completion(total_diaries, total_users, clusters_created):
    try:
        if not SNS_TOPIC_ARN:
            logger.warning("SNS_TOPIC_ARN 환경변수가 설정되지 않았습니다.")
            return
            
        current_time = datetime.now()
        message = {
            "totalDiaries": total_diaries,
            "totalUsers": total_users,
            "clustersCreated": clusters_created,
            "event": "batch_clustering_completed",
            "completedAt": current_time.isoformat(),
            "completedAtKST": (current_time + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(message),
            Subject='batch_clustering_completed'
        )
        logger.info(f"클러스터링 완료 SNS 메시지 발행: {total_users}명 처리 완료")
        
    except Exception as e:
        logger.error(f"SNS 발행 오류: {str(e)}")

def cosine_similarity(a, b):
    """두 벡터 간 코사인 유사도 계산 (순수 Python)"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    # 제로 벡터 처리
    if norm_a == 0 or norm_b == 0:
        return 0.1  # 최소값 부여하여 추천 가능하게

    return dot_product / (norm_a * norm_b)


def euclidean_distance(a, b):
    """유클리드 거리 계산"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def simple_kmeans_python(vectors, n_clusters, max_iters=100, random_seed=42):
    """순수 Python으로 구현된 K-means"""
    n_samples = len(vectors)
    n_features = len(vectors[0]) if vectors else 0

    if n_samples < n_clusters:
        # 샘플 수가 클러스터 수보다 적으면 각 샘플을 개별 클러스터로
        labels = list(range(n_samples))
        centroids = [list(vector) for vector in vectors]
        return labels, centroids

    # 랜덤 시드 설정
    random.seed(random_seed)

    # K-means++ 초기화
    centroids = []
    centroids.append(list(vectors[random.randint(0, n_samples - 1)]))

    for _ in range(1, n_clusters):
        distances = []
        for vector in vectors:
            min_dist = min(
                euclidean_distance(vector, centroid) ** 2 for centroid in centroids
            )
            distances.append(min_dist)

        total_dist = sum(distances)
        if total_dist == 0:
            centroids.append(list(vectors[random.randint(0, n_samples - 1)]))
            continue

        probabilities = [d / total_dist for d in distances]
        cumulative_probs = []
        cumsum = 0
        for p in probabilities:
            cumsum += p
            cumulative_probs.append(cumsum)

        r = random.random()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(list(vectors[j]))
                break

    # K-means 반복
    for iteration in range(max_iters):
        # 각 점을 가장 가까운 중심점에 할당
        labels = []
        for vector in vectors:
            distances = [euclidean_distance(vector, centroid) for centroid in centroids]
            closest_cluster = distances.index(min(distances))
            labels.append(closest_cluster)

        # 새로운 중심점 계산
        new_centroids = []
        for k in range(n_clusters):
            cluster_points = [vectors[i] for i in range(n_samples) if labels[i] == k]

            if cluster_points:
                # 평균 계산
                new_centroid = []
                for feature_idx in range(n_features):
                    feature_sum = sum(point[feature_idx] for point in cluster_points)
                    new_centroid.append(feature_sum / len(cluster_points))
                new_centroids.append(new_centroid)
            else:
                # 빈 클러스터면 기존 중심점 유지
                new_centroids.append(centroids[k][:])

        # 수렴 체크
        converged = True
        for old, new in zip(centroids, new_centroids):
            for old_val, new_val in zip(old, new):
                if abs(old_val - new_val) > 1e-4:
                    converged = False
                    break
            if not converged:
                break

        if converged:
            logger.info(f"K-means 수렴: {iteration + 1}번째 반복")
            break

        centroids = new_centroids

    return labels, centroids


def get_completed_diaries(connection) -> List[Dict]:
    """분석 완료된 모든 일기 조회"""
    with connection.cursor() as cursor:
        query = """
        SELECT 
            id,
            user_id,
            emotion_vector,
            primary_emotion,
            created_at
        FROM diaries 
        WHERE analysis_status = 'completed' 
            AND emotion_vector IS NOT NULL
            AND JSON_LENGTH(emotion_vector) = 10
        ORDER BY created_at DESC
        """
        cursor.execute(query)
        return cursor.fetchall()


def parse_emotion_vectors(
    diaries: List[Dict],
) -> Tuple[List[List[float]], List[int], List[int]]:
    """감정 벡터 파싱 및 검증"""
    vectors = []
    diary_ids = []
    user_ids = []

    for diary in diaries:
        try:
            # JSON 문자열을 파이썬 리스트로 변환
            if isinstance(diary["emotion_vector"], str):
                emotion_vector = json.loads(diary["emotion_vector"])
            else:
                emotion_vector = diary["emotion_vector"]

            # 벡터 길이 검증
            if len(emotion_vector) == 10:
                vectors.append([float(x) for x in emotion_vector])
                diary_ids.append(diary["id"])
                user_ids.append(diary["user_id"])
            else:
                logger.warning(
                    f"일기 {diary['id']}: 잘못된 벡터 길이 {len(emotion_vector)}"
                )

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"일기 {diary['id']}: 벡터 파싱 실패 - {str(e)}")

    return vectors, diary_ids, user_ids


def perform_clustering(
    vectors: List[List[float]], min_cluster_size: int = 3
) -> Tuple[List[int], List[List[float]]]:
    """K-means 클러스터링 수행"""
    n_samples = len(vectors)

    if n_samples < min_cluster_size:
        logger.warning(f"샘플 수가 너무 적습니다: {n_samples}")
        # 모든 점을 클러스터 0으로 할당, 평균을 중심점으로
        labels = [0] * n_samples
        if vectors:
            n_features = len(vectors[0])
            centroid = [
                sum(vectors[i][j] for i in range(n_samples)) / n_samples
                for j in range(n_features)
            ]
            centroids = [centroid]
        else:
            centroids = [[0.0] * 10]
        return labels, centroids

    # 최적 클러스터 수 결정
    if n_samples < 10:
        n_clusters = 2
    elif n_samples < 50:
        n_clusters = min(5, n_samples // 3)
    else:
        n_clusters = min(10, n_samples // 10)

    logger.info(f"클러스터링 수행: {n_samples}개 샘플, {n_clusters}개 클러스터")

    labels, centroids = simple_kmeans_python(
        vectors=vectors, n_clusters=n_clusters, max_iters=300, random_seed=42
    )

    return labels, centroids


def save_clustering_results(connection, centroids: List[List[float]]) -> Dict[int, int]:
    """클러스터링 결과를 데이터베이스에 저장"""
    cluster_id_mapping = {}

    with connection.cursor() as cursor:
        # 기존 클러스터 캐시 삭제
        cursor.execute("DELETE FROM clustering_cache")

        # 새로운 클러스터 중심점 저장
        for i, centroid in enumerate(centroids):
            centroid_json = json.dumps(centroid)

            cursor.execute(
                """
                INSERT INTO clustering_cache (cluster_id, centroid_vector, member_count, created_at)
                VALUES (%s, %s, %s, NOW())
            """,
                (i, centroid_json, 0),
            )

            cluster_id_mapping[i] = cursor.lastrowid

    connection.commit()
    return cluster_id_mapping


def get_user_current_communities(connection, user_id: int) -> List[int]:
    """사용자의 현재 활성 커뮤니티 조회"""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT community_id 
            FROM community_members 
            WHERE user_id = %s AND is_active = 1
        """,
            (user_id,),
        )
        return [row["community_id"] for row in cursor.fetchall()]


def get_or_create_community(
    connection, cluster_id: int, emotion_theme: str = None
) -> int:
    """클러스터에 해당하는 커뮤니티 조회 또는 생성"""
    with connection.cursor() as cursor:
        # 기존 커뮤니티 조회
        cursor.execute(
            """
            SELECT id FROM communities 
            WHERE name LIKE %s AND is_default = 1
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (f"감정나눔방 #{cluster_id}%",),
        )

        result = cursor.fetchone()
        if result:
            return result["id"]

        # 새 커뮤니티 생성
        community_name = f"감정나눔방 #{cluster_id}"
        description = f"클러스터 {cluster_id}의 비슷한 감정을 가진 사람들의 나눔방"

        cursor.execute(
            """
            INSERT INTO communities (name, description, creator_id, is_default, emotion_theme, created_at)
            VALUES (%s, %s, NULL, 1, %s, NOW())
        """,
            (community_name, description, emotion_theme),
        )

        community_id = cursor.lastrowid
        connection.commit()

        logger.info(f"새 커뮤니티 생성: {community_name} (ID: {community_id})")
        return community_id


def create_community_recommendation_alert(
    connection, user_id: int, community_id: int, similarity_score: float, reason: str
):
    """커뮤니티 추천 알림 생성"""
    with connection.cursor() as cursor:
        # 커뮤니티 정보 조회
        cursor.execute(
            """
            SELECT name, description FROM communities WHERE id = %s
        """,
            (community_id,),
        )
        community = cursor.fetchone()

        if not community:
            return

        message = f"""💫 새로운 나눔방을 추천드려요!

🏠 **{community['name']}**
📝 {community['description']}

🎯 **추천 이유**: {reason}
📊 **감정 유사도**: {similarity_score:.1%}

이 나눔방에 참여하여 비슷한 감정을 가진 분들과 소통해보세요! 
기존 나눔방은 그대로 유지되며, 원하시면 언제든 추가로 참여하실 수 있습니다."""

        # 중복 알림 방지 (최근 24시간 내 같은 커뮤니티 추천 체크)
        cursor.execute(
            """
            SELECT id FROM user_alerts 
            WHERE user_id = %s 
                AND alert_type = 'community_invite'
                AND message LIKE %s
                AND created_at > DATE_SUB(NOW(), INTERVAL 24 HOUR)
        """,
            (user_id, f"%{community['name']}%"),
        )

        if cursor.fetchone():
            logger.info(f"사용자 {user_id}에게 이미 최근 알림 발송됨")
            return

        # 새 알림 생성
        cursor.execute(
            """
            INSERT INTO user_alerts (user_id, alert_type, message, created_at, is_read)
            VALUES (%s, 'community_invite', %s, NOW(), 0)
        """,
            (user_id, message),
        )

        connection.commit()
        logger.info(
            f"커뮤니티 추천 알림 생성: 사용자 {user_id} → 커뮤니티 {community_id}"
        )


def process_user_community_recommendations(
    connection,
    user_id: int,
    user_vector: List[float],
    centroids: List[List[float]],
    current_cluster: int,
):
    """사용자별 커뮤니티 추천 처리"""
    try:
        # 현재 활성 커뮤니티 조회
        current_communities = get_user_current_communities(connection, user_id)

        # 모든 클러스터와의 유사도 계산
        similarities = []
        for centroid in centroids:
            similarity = cosine_similarity(user_vector, centroid)
            similarities.append(similarity)

        # 상위 3개 클러스터 선택
        cluster_scores = [(i, score) for i, score in enumerate(similarities)]
        cluster_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations_made = 0
        for cluster_id, similarity_score in cluster_scores[:3]:
            if recommendations_made >= 2:  # 최대 2개 추천
                break

            # 최소 유사도 조건 (30% 이상)
            if similarity_score < 0.05:
                continue

            # 해당 클러스터의 커뮤니티 조회/생성
            community_id = get_or_create_community(connection, cluster_id)

            # 이미 가입된 커뮤니티면 스킵
            if community_id in current_communities:
                continue

            # 추천 이유 생성
            if cluster_id == current_cluster:
                reason = "현재 감정과 가장 유사한 커뮤니티입니다"
            elif similarity_score > 0.8:
                reason = "감정 패턴이 매우 유사합니다"
            elif similarity_score > 0.6:
                reason = "비슷한 감정을 자주 경험합니다"
            elif similarity_score > 0.4:
                reason = "관심 있을 만한 감정 주제입니다"
            else:
                reason = "새로운 감정 경험을 나눌 수 있는 커뮤니티입니다"

            # 추천 알림 생성
            create_community_recommendation_alert(
                connection, user_id, community_id, similarity_score, reason
            )

            recommendations_made += 1

    except Exception as e:
        logger.error(f"사용자 {user_id} 추천 처리 실패: {str(e)}")


def update_clustering_cache_member_counts(
    connection, labels: List[int], user_ids: List[int]
):
    """클러스터별 멤버 수 업데이트"""
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    with connection.cursor() as cursor:
        for cluster_id, count in cluster_counts.items():
            cursor.execute(
                """
                UPDATE clustering_cache 
                SET member_count = %s 
                WHERE cluster_id = %s
            """,
                (count, cluster_id),
            )

    connection.commit()


def process_individual_analysis(
    connection, diary_id: int, user_id: int, emotion_vector: List[float]
):
    """개별 일기 분석 처리 (SQS 트리거용)"""
    try:
        logger.info(f"개별 일기 분석 시작: diary_id={diary_id}, user_id={user_id}")

        # 현재 클러스터 중심점들 조회
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT cluster_id, centroid_vector 
                FROM clustering_cache 
                ORDER BY cluster_id
            """
            )
            clusters = cursor.fetchall()

        if not clusters:
            logger.warning(
                "클러스터 캐시가 비어있습니다. 전체 재클러스터링을 실행합니다."
            )
            return process_batch_clustering(connection)

        # 클러스터 중심점들을 리스트로 변환
        centroids = []
        for cluster in clusters:
            if isinstance(cluster["centroid_vector"], str):
                centroid = json.loads(cluster["centroid_vector"])
            else:
                centroid = cluster["centroid_vector"]
            centroids.append([float(x) for x in centroid])

        # 가장 유사한 클러스터 찾기
        similarities = []
        for centroid in centroids:
            similarity = cosine_similarity(emotion_vector, centroid)
            similarities.append(similarity)

        best_cluster = similarities.index(max(similarities))
        best_similarity = similarities[best_cluster]

        logger.info(
            f"사용자 {user_id}의 최적 클러스터: {best_cluster} (유사도: {best_similarity:.3f})"
        )

        # 커뮤니티 추천 처리
        process_user_community_recommendations(
            connection, user_id, emotion_vector, centroids, best_cluster
        )

        return {
            "user_id": user_id,
            "recommended_cluster": best_cluster,
            "similarity_score": float(best_similarity),
        }

    except Exception as e:
        logger.error(f"개별 분석 실패: {str(e)}")
        raise


def process_batch_clustering(connection):
    """전체 일기 데이터로 배치 클러스터링 수행"""
    try:
        logger.info("배치 클러스터링 시작")

        # 완료된 모든 일기 조회
        diaries = get_completed_diaries(connection)

        if len(diaries) < 3:
            logger.warning("클러스터링을 위한 충분한 데이터가 없습니다.")
            return {"message": "데이터 부족으로 클러스터링 불가"}

        # 감정 벡터 파싱
        vectors, diary_ids, user_ids = parse_emotion_vectors(diaries)

        if len(vectors) < 3:
            logger.warning("유효한 감정 벡터가 부족합니다.")
            return {"message": "유효한 데이터 부족으로 클러스터링 불가"}

        # 클러스터링 수행
        labels, centroids = perform_clustering(vectors)

        # 결과 저장
        save_clustering_results(connection, centroids)
        update_clustering_cache_member_counts(connection, labels, user_ids)

        # 각 사용자별 커뮤니티 추천 처리
        processed_users = set()
        for i, (user_id, label) in enumerate(zip(user_ids, labels)):
            if user_id not in processed_users:
                user_vector = vectors[i]
                process_user_community_recommendations(
                    connection, user_id, user_vector, centroids, label
                )
                processed_users.add(user_id)

        logger.info(
            f"배치 클러스터링 완료: {len(centroids)}개 클러스터, {len(processed_users)}명 처리"
        )

        publish_clustering_completion(
            total_diaries=len(diaries),
            total_users=len(processed_users), 
            clusters_created=len(centroids)
        )

        return {
            "message": "클러스터링 완료",
            "clusters_created": len(centroids),
            "users_processed": len(processed_users),
        }

    except Exception as e:
        logger.error(f"배치 클러스터링 실패: {str(e)}")
        raise



def lambda_handler(event, context):
    """Lambda 메인 핸들러 - API Gateway 지원 추가"""
    connection = None
    
    try:
        logger.info(f"Lambda 실행 시작: {json.dumps(event)}")
        
        # DB 연결
        connection = get_db_connection()
        
        # SQS 이벤트 처리 (개별 일기 분석)
        if 'Records' in event:
            results = []
            
            for record in event['Records']:
                try:
                    # SQS 메시지 파싱
                    message_body = json.loads(record['body'])
                    
                    # SNS 메시지인 경우 Message 필드 파싱
                    if 'Message' in message_body:
                        message_data = json.loads(message_body['Message'])
                    else:
                        message_data = message_body
                    
                    # 필수 정보 추출
                    diary_id = message_data.get('diaryId')
                    user_id = message_data.get('userId')
                    event_type = message_data.get('event')
                    
                    # 유효성 검사
                    if event_type != 'diary_analyzed':
                        logger.warning(f"잘못된 이벤트 타입: {event_type}")
                        continue
                    
                    if not diary_id or not user_id:
                        logger.warning(f"필수 정보 누락: diary_id={diary_id}, user_id={user_id}")
                        continue
                    
                    # DB에서 감정 벡터 조회
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            SELECT emotion_vector 
                            FROM diaries 
                            WHERE id = %s AND user_id = %s AND analysis_status = 'completed'
                        """, (diary_id, user_id))
                        
                        diary_result = cursor.fetchone()
                        
                        if not diary_result:
                            logger.warning(f"일기를 찾을 수 없음: diary_id={diary_id}")
                            continue
                        
                        # 감정 벡터 파싱
                        emotion_vector = diary_result['emotion_vector']
                        if isinstance(emotion_vector, str):
                            emotion_vector = json.loads(emotion_vector)
                        
                        if len(emotion_vector) != 10:
                            logger.warning(f"잘못된 감정 벡터 길이: {len(emotion_vector)}")
                            continue
                        
                        emotion_vector = [float(x) for x in emotion_vector]
                    
                    # 개별 분석 처리
                    result = process_individual_analysis(connection, diary_id, user_id, emotion_vector)
                    results.append(result)
                    
                except Exception as record_error:
                    logger.error(f"레코드 처리 오류: {str(record_error)}")
                    continue
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': '개별 분석 완료',
                    'processed': len(results),
                    'results': results
                })
            }
        
        # EventBridge/CloudWatch Events (배치 처리)
        elif event.get('source') == 'aws.events':
            result = process_batch_clustering(connection)
            
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }
        
        # API Gateway 호출 처리
        elif 'httpMethod' in event:
            http_method = event.get('httpMethod', '')
            resource_path = event.get('resource', '')
            path_parameters = event.get('pathParameters', {})
            
            logger.info(f"API Gateway 호출: {http_method} {resource_path}")
            
            # CORS 헤더 설정
            headers = {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
            }
            
            # OPTIONS 메서드 (CORS preflight)
            if http_method == 'OPTIONS':
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({'message': 'CORS preflight'})
                }
            
            # GET /clustering/user/{userId} - 개별 사용자 분석
            if http_method == 'GET' and resource_path == '/clustering/user/{userId}':
                user_id = path_parameters.get('userId')
                
                if not user_id:
                    return {
                        'statusCode': 400,
                        'headers': headers,
                        'body': json.dumps({'error': 'userId 파라미터가 필요합니다'})
                    }
                
                try:
                    user_id = int(user_id)
                except ValueError:
                    return {
                        'statusCode': 400,
                        'headers': headers,
                        'body': json.dumps({'error': '유효하지 않은 userId입니다'})
                    }
                
                # 사용자의 최신 일기 조회
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, emotion_vector 
                        FROM diaries 
                        WHERE user_id = %s AND analysis_status = 'completed'
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (user_id,))
                    
                    diary = cursor.fetchone()
                    if not diary:
                        return {
                            'statusCode': 404,
                            'headers': headers,
                            'body': json.dumps({'error': f'사용자 {user_id}의 분석된 일기가 없습니다'})
                        }
                    
                    emotion_vector = diary['emotion_vector']
                    if isinstance(emotion_vector, str):
                        emotion_vector = json.loads(emotion_vector)
                    
                    emotion_vector = [float(x) for x in emotion_vector]
                
                result = process_individual_analysis(connection, diary['id'], user_id, emotion_vector)
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'message': '개별 사용자 분석 완료',
                        'user_id': user_id,
                        'result': result
                    })
                }
            
            # POST /clustering/batch - 전체 배치 분석
            elif http_method == 'POST' and resource_path == '/clustering/batch':
                result = process_batch_clustering(connection)
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'message': '배치 클러스터링 완료',
                        'result': result
                    })
                }
            
            # 지원하지 않는 경로
            else:
                return {
                    'statusCode': 404,
                    'headers': headers,
                    'body': json.dumps({'error': '지원하지 않는 API 경로입니다'})
                }
        
        # 기존 방식 (하위 호환성)
        else:
            # 요청 타입에 따라 분기
            request_type = event.get('request_type', 'batch')
            
            if request_type == 'batch':
                result = process_batch_clustering(connection)
            else:
                # 특정 사용자 분석
                user_id = event.get('user_id')
                if not user_id:
                    raise ValueError("user_id가 필요합니다")
                
                # 사용자의 최신 일기 조회
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, emotion_vector 
                        FROM diaries 
                        WHERE user_id = %s AND analysis_status = 'completed'
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (user_id,))
                    
                    diary = cursor.fetchone()
                    if not diary:
                        raise ValueError(f"사용자 {user_id}의 분석된 일기가 없습니다")
                    
                    emotion_vector = diary['emotion_vector']
                    if isinstance(emotion_vector, str):
                        emotion_vector = json.loads(emotion_vector)
                    
                    emotion_vector = [float(x) for x in emotion_vector]
                
                result = process_individual_analysis(connection, diary['id'], user_id, emotion_vector)
            
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }
    
    except Exception as e:
        logger.error(f"Lambda 실행 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # API Gateway 오류 응답
        if 'httpMethod' in event:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': str(e)})
            }
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    
    finally:
        if connection:
            connection.close()


# 테스트용 로컬 실행
if __name__ == "__main__":
    # 테스트 이벤트
    test_event = {
        "Records": [
            {
                "body": json.dumps(
                    {
                        "Message": json.dumps(
                            {
                                "diaryId": 811,
                                "userId": 2,
                                "emotionVector": [
                                    0.1,
                                    0.7,
                                    0.0,
                                    0.3,
                                    0.0,
                                    0.0,
                                    0.2,
                                    0.0,
                                    0.1,
                                    0.0,
                                ],
                                "event": "diary_analyzed",
                            }
                        )
                    }
                )
            }
        ]
    }

    print("테스트 실행 결과:")
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
