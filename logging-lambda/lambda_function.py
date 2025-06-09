import json
import urllib3
import os
from datetime import datetime, timezone, timedelta

def lambda_handler(event, context):
    # 환경변수에서 설정값 가져오기 (필수값들 - 기본값 없음)
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    channel = os.environ.get('SLACK_CHANNEL')
    timezone_offset = int(os.environ.get('TIMEZONE_OFFSET', '9'))  # KST = UTC+9
    
    # 필수 환경변수 확인
    if not webhook_url:
        print("SLACK_WEBHOOK_URL 환경변수가 설정되지 않았습니다.")
        return {'statusCode': 400, 'body': 'Webhook URL not configured'}
    
    if not channel:
        print("SLACK_CHANNEL 환경변수가 설정되지 않았습니다.")
        return {'statusCode': 400, 'body': 'Slack channel not configured'}
    
    try:
        http = urllib3.PoolManager()
        
        for record in event['Records']:
            # SNS 메시지 파싱
            sns_message = json.loads(record['Sns']['Message'])
            subject = record['Sns']['Subject']
            timestamp = record['Sns']['Timestamp']
            
            # UTC를 한국시간으로 변환
            utc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            kst_time = utc_time + timedelta(hours=timezone_offset)
            time_str = kst_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # 메시지 타입에 따라 다른 포맷
            if subject == 'diary_created':
                slack_message = {
                    "channel": channel,
                    "username": "일기봇",
                    "text": "새로운 일기가 작성되었습니다!",
                    "attachments": [{
                        "color": "#36a64f",
                        "fields": [
                            {
                                "title": "사용자 ID",
                                "value": f"#{sns_message.get('userId', 'N/A')}",
                                "short": True
                            },
                            {
                                "title": "일기 ID", 
                                "value": f"#{sns_message.get('diaryId', 'N/A')}",
                                "short": True
                            },
                            {
                                "title": "상태",
                                "value": "분석 중...",
                                "short": True
                            },
                            {
                                "title": "작성 시간",
                                "value": time_str,
                                "short": True
                            }
                        ],
                        "footer": "감정 일기 시스템"
                    }]
                }
                
            elif subject == 'diary_analyzed':
                # 감정 벡터에서 주요 감정 추출 (상위 2개)
                emotion_vector = sns_message.get('emotionVector', [])
                emotions = [
                    "기쁨", "슬픔", "분노", "불안", "설렘",
                    "지루함", "외로움", "만족", "실망", "무기력"
                ]
                
                if len(emotion_vector) == 10:
                    # 감정 점수와 라벨을 묶어서 정렬
                    emotion_scores = list(zip(emotions, emotion_vector))
                    emotion_scores.sort(key=lambda x: x[1], reverse=True)
                    top_emotions = [emotion_scores[0][0], emotion_scores[1][0]]
                    primary_emotion_text = f"{top_emotions[0]}, {top_emotions[1]}"
                else:
                    primary_emotion_text = "분석 완료"
                
                slack_message = {
                    "channel": channel,
                    "username": "일기봇",
                    "text": "일기 분석이 완료되었습니다!",
                    "attachments": [{
                        "color": "#0066cc",
                        "fields": [
                            {
                                "title": "사용자 ID",
                                "value": f"#{sns_message.get('userId', 'N/A')}",
                                "short": True
                            },
                            {
                                "title": "일기 ID",
                                "value": f"#{sns_message.get('diaryId', 'N/A')}",
                                "short": True
                            },
                            {
                                "title": "상태",
                                "value": "분석 완료",
                                "short": True
                            },
                            {
                                "title": "주요 감정",
                                "value": primary_emotion_text,
                                "short": True
                            },
                            {
                                "title": "분석 완료 시간",
                                "value": time_str,
                                "short": False
                            }
                        ],
                        "footer": "감정 일기 시스템"
                    }]
                }
                
            elif subject == 'batch_clustering_completed':
                # 배치 클러스터링 완료 알림
                total_users = sns_message.get('totalUsers', 0)
                clustered_users = sns_message.get('clusteredUsers', 0)
                clusters_created = sns_message.get('clustersCreated', 0)
                
                slack_message = {
                    "channel": channel,
                    "username": "클러스터링봇",
                    "text": "배치 클러스터링이 완료되었습니다!",
                    "attachments": [{
                        "color": "#ff6b35",
                        "fields": [
                            {
                                "title": "전체 사용자",
                                "value": f"{total_users}명",
                                "short": True
                            },
                            {
                                "title": "클러스터링된 사용자",
                                "value": f"{clustered_users}명",
                                "short": True
                            },
                            {
                                "title": "생성된 클러스터",
                                "value": f"{clusters_created}개",
                                "short": True
                            },
                            {
                                "title": "처리 완료 시간",
                                "value": time_str,
                                "short": True
                            }
                        ],
                        "footer": "감정 일기 시스템 - 배치 처리"
                    }]
                }
            else:
                # 알 수 없는 메시지 타입
                slack_message = {
                    "channel": channel,
                    "username": "일기봇",
                    "text": f"알 수 없는 이벤트: {subject}",
                    "attachments": [{
                        "color": "warning",
                        "text": f"```{json.dumps(sns_message, ensure_ascii=False, indent=2)}```"
                    }]
                }
            
            # Slack으로 전송
            response = http.request(
                'POST',
                webhook_url,
                body=json.dumps(slack_message, ensure_ascii=False),
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Slack 전송 결과: {response.status}")
            if response.status != 200:
                print(f"Slack 전송 실패: {response.data}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'{len(event["Records"])}개 메시지 처리 완료'
            })
        }
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }