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
                        "footer": "일기 자동화 시스템"
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
                        "footer": "일기 자동화 시스템"
                    }]
                }
            
            elif subject == 'user_communities_assigned':
                # 사용자 커뮤니티 가입 알림
                user_id = sns_message.get('user_id', 'N/A')
                diary_id = sns_message.get('diary_id', 'N/A')
                assigned_cluster = sns_message.get('assigned_cluster', 'N/A')
                similarity_score = sns_message.get('similarity_score', 0)
                communities_joined = sns_message.get('communities_joined', [])
                communities_joined_count = sns_message.get('communities_joined_count', 0)
                
                # 가입된 커뮤니티 목록 텍스트 생성
                if communities_joined_count > 0:
                    community_list = []
                    for community in communities_joined:
                        cluster_id = community.get('cluster_id', 'N/A')
                        community_id = community.get('community_id', 'N/A')
                        reason = community.get('reason', '자동 매칭')
                        score = community.get('similarity_score', 0)
                        community_list.append(f"• 감정나눔방 #{cluster_id} (ID: {community_id}) - {reason} ({score:.1%})")
                    
                    communities_text = "\n".join(community_list)
                    status_text = f"{communities_joined_count}개 커뮤니티 가입 완료"
                    color = "#ff6b35"
                    diary_bot_message = f"일기 분석 후 감정 클러스터링이 완료되었습니다! 클러스터 #{assigned_cluster}에 할당되었고, {communities_joined_count}개의 새로운 감정나눔방에 참여하게 되었습니다."
                else:
                    communities_text = "가입 조건에 맞는 커뮤니티가 없습니다"
                    status_text = "커뮤니티 가입 없음"
                    color = "#999999"
                    diary_bot_message = f"일기 분석 후 감정 클러스터링이 완료되었습니다! 클러스터 #{assigned_cluster}에 할당되었지만, 이미 적절한 감정나눔방에 참여하고 계시네요."
                
                # 커뮤니티봇 메시지
                slack_message = {
                    "channel": channel,
                    "username": "커뮤니티봇",
                    "text": "사용자 커뮤니티 가입이 처리되었습니다!",
                    "attachments": [{
                        "color": color,
                        "fields": [
                            {
                                "title": "사용자 ID",
                                "value": f"#{user_id}",
                                "short": True
                            },
                            {
                                "title": "일기 ID",
                                "value": f"#{diary_id}",
                                "short": True
                            },
                            {
                                "title": "할당된 클러스터",
                                "value": f"클러스터 #{assigned_cluster}",
                                "short": True
                            },
                            {
                                "title": "감정 유사도",
                                "value": f"{similarity_score:.1%}",
                                "short": True
                            },
                            {
                                "title": "처리 상태",
                                "value": status_text,
                                "short": False
                            },
                            {
                                "title": "가입된 커뮤니티",
                                "value": communities_text,
                                "short": False
                            },
                            {
                                "title": "처리 완료 시간",
                                "value": time_str,
                                "short": True
                            }
                        ],
                        "footer": "일기 자동화 시스템 - 커뮤니티 매칭"
                    }]
                }
                
                # 커뮤니티봇 메시지 전송
                response = http.request(
                    'POST',
                    webhook_url,
                    body=json.dumps(slack_message, ensure_ascii=False),
                    headers={'Content-Type': 'application/json'}
                )
                
                # 일기봇 추가 메시지
                diary_bot_slack_message = {
                    "channel": channel,
                    "username": "일기봇",
                    "text": diary_bot_message,
                    "attachments": [{
                        "color": "#4CAF50",
                        "fields": [
                            {
                                "title": "사용자 ID",
                                "value": f"#{user_id}",
                                "short": True
                            },
                            {
                                "title": "일기 ID",
                                "value": f"#{diary_id}",
                                "short": True
                            },
                            {
                                "title": "할당된 클러스터",
                                "value": f"클러스터 #{assigned_cluster}",
                                "short": True
                            },
                            {
                                "title": "감정 유사도",
                                "value": f"{similarity_score:.1%}",
                                "short": True
                            }
                        ],
                        "footer": "일기 자동화 시스템 - 클러스터링 완료"
                    }]
                }
                
                # 일기봇 메시지 전송 (약간의 지연 후)
                response2 = http.request(
                    'POST',
                    webhook_url,
                    body=json.dumps(diary_bot_slack_message, ensure_ascii=False),
                    headers={'Content-Type': 'application/json'}
                )
                
                print(f"Slack 전송 결과 - 커뮤니티봇: {response.status}, 일기봇: {response2.status}")
                continue  # 이미 두 개 메시지를 모두 보냈으므로 다음 레코드로
                
            elif subject == 'batch_community_assignment_completed':
                # 사용자 커뮤니티 가입 알림
                user_id = sns_message.get('user_id', 'N/A')
                diary_id = sns_message.get('diary_id', 'N/A')
                assigned_cluster = sns_message.get('assigned_cluster', 'N/A')
                similarity_score = sns_message.get('similarity_score', 0)
                communities_joined = sns_message.get('communities_joined', [])
                communities_joined_count = sns_message.get('communities_joined_count', 0)
                
                # 가입된 커뮤니티 목록 텍스트 생성
                if communities_joined_count > 0:
                    community_list = []
                    for community in communities_joined:
                        cluster_id = community.get('cluster_id', 'N/A')
                        community_id = community.get('community_id', 'N/A')
                        reason = community.get('reason', '자동 매칭')
                        score = community.get('similarity_score', 0)
                        community_list.append(f"• 감정나눔방 #{cluster_id} (ID: {community_id}) - {reason} ({score:.1%})")
                    
                    communities_text = "\n".join(community_list)
                    status_text = f"{communities_joined_count}개 커뮤니티 가입 완료"
                    color = "#ff6b35"
                else:
                    communities_text = "가입 조건에 맞는 커뮤니티가 없습니다"
                    status_text = "커뮤니티 가입 없음"
                    color = "#999999"
                
                slack_message = {
                    "channel": channel,
                    "username": "커뮤니티봇",
                    "text": "사용자 커뮤니티 가입이 처리되었습니다!",
                    "attachments": [{
                        "color": color,
                        "fields": [
                            {
                                "title": "사용자 ID",
                                "value": f"#{user_id}",
                                "short": True
                            },
                            {
                                "title": "일기 ID",
                                "value": f"#{diary_id}",
                                "short": True
                            },
                            {
                                "title": "할당된 클러스터",
                                "value": f"클러스터 #{assigned_cluster}",
                                "short": True
                            },
                            {
                                "title": "감정 유사도",
                                "value": f"{similarity_score:.1%}",
                                "short": True
                            },
                            {
                                "title": "처리 상태",
                                "value": status_text,
                                "short": False
                            },
                            {
                                "title": "가입된 커뮤니티",
                                "value": communities_text,
                                "short": False
                            },
                            {
                                "title": "처리 완료 시간",
                                "value": time_str,
                                "short": True
                            }
                        ],
                        "footer": "감정 일기 시스템 - 커뮤니티 매칭"
                    }]
                }
                
            elif subject == 'batch_community_assignment_completed':
                # 배치 커뮤니티 가입 완료 알림
                total_diaries = sns_message.get('total_diaries', 0)
                total_users = sns_message.get('total_users', 0)
                clusters_created = sns_message.get('clusters_created', 0)
                total_assignments = sns_message.get('total_community_assignments', 0)
                users_with_new_communities = sns_message.get('users_with_new_communities', 0)
                processing_time = sns_message.get('processing_time_seconds', 0)
                
                slack_message = {
                    "channel": channel,
                    "username": "자동처리봇",
                    "text": "자동 커뮤니티 가입 처리가 완료되었습니다!",
                    "attachments": [{
                        "color": "#9c27b0",
                        "fields": [
                            {
                                "title": "처리된 일기",
                                "value": f"{total_diaries}개",
                                "short": True
                            },
                            {
                                "title": "처리된 사용자",
                                "value": f"{total_users}명",
                                "short": True
                            },
                            {
                                "title": "생성된 클러스터",
                                "value": f"{clusters_created}개",
                                "short": True
                            },
                            {
                                "title": "총 커뮤니티 가입",
                                "value": f"{total_assignments}건",
                                "short": True
                            },
                            {
                                "title": "커뮤니티에 가입한 사용자",
                                "value": f"{users_with_new_communities}명",
                                "short": True
                            },
                            {
                                "title": "처리 시간",
                                "value": f"{processing_time:.1f}초",
                                "short": True
                            },
                            {
                                "title": "완료 시간",
                                "value": time_str,
                                "short": False
                            }
                        ],
                        "footer": "일기 자동화 시스템 - 자동 처리"
                    }]
                }
                
            elif subject == 'batch_clustering_completed':
                # 기존 배치 클러스터링 완료 알림 (호환성 유지)
                total_users = sns_message.get('totalUsers', 0)
                clustered_users = sns_message.get('clusteredUsers', 0)
                clusters_created = sns_message.get('clustersCreated', 0)
                
                slack_message = {
                    "channel": channel,
                    "username": "클러스터링봇",
                    "text": "클러스터링이 완료되었습니다!",
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
                        "footer": "감정 일기 시스템"
                    }]
                }
            else:
                # 알 수 없는 메시지 타입
                slack_message = {
                    "channel": channel,
                    "username": "시스템봇",
                    "text": f"새로운 이벤트가 발생했습니다: {subject}",
                    "attachments": [{
                        "color": "#ffc107",
                        "fields": [
                            {
                                "title": "이벤트 타입",
                                "value": subject,
                                "short": True
                            },
                            {
                                "title": "발생 시간",
                                "value": time_str,
                                "short": True
                            },
                            {
                                "title": "메시지 내용",
                                "value": f"```{json.dumps(sns_message, ensure_ascii=False, indent=2)}```",
                                "short": False
                            }
                        ],
                        "footer": "일기 자동화 시스템"
                    }]
                }
            
            # Slack으로 전송 (user_communities_assigned는 이미 위에서 처리됨)
            if subject != 'user_communities_assigned':
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