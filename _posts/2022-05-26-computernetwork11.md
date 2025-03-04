---
title: '[Blog]컴퓨터 네트워크 11장 정리'
layout: single
categories:
  - 시험 요약 정리
tag:
  - Blog
  - Computernetwork
  - studyalone
toc: true
toc_label: "on this page"
toc_sticky: true
published : false
---
⏰ 이 내용은 데이터통신 개정 2판을 참고해 정리한 내용입니다.
# Intro
- 데이터 링크층의 주요 기능
  - 오류제어, 흐름제어
  - 프레임짜기
  - 프로토콜

# 프레임짜기
- 비트를 프레임안에 만들어 넣어 프레임이 다른 프레임과 구분되도록함
> __고정길이 프레임__
- ATM, 광역 네트워크
- 프레임 길이 자체가 경계이기 때문에 따로 프레임 경계가 필요없음
> __가변 길이 프레임
- 주로 LAN에 사용
- 프레임의 시작과 끝을 알리는 경계가 필요함
1️⃣ 문자 중심 프로토콜
- 전달 프레임이 바이트 중심
- 시작과 마지막에 플래그 추가
- 프레임에서 프래그와 같은 비트 패턴이 나와서 플래그로 인식되는 것을 방지하기 위해 프레임에 나오는 플래그 패턴 앞에 탈출문자를 삽입한다.
  - 수신측은 탈출문자 뒤에 오는 문자를 일반 데이터로 받음
2️⃣ 비트 중심 프로토콜
- 전송 프레임이 비트 중심
- 프레임 시작과 끝에 플래그 비트 01111110 추가
- 프레임에서 플래그와 같은 패턴이 나와서 플래그로 인식되는 것을 방지하기 위해 0다음에 1이 다섯개 나오면 뒤에 0비트 추가해줌- 

# 흐름제어와 오류제어
> 흐름제어
- 송신측에서 수신측의 확인 응답 받기 전에 보낼 수 있는 데이터의 양을 제한
> 오류제어
- 오류 발생 시 데이터 재전송 요구하는 ARQ기반

# 프로토콜 종류
- 잡음 있는 채널(일반적 채널)
  - 정지 후 대기 ARQ 프로토콜
  - N-복귀 ARQ 프로토콜
  - 선택적 반복 ARQ 프로토콜
- 잡음 없는 채널(프레임, 손실, 손상 없는 이상적 채널)
  - 가장 단순한 프로토콜
  - 정지 후 대기 프로토콜

# 잡음 없는 채널
1. 가장 단순한 프로토콜
- 오류제어나 흐름제어가 없음
- 수신측에서 수신프레임 처리를 매우 빠르게하며 프레임이 넘치지 않는다는 가정 하에 프레임 수신 즉시 헤더 제거 후 네트워크층에 전달
- 송신측 n개의 이벤트 동안 수신측 n개의 이벤트

2. 정지 후 대기 프로토콜
- 수신측 프레임이 넘치는 것을 알리는 방법
- 송신측은 프레임 송신 후 수신측의 확인응답을 기다리며 확인응답 후에 다음 프레임 전송
- 수신측 n개의 이벤트 발생 동안 송신측 2n개의 이벤트 발생

# 잡음 있는 채널
- ARQ : 오류 발생 시 송신측에 재전송 요구
> __정지 후 대기 ARQ__
- __송신측__
  - 프레임 보낸 뒤 사본 저장해 두었다가 타이머가 종료될 때까지 확인응답(ACK)가 없으면 프레임을 재전송한다.
  - 만약 수신측에 프레임이 전달 되었지만 ACK가 손실된 경우에는 수신 측에서 제어변수 R<sub>n</sub>과 비교해 프레임이 재전송 됐음을 확인한다.
  - 다음 보낼 프레임을 가리키는 S<sub>n</sub>이 있다.(01010101....)
- __수신측__
  - 프래임 받으면 제어변수 R<sub>n</sub>이 다음에 받을 프레임을 가리킨다.(01010101....)
  - 프레임이 도착하지 않으면 송신측이 프레임을 재전송하도록 기다린다.
1. 정상 송수신된 경우
2. ACK가 손실된 경우
  - 송신측은 타이머 종료 시 프레임를 재전송
  - 수신측은 R<sub>n</sub>과 비교해 프레임이 재전송 됨을 안다.
3. 송신 프레임 손상/손실
  - 송신측은 타이머 종료 시 프레임를 재전송
  - 수신측은 송신측이 프레임 재전송하길 기다림
- 회선의 가용도 = (프레임 길이)/(대역폭 지연 곱)

> __N-복귀 ARQ__
- 전송 효율을 높이기위해 송신자가 수신자의 확인응답 기다리는 동안 여러 프레임을 보내 채널의 사용도 높임
- __송신창__
  - 전송완료(S<sub>f</sub>), 전송 중, 전송예정(S<sub>n</sub>) 세 종류로 나뉨
  - 크기가 2<sup>m</sup> - 1 (m=순서 번호 위한 비트 수)
  - 전송 프레임 손상/ 손실 또는 수신 프레임 손실 시 마지막 확인응답받은 프레임 기준으로 이후 프레임 모두 전송
- __수신창__
  - 크기가 1
  - 정상 수신 시 R<sub>n</sub>은 다음에 받을 프레임 가리킴
  - 모든 프레임에 대한 확인응답 필요없이 누적해서 확인응답 보낼 수 있음.
  - 수신 프레임 손상 시 타임아웃 기다리고 재전송받음
  - 정상 수신 시 ACK 전송

> __선택적 복귀 AQR__
- 효율성 높음
- 송신 프레임에 대한 ACK가 없으면 N-복귀 AQR은 마지막 ACK를 기준으로 이후 모든 프레임을 재전송하지만 선택적 AQR은 __정상 송신되지 못한 프레임만을 재전송한다.__
- 추가적인 정렬 등 복잡하고 추가적인 버퍼가 필요하다.
- __송신창__과 __수신창__의 크기가 2<sup>m - 1</sup>로 같다.(m=순서 번호 위한 비트 수)
  - __수신창__은 송신창으로부터 프레임 받아서 추가 버퍼에 보관 후 정렬되지 않은 프레임을 정렬한 뒤 네트워크층으로 전송한다.
- NAK : ACK(확인응답) 반대
- 전송, 재전송 프레임마다 타이머가 필요해 타이머에 번호 붙임

> __피기백킹__
- 양방향 오류 제어가 가능

# HDLC (고차원 데이터 링크 제어)
- 비트 중심 프로토콜
- 비트 전송을 기본으로 하는 범용 데이터 링크 제어 절차
- 오류 제어가 엄밀(CRC 필드 방식, ARQ이용)
- 데이터 링크 제어 프로토콜의 전신
- 모든 통신 방식 지원(전이중, 반이중, 점대점, 다지점)
- 수신측의 확인응답 기다리지 않고 연속적 데이터 전송 가능

> 구성과 전송모드
- 비동기 균형모드(ABM)
  - 전이중, 점대점 통신방식에서 가장 효과적으로 사용됨. 가장 널리 사용됨
  - 균형적인 링크 구성
  - 각 국이 대등하게 명령과 응답하며 동작

> 프레임
- 플레그 필드 : 프레임의 시작과 끝을 나타냄. 수신자를 위한 동기화패턴 역할
- 주소필드 : 종국이 보내면 생성지 주소, 주국이 보내면 목적지 주소
- 제어필드 : 흐름제어, 오류제어에 사용되는 프레임
- 정보필드 : 네트워크층 사용자 정보나 네트워크 관리 정보 담음
- FCS 필드 : 오류 제어에 사용
- `I-프레임` : 정보 프레임. 데이터 전달 담당. 사용자 정보 담음 (플+주+제+정+FCS+플) 
- `U-프레임` : 비번호 프레임. 연결과 해제 담당. 링크 관리 정보 담음 (플+주+제+정+FCS+플)
  - 한 노드에서 `SABM`프레임 보내면 `UA`긍정응답으로 연결 설정
  - 한 노드에서 `DISC`프레임 보내면 `UA`긍정응답으로 연결 해제
- `S-프레임` : 제어 프레임. 제어 담당. 제어 정보만 담음 (플+주+제+FCS+플)

> 피기백킹
- 수신측에서 송신측에 ACK보내는데 대역폭을 차지하게되고 이를 줄이기 위해 피기백킹으로 수신측에서 송신측으로 데이터 전송시 확인응답을 같이 보냄
- 오류 없는 피기백킹
  - 0번 프레임 I-프레임으로부터 시작됨
  - 수신측에서 데이터 다 보내면 S-프레임으로 확인응답 보냄
- 오류 있는 피기백킹
  - 오류 발생하면 수신측에서 S-프레임으로 거부 프레임 전송
  - N-복귀 ARQ프로토콜을 사용해 마지막으로 확인응답 받은 프레임 기준으로 이후 프레임 모두 재전송

# PPP
- HDLC와 유사하지만 PPP가 더 주로 쓰임
- 연결 양 끝 노드에 점대점 직렬 링크 구성해 데이터 전달
- 여러 프로토콜 캡슐화에 사용되며 주로 IP 캡슐화용 프로토콜로 사용

> 서비스
- 장치간 인증방법 지정
- 장치간 교환프레임 형식 지정
- 장치가 링크 설정 교섭하고 데이터교환 방식 지정
- 네트워크 주소 구성 지원
> 빠진 기능
- 에러제어 : 에러검출만 지원함
- 흐름제어 : 수신측을 신경쓰지않고 연속적 데이터 전달 가능

> 지원 기능  
> _(송수신측 협상으로 기능추가 결정함)_
- 암호화 기능
- 압축기능
- 링크통합
- 에러검출
- 인증기능

> 프레임 구성
- HDLC와 프레임구성이 비슷함
- 플레그필드
  - 프레임의 시작과 끝. 수신자의 동기화 패턴역할
- 주소필드
  - 일정한 값 `11111111`을 사용하지만 송수신측간 협상으로 생략 가능
- 제어필드
  - 일정한 값 `11000000`을 사용하지만 송수신측간 협상으로 생략 가능
- 프로토콜필드
  - 프로토콜 정보를 담음
- 페이로드
  - 사용자정보나 다른 정보 전달하며 크기가 가면적
- FCS
  - 2바이트, 4바이트 크기의 표준 CRC
> 천이상태
- __정지__
  - 전송할 데이터 없어 반송파감지가 없는 상태
- __설정__
  - 반송파 감지되고 한 장치에서 통신을 시작하면 설정상태
- __인증__
  - 인증이 필요한 경우와 필요하지 않은 경우로 나뉨.
  - 인증 필요한 경우 ▶️ 인증실패 : 정지상태로 넘어감
- __네트워크__
  - 인증 필요한 경우 ▶️ 인증성공 또는 인증이 필요없는 경우에 바로 네트워크 프로토콜 협상하는 상태
- __열림__
  - 네트워크층 구성이 완료된 경우 데이터 전송이 시작되는 상태
- __연결해제__
  - 모든 데이터 전송 후 연결해제되는 상태. 이후 __정지__상태가 됨.

> 다중화(프로토콜의 종류)
- LCP
  - 링크제어프로토콜
  - PPP데이터링크 개설, 유지, 종료
  - 직렬 연결 회선제어 관리
  - 인증용프로토콜, 프레임 최대길이 등 결정
- AP
  - 인증용 프로토콜
  - 네트워크 사용자 인증이나 단말인증관련 보안서비스 제공
  - PAP
    - 원격지 노드에 인증 시 2-way-handshake사용
    - 평문전달로 보안 취약
    - 송신노드가 username과 그에 맞는 password보내면 수신노드가 인증성공여부를 응답하며 이 경우 라우터의 password는 동일해야함
  - CHAP
    - 원격지 노드에 인증시 3-way-handshake사용
    - 보안 강화되고 단계가 증가
    - 송신노드가 수신노드에 통신의사를 신호로 밝히면 수신노드는 송신노드에 challenge패킷을 만들어 보냄.  
      송신노드는 challenge패킷을 처리해 응답패킷을 전송.
      수신노드는 응답패킷을 처리해 인증성공여부패킷을 전송해줌
- NCP
  - 네트워크 제어 프로토콜
  - 네트워크층에 대한 경로 설정
- DATA
  - IP패킷은 헤더와 사용자데이터로 구성
- LCP

> PPP와 HDLC의 차이
1. PPP는 바이트중심 프로토콜이지만  
  HDLC는 비트중심 프로토콜
2. PPP는 점대점 프로토콜이지만  
  HDLC는 고수준 데이터링크층 프로토콜
3. PPP는 CISCO 장치와도 상호작용하지만  
  HDLC는 비CISCO장치와는 상호작용 불가
4. PPP는 프레임에 프로토콜 필드가 있지만  
  HDLC는 프레임에 프로토콜 필드가 없음
5. PPP는 인증기능을 제공하지만  
  HDLC는 인증기능을 제공하지 
