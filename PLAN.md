# 소설 블럭 검색 시스템 구현 가이드

## 환경
- OS: Windows 11, powershell
- python 3.12

## 🎯 목적
장편 소설 생성 시 현재 스토리와 관련된 과거 블럭들을 빠르게 찾아 컨텍스트 오버플로우를 방지하는 시스템

## 📊 데이터 구조
- **저장 대상**: `block_data.json`의 `LLMOutput` 텍스트만
- **메타데이터**: 최소한 (`block_id`, `word_count`, `created_at`)

## 🛠️ 구현 과정

### ✅ 1. 환경 설정 (완료됨)
```powershell
# 1) 레포지토리 클론 (완료)
# 2) 가상환경 생성
python -m venv venv
.\venv\Scripts\activate

# 3) 의존성 설치
pip install -r requirements.txt

# 4) .env 파일 생성
# 프로젝트 루트에 .env 파일을 만들고 다음 내용 추가:
```

**`.env` 파일 내용:**
```env
EMBEDDING_MODEL=intfloat/multilingual-e5-large
COLLECTION_NAME=novel_blocks
QDRANT_URL=http://localhost:6333
QDRANT_SEARCH_LIMIT=10
TOOL_STORE_DESCRIPTION="소설 블럭을 벡터 데이터베이스에 저장"
TOOL_FIND_DESCRIPTION="스토리와 유사한 과거 블럭들을 검색"
```

```powershell
# 5) Docker Desktop 설치
winget install --id Docker.DockerDesktop -e

# 6) Qdrant 실행 (완료)
docker run -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 🔄 2. 블럭 처리 스크립트 생성 (완료됨)

#### A. `scripts/preprocess_blocks.py` - 전처리 스크립트
- **기능**: 원본 JSON에서 `LLMOutput`만 추출하여 새로운 JSON 파일 생성
- **사용법**: `python scripts/preprocess_blocks.py <입력파일> [출력파일]`
- **예시**: `python scripts/preprocess_blocks.py block_data.json`
- **출력**: 블럭 개수, 샘플 미리보기, 다음 단계 안내

#### B. `scripts/store_blocks.py` - 저장 스크립트  
- **기능**: 전처리된 JSON 파일의 블럭들을 Qdrant 벡터 DB에 저장
- **사용법**: `python scripts/store_blocks.py [전처리된_파일]`
- **특징**: 진행상황 표시, 오류 처리, 환경변수 자동 로드

#### C. `scripts/search_blocks.py` - 검색 스크립트
- **기능**: 현재 스토리와 의미적으로 유사한 블럭들을 검색
- **사용법**: `python scripts/search_blocks.py "현재 스토리 내용" [개수]`
- **출력**: 관련 블럭들을 유사도 순으로 표시 (ID, 내용, 메타데이터)

### 🎯 3. 실제 사용 방법

#### 방법 1: 데이터 전처리 및 저장
```powershell
# 1단계: 원본 데이터에서 LLMOutput만 추출
python scripts/preprocess_blocks.py block_data.json
# → block_data_preprocessed.json 생성됨

# 2단계: 전처리된 데이터를 Qdrant에 저장
python scripts/store_blocks.py block_data_preprocessed.json
```

#### 방법 2: 관련 블럭 검색 (커맨드라인)
```powershell
python scripts/search_blocks.py "회복 뒤 말론이 처음 마법을 배움"
# → 관련 블럭들이 의미적 유사성 순으로 출력됨
```

#### 방법 3: LLM과 연동 (Claude Desktop)
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "novel-blocks": {
      "command": "python", 
      "args": ["-m", "mcp_server_qdrant.main"],
      "cwd": "C:/Users/ink0513/Documents/GitHub/mcp-server-qdrant",
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "novel_blocks",
        "EMBEDDING_MODEL": "intfloat/multilingual-e5-large"
      }
    }
  }
}
```

### 📁 현재 프로젝트 구조
```
mcp-server-qdrant/
├── .env                     # 환경변수 (생성 필요)
├── requirements.txt         # ✅ 완료
├── block_data.json         # 샘플 데이터
├── venv/                   # ✅ 완료  
├── qdrant_storage/         # Qdrant 데이터 (자동 생성)
├── scripts/                   # ✅ 완료
│   ├── preprocess_blocks.py       # 1단계: LLMOutput 추출 → JSON
│   ├── store_blocks.py            # 2단계: 전처리된 JSON → Qdrant 저장
│   └── search_blocks.py           # 3단계: 관련 블럭 검색
├── block_data_preprocessed.json   # 🔄 전처리된 데이터 (생성될 파일)
└── src/mcp_server_qdrant/  # 기존 MCP 서버
```

## ⚡ 핵심 특징
- **수학적 검색**: LLM 없이 벡터 유사도로 관련 블럭 찾기
- **한국어 최적화**: 다국어 임베딩 모델 사용
- **컨텍스트 관리**: 만개 블럭 중 관련 블럭만 선별
- **두 가지 사용법**: 커맨드라인 직접 검색 + LLM 도구 연동

## 🚀 다음 할 일
1. 스크립트 테스트 및 성능 확인