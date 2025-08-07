#!/usr/bin/env python3
"""
소설 블럭 저장 스크립트

전처리된 JSON 파일의 블럭들을 Qdrant 벡터 데이터베이스에 저장합니다.
MCP 서버의 qdrant-store 도구를 사용합니다.
"""

import json
import sys
import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
load_dotenv()

# MCP 서버 모듈 import
sys.path.append(str(Path(__file__).parent.parent / "src"))
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.settings import EmbeddingProviderSettings

async def store_novel_blocks(preprocessed_file="block_data_preprocessed.json"):
    """
    전처리된 JSON 파일에서 블럭들을 읽어 Qdrant에 저장
    
    Args:
        preprocessed_file: 전처리된 JSON 파일 경로
    """
    
    # 파일 존재 확인
    if not Path(preprocessed_file).exists():
        print(f"❌ 오류: {preprocessed_file} 파일이 존재하지 않습니다.")
        print("💡 먼저 전처리를 실행하세요: python scripts/preprocess_blocks.py")
        return False
    
    try:
        # 전처리된 데이터 로드
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
        
        print(f"📖 전처리된 데이터 로드: {len(blocks)}개 블럭")
        
        # 환경변수에서 설정 로드
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("COLLECTION_NAME", "novel_blocks")
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        
        print(f"🔧 설정:")
        print(f"   Qdrant URL: {qdrant_url}")
        print(f"   컬렉션: {collection_name}")
        print(f"   임베딩 모델: {embedding_model}")
        
        # 임베딩 제공자 생성 (기존 레포 방식 - 환경변수 자동 로드)
        embedding_settings = EmbeddingProviderSettings()
        embedding_provider = create_embedding_provider(embedding_settings)
        
        # Qdrant 커넥터 생성 (기존 레포 방식 - 위치 인자 사용)
        qdrant = QdrantConnector(
            qdrant_url,           # 1. qdrant_url
            None,                 # 2. qdrant_api_key (로컬이므로 None)
            collection_name,      # 3. collection_name  
            embedding_provider,   # 4. embedding_provider
            None,                 # 5. qdrant_local_path (선택적)
        )
        
        print("🚀 Qdrant에 블럭 저장 시작...")
        
        # 각 블럭을 Qdrant에 저장
        stored_count = 0
        for i, block in enumerate(blocks):
            try:
                # Entry 객체 생성
                entry = Entry(
                    content=block['content'],
                    metadata=block['metadata']
                )
                
                # Qdrant에 저장
                await qdrant.store(entry, collection_name=collection_name)
                stored_count += 1
                
                # 진행상황 표시
                if (i + 1) % 10 == 0:
                    print(f"   📝 {i + 1}/{len(blocks)} 블럭 저장 완료...")
                    
            except Exception as e:
                print(f"⚠️  블럭 저장 실패 (ID: {block['metadata'].get('block_id')}): {e}")
                continue
        
        print(f"✅ 저장 완료!")
        print(f"   성공: {stored_count}/{len(blocks)} 블럭")
        print(f"   컬렉션: {collection_name}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 저장 중 오류 발생: {e}")
        return False

async def main():
    """메인 실행 함수"""
    print("🔄 소설 블럭 저장 시작...")
    
    # 커맨드라인 인자 처리
    preprocessed_file = sys.argv[1] if len(sys.argv) > 1 else "block_data_preprocessed.json"
    
    success = await store_novel_blocks(preprocessed_file)
    
    if success:
        print("\n🎯 다음 단계:")
        print("   python scripts/search_blocks.py \"현재 스토리 내용\"")
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
