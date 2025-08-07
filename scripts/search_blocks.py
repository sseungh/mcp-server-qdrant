#!/usr/bin/env python3
"""
소설 블럭 검색 스크립트

현재 작성 중인 스토리와 의미적으로 유사한 블럭들을 검색합니다.
Qdrant 벡터 데이터베이스에서 코사인 유사도를 사용하여 검색합니다.
"""

import sys
import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
load_dotenv()

# MCP 서버 모듈 import
sys.path.append(str(Path(__file__).parent.parent / "src"))
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.settings import EmbeddingProviderSettings

async def search_related_blocks(current_story, limit=10):
    """
    현재 스토리와 관련된 블럭들을 검색
    
    Args:
        current_story: 현재 작성 중인 스토리 텍스트
        limit: 반환할 최대 블럭 수
        
    Returns:
        관련 블럭들의 리스트
    """
    
    try:
        # 환경변수에서 설정 로드
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("COLLECTION_NAME", "novel_blocks")
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        search_limit = int(os.getenv("QDRANT_SEARCH_LIMIT", str(limit)))
        
        print(f"🔍 검색 설정:")
        print(f"   Qdrant URL: {qdrant_url}")
        print(f"   컬렉션: {collection_name}")
        print(f"   임베딩 모델: {embedding_model}")
        print(f"   검색 한도: {search_limit}")
        print(f"   검색어: \"{current_story[:50]}{'...' if len(current_story) > 50 else ''}\"")
        print()
        
        # 임베딩 제공자 생성 (기존 레포 방식 - 환경변수 자동 로드)
        embedding_settings = EmbeddingProviderSettings()
        embedding_provider = create_embedding_provider(embedding_settings)
        
        # 점수 정보를 위해 직접 Qdrant 클라이언트 사용
        from qdrant_client import AsyncQdrantClient
        
        qdrant_client = AsyncQdrantClient(url=qdrant_url)
        
        print("🚀 관련 블럭 검색 중...")
        
        # 검색어를 벡터로 변환
        query_vector = await embedding_provider.embed_query(current_story)
        vector_name = embedding_provider.get_vector_name()
        
        # Qdrant에서 검색 (점수 포함)
        search_results = await qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=search_limit,
        )
        
        # 결과를 Entry 형태로 변환 (점수 정보 포함)
        results = []
        for result in search_results.points:
            results.append({
                'content': result.payload["document"],
                'metadata': result.payload.get("metadata"),
                'score': result.score  # 점수 정보 추가
            })
        
        if not results:
            print("❌ 검색 결과가 없습니다.")
            print("💡 먼저 블럭을 저장했는지 확인하세요: python scripts/store_blocks.py")
            return []
        
        print(f"✅ {len(results)}개의 관련 블럭을 찾았습니다:")
        print("=" * 80)
        
        # 검색 결과 표시 (점수 순으로 정렬됨)
        for i, entry in enumerate(results, 1):
            print(f"\n📝 블럭 #{i} (유사도: {entry['score']:.4f})")
            print(f"ID: {entry['metadata'].get('block_id', 'N/A')}")
            print(f"단어수: {entry['metadata'].get('word_count', 'N/A')}")
            print(f"생성일: {entry['metadata'].get('created_at', 'N/A')}")
            print("내용:")
            
            # 블럭 내용 미리보기 (너무 길면 잘라서 표시)
            content = entry['content']
            if len(content) > 200:
                content = content[:200] + "..."
            
            print(f"  {content}")
            
            if i < len(results):
                print("-" * 40)
        
        return results
        
    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        return []

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("❌ 사용법: python scripts/search_blocks.py \"현재 스토리 내용\"")
        print()
        print("예시:")
        print("  python scripts/search_blocks.py \"주인공이 새로운 마법을 배운다\"")
        print("  python scripts/search_blocks.py \"위험한 상황에서의 선택\"")
        sys.exit(1)
    
    current_story = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("🔄 소설 블럭 검색 시작...")
    print()
    
    # 비동기 검색 실행
    results = asyncio.run(search_related_blocks(current_story, limit))
    
    if results:
        print("\n🎯 사용법:")
        print("   이 블럭들을 LLM의 컨텍스트로 사용하여 새로운 블럭을 생성하세요.")
    else:
        print("\n💡 도움말:")
        print("   1. 먼저 블럭이 저장되었는지 확인: python scripts/store_blocks.py")
        print("   2. Qdrant가 실행 중인지 확인: docker ps")
        print("   3. .env 파일의 설정을 확인하세요")

if __name__ == "__main__":
    main()
