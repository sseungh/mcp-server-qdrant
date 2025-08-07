#!/usr/bin/env python3
"""
소설 블럭 전처리 스크립트

block_data.json에서 LLMOutput만 추출하여 
block_data_preprocessed.json 파일을 생성합니다.
"""

import json
import sys
from pathlib import Path

def extract_llm_outputs(input_file="block_data.json", output_file="block_data_preprocessed.json"):
    """
    원본 JSON에서 LLMOutput만 추출하여 새로운 JSON 파일 생성
    
    Args:
        input_file: 원본 block_data.json 파일 경로
        output_file: 생성될 전처리된 JSON 파일 경로
    """
    
    # 입력 파일 존재 확인
    if not Path(input_file).exists():
        print(f"❌ 오류: {input_file} 파일이 존재하지 않습니다.")
        return False
    
    try:
        # 원본 데이터 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # JSON 구조 확인 및 블럭 배열 추출
        if 'data' in json_data and 'data' in json_data['data']:
            data = json_data['data']['data']  # API 응답 구조: data.data[]
        elif isinstance(json_data, list):
            data = json_data  # 이미 배열인 경우
        else:
            print(f"❌ 오류: 지원하지 않는 JSON 구조입니다.")
            return False
        
        print(f"📖 원본 데이터 로드: {len(data)}개 항목")
        
        # LLMOutput만 추출하여 새로운 구조 생성
        preprocessed_blocks = []
        valid_blocks = 0
        
        for item in data:
            if 'LLMOutput' in item and item['LLMOutput'].strip():
                preprocessed_blocks.append({
                    'content': item['LLMOutput'].strip(),  # 실제 소설 블럭 텍스트
                    'metadata': {
                        'block_id': item.get('_id'),
                        'word_count': item.get('metadata', {}).get('wordCount'),
                        'created_at': item.get('createdAt')
                    }
                })
                valid_blocks += 1
        
        # 전처리된 데이터를 새 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(preprocessed_blocks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 전처리 완료!")
        print(f"   유효한 블럭: {valid_blocks}개")
        print(f"   저장 파일: {output_file}")
        
        # 샘플 블럭 미리보기
        if preprocessed_blocks:
            sample = preprocessed_blocks[0]['content']
            preview = sample[:100] + "..." if len(sample) > 100 else sample
            print(f"   샘플 블럭: {preview}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("❌ 사용법: python scripts/preprocess_blocks.py <입력파일> [출력파일]")
        print()
        print("예시:")
        print("  python scripts/preprocess_blocks.py block_data.json")
        print("  python scripts/preprocess_blocks.py block_data.json my_preprocessed.json")
        sys.exit(1)
    
    print("🔄 소설 블럭 전처리 시작...")
    
    # 커맨드라인 인자 처리 - 입력파일은 필수
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "block_data_preprocessed.json"
    
    success = extract_llm_outputs(input_file, output_file)
    
    if success:
        print("\n🎯 다음 단계:")
        print(f"   python scripts/store_blocks.py {output_file}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
