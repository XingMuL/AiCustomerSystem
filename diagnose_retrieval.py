import sys
from backend.pipeline.rag_pipeline import RAGPipeline

print('=' * 60)
print('【诊断 1】查询: 小米扫地机器人')
print('=' * 60)
rag = RAGPipeline()
result = rag.retrieve('小米扫地机器人')
print(f'改写查询: {result.get("rewritten_query", "")}')
print(f'子块候选数: {len(result.get("child_hits", []))}')
print(f'父块数: {len(result.get("parent_chunks", []))}')
print()
print('【子块候选 Top5】')
for i, c in enumerate(result.get("child_hits", [])[:5]):
    content = c.content.replace('\n', ' ')[:200]
    print(f'  [{i+1}] score={c.score:.4f} chapter={c.chapter_title[:30]}')
    print(f'      {content}')
print()
print('【父块内容 Top3】')
for i, p in enumerate(result.get("parent_chunks", [])[:3]):
    content = (p.get('content') or '').replace('\n', ' ')[:200]
    print(f'  [{i+1}] chapter={p.get("chapter_title", "")[:30]} len={len(p.get("content", ""))}')
    print(f'      {content}')
print()
print('=' * 60)
print('【诊断 2】查询: 商品 008')
print('=' * 60)
result2 = rag.retrieve('商品 008')
print(f'改写查询: {result2.get("rewritten_query", "")}')
print(f'子块候选数: {len(result2.get("child_hits", []))}')
for i, c in enumerate(result2.get("child_hits", [])[:5]):
    content = c.content.replace('\n', ' ')[:200]
    print(f'  [{i+1}] score={c.score:.4f} chapter={c.chapter_title[:30]}')
    print(f'      {content}')
print()
print('=' * 60)
print('【诊断 3】查看 Qdrant 子块集合中包含 扫地机器人 的文档')
print('=' * 60)
from backend.retrieval.vector_store import VectorStore
vs = VectorStore()
try:
    print(f'Qdrant 中总子块数: {vs.client.count(vs.COLLECTION_CHILD).count}')
    # 找包含 扫地/008 的子块
    found_saodi = []
    scroll_result2, _ = vs.client.scroll(
        collection_name=vs.COLLECTION_CHILD,
        limit=2000,
        with_payload=True,
        with_vectors=False,
    )
    for pt in scroll_result2:
        payload = pt.payload or {}
        content = payload.get('content_snippet', payload.get('content', ''))
        if '扫地' in content or '008' in content:
            found_saodi.append({'chapter_title': payload.get('chapter_title', ''), 'content': content[:150], 'doc_id': payload.get('doc_id', '')})
    print(f'找到包含 扫地/008 的子块数: {len(found_saodi)}')
    for item in found_saodi[:5]:
        print(f'  doc={item["doc_id"][:16]} chapter={item["chapter_title"][:30]}')
        c = item["content"].replace('\n', ' ')
        print(f'      {c}')
except Exception as e:
    print(f'Qdrant 查询异常: {e}')
print()
print('=' * 60)
print('【诊断 4】查看知识库 Markdown 文件中商品 008 周围的章节')
print('=' * 60)
import os
md_files = sorted(os.listdir('backend/markdown'))
structured_files = [f for f in md_files if 'structured' in f]
print(f'找到 {len(structured_files)} 个结构化文件')
if structured_files:
    latest = structured_files[-1]
    print(f'读取最新文件: {latest}')
    with open(f'backend/markdown/{latest}', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 查找 商品 008 的位置
    for i, line in enumerate(lines):
        if '商品 008' in line:
            print(f'  第 {i} 行: {line.strip()[:80]}')
            # 查看前后章节标题
            start = max(0, i - 3)
            end = min(len(lines), i + 15)
            print(f'  上下文 {start}-{end} 行:')
            for j in range(start, end):
                print(f'    {j}: {lines[j].rstrip()[:80]}')
            break
print()
print('=' * 60)
print('【诊断 5】查询: CAT-HOME-008 (商品编号)')
print('=' * 60)
result3 = rag.retrieve('CAT-HOME-008')
print(f'子块候选数: {len(result3.get("child_hits", []))}')
for i, c in enumerate(result3.get("child_hits", [])[:3]):
    content = c.content.replace('\n', ' ')[:200]
    print(f'  [{i+1}] score={c.score:.4f} chapter={c.chapter_title[:30]}')
    print(f'      {content}')
print()
print('=' * 60)
print('【诊断 6】查看章节切分逻辑分析')
print('=' * 60)
from backend.chunking.parent_child_splitter import ChapterSplitter, ParentChildSplitter
from backend.chunking.toc_extractor import Chapter, TOCExtractor
if structured_files:
    latest = structured_files[-1]
    with open(f'backend/markdown/{latest}', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 手动测试章节切分
    splitter = ChapterSplitter()
    # 构造一个简单的 Document 对象
    class SimpleDoc:
        def __init__(self, text):
            self.raw_text = text
            self.toc = None

    chapters = splitter.split(SimpleDoc(raw_text))
    print(f'总章节数: {len(chapters)}')
    # 查找包含 008 的章节
    for i, ch in enumerate(chapters):
        title_str = (ch.title or '').strip()
        if '008' in title_str or '扫地' in ch.content[:200]:
            print(f'  Chapter [{i}] title={title_str[:50]} content_len={len(ch.content)}')
            preview = ch.content[:200].replace('\n', ' ')
            print(f'      前200字符: {preview}')
    print()
    # 父子块切分
    pc_splitter = ParentChildSplitter()
    child_chunks, parent_chunks = pc_splitter.split_chapters(chapters)
    print(f'父子块切分: {len(child_chunks)} 子块, {len(parent_chunks)} 父块')
    # 查找包含 008 的子块
    for i, cc in enumerate(child_chunks):
        if '008' in cc.content or '扫地' in cc.content:
            print(f'  Child [{i}] chapter={cc.chapter_title[:30]} parent={cc.parent_id[:16]}')
            cc_content = cc.content[:150].replace('\n', ' ')
            print(f'      内容: {cc_content}')
