import asyncio
from src.core.knowledge_flow import KnowledgeFlow

async def main():
    query = "large language models"
    kf = KnowledgeFlow()
    print(f"Testing arXiv search for: {query} (top 10)")
    res = await kf.search_manager.search(query, platform="arxiv", max_results=10, sort_by="relevance")
    arxiv_res = res.get("arxiv", {})
    count = arxiv_res.get("result_count", 0)
    print(f"result_count: {count}")
    items = arxiv_res.get("results", [])
    for i, it in enumerate(items[:10], 1):
        title = it.get("title", "").strip()
        published = it.get("published", "")
        print(f"{i}. {title} | {published}")

if __name__ == "__main__":
    asyncio.run(main()) 