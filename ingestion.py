"""
Fetches papers from arXiv

"""

import arxiv

def fetch_papers(
    category: str = 'cs.AI',
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ):
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    papers = client.results(search)
    papers_list = list(papers)

    return papers_list

if __name__ == "__main__":
    papers = fetch_papers()

    for paper in papers:
        print(paper.get_short_id(), paper.title)