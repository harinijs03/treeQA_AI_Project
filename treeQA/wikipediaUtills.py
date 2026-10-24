import re
import wikipediaapi
from treeQA_Config import TREEQA_WIKIPEDIA_CHUNK_WORDS


wiki_wiki = wikipediaapi.Wikipedia(user_agent="TreeQA/1.0 (research demo)", language="en")
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb

        from treeQA_Config import PersistentClient_Path, chroma_collection_name

        client = chromadb.PersistentClient(path=PersistentClient_Path)
        _collection = client.get_or_create_collection(
            name=chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def get_article_sections(title):




    page = wiki_wiki.page(title)
    if not page.exists():
        print(f"Article '{title}' does not exist.")
        return []

    sections = []

    main_content = page.summary.strip()
    if main_content:
        sections.append({
            "title": "Introduction",
            "content": main_content
        })

    def extract_sections(section, prefix='', sections=None):
        section_title = f"{prefix}{section.title}"
        if "References" in section_title or "External links" in section_title:
            return

        content = section.text.strip()
        if content:
            sections.append({
                "title": section_title,
                "content": content
            })

        for subsection in section.sections:
            extract_sections(subsection, prefix=section_title + " > ", sections=sections)

    for section in page.sections:
        extract_sections(section, sections=sections)

    return sections


def split_text_by_tokens(section, max_tokens=TREEQA_WIKIPEDIA_CHUNK_WORDS):
    """
    Split a section into sentence-preserving word-limited chunks.
    """
    content = section['content']
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            continue
        sentence_length = len(sentence.split())

        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    return chunks


def is_exists(article_title):



    try:
        result = _get_collection().get(ids=[f"{article_title}_0_0"])
        return bool(result['ids'])
    except Exception as e:
        print(f"Error checking existence: {e}")
        return False


def embed_and_store(sections, article_title):
    from embedding.embeddingModel import getEmbeddings

    collection = _get_collection()
    all_documents = []
    all_metadatas = []
    all_ids = []

    for idx, section in enumerate(sections):
        split_texts = split_text_by_tokens(section)
        for part_idx, part_text in enumerate(split_texts):
            doc_id = f"{article_title}_{idx}_{part_idx}"
            all_documents.append(part_text)
            all_metadatas.append({
                "article_title": article_title,
                "title": section['title'],
                "content": part_text
            })
            all_ids.append(doc_id)

    batch_size = 100
    total_batches = (len(all_documents) + batch_size - 1) // batch_size

    # Now, insert all data into the collection in batches
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_documents))
        batch_documents = all_documents[start:end]
        batch_metadatas = all_metadatas[start:end]
        batch_ids = all_ids[start:end]
        batch_embeddings = getEmbeddings(batch_documents)
        if batch_embeddings is None or len(batch_embeddings) != len(batch_documents):
            raise RuntimeError(
                f"Expected {len(batch_documents)} embeddings for Wikipedia article "
                f"{article_title!r}; received {0 if batch_embeddings is None else len(batch_embeddings)}."
            )
        collection.add(
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"{article_title} batch {batch_idx + 1} of {total_batches} successfully embedded and stored.")


def query_article(query, top_k,article_name):



    from embedding.embeddingModel import getEmbeddings

    embeddings = getEmbeddings([query])
    collection = _get_collection()
    article_filter = {"article_title": {"$eq": article_name}}
    article_chunk_ids = collection.get(where=article_filter, include=[])['ids']
    article_chunk_count = len(article_chunk_ids)
    if not article_chunk_count:
        return []
    results = collection.query(
        query_embeddings=embeddings,
        n_results=min(top_k, article_chunk_count),
        include=['metadatas', 'distances' ],
        where=article_filter
    )
    query_results = []
    if 'distances' in results:
        for result_id, metadata, distance in zip(
            results['ids'][0],
            results['metadatas'][0],
            results['distances'][0],
        ):
            query_results.append({
                "id": result_id,
                "title": metadata['title'],
                "content": metadata['content'],
                "similarity": 1.0 - float(distance),
            })
    return query_results


def getWikipediaResultByNV(article_name, query , top_k=3):



    if article_name:
        print(article_name)
        if is_exists(article_name):
            print("Article already exists in the database.")
        else:
            sections = get_article_sections(article_name)
            if sections:
                print(f"Extracted {len(sections)} sections from '{article_name}'.")
                embed_and_store(sections, article_name)

        results = query_article(query, top_k, article_name)
        return results
    return None



def embed_and_query_direct(article_name, query, top_k):
    if article_name:
        print(f"Find article {article_name}.")
    # 1. Fetch article
    sections = get_article_sections(article_name)
    if not sections:
        print(f"Article '{article_name}' not found or has no content.")
        return []

    # 2. Split into chunks
    all_chunks_data = []
    #print(f"Splitting article '{article_name}' into chunks...")
    for idx, section in enumerate(sections):
        split_texts = split_text_by_tokens(section)
        for part_idx, part_text in enumerate(split_texts):
            all_chunks_data.append({
                "article_title": article_name,
                "title": section['title'],
                "content": part_text,
                # Optional: Add an ID if needed later, though not strictly necessary for direct query
                "id": f"{article_name}_{idx}_{part_idx}"
            })
    #print(f"Generated {len(all_chunk_texts)} chunks.")

    if not all_chunks_data:
        print("No text chunks generated.")
        return []

    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 2 and term not in {
            "about", "after", "among", "from", "have", "that", "their",
            "there", "these", "this", "what", "when", "where", "which",
            "while", "with", "would",
        }
    }
    if not query_terms:
        return []

    def score(text):
        terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        overlap = len(query_terms & terms)
        return overlap / ((len(query_terms) * len(terms)) ** 0.5) if overlap else 0.0

    ranked_chunks = sorted(
        (
            (score(chunk["content"]), index)
            for index, chunk in enumerate(all_chunks_data)
        ),
        reverse=True,
    )
    top_k_indices = [index for rank, index in ranked_chunks[:top_k] if rank > 0]

    # 6. Format results
    #print(f"Found {len(top_k_indices)} relevant chunks.")
    results = []
    for i in top_k_indices:
        results.append({
            "id": all_chunks_data[i]["id"],
            "title": all_chunks_data[i]["title"],
            "content": all_chunks_data[i]["content"],
            "similarity": score(all_chunks_data[i]["content"])
        })

    return results

# New main function for direct query
def getWikipediaResultDirect(article_name, query, top_k=3):
    """
    Fetches and lexically ranks a Wikipedia article without external embeddings.
    """
    if not article_name:
        print("Article name is required.")
        return None

    # Use the globally defined embedding_function


    results = embed_and_query_direct(article_name, query, top_k) # Pass the function/object
    return results


if __name__ == "__main__":
    print(getWikipediaResultByNV("West Germanic languages", "Which countries speak West Germanic languages?",3))

    print(getWikipediaResultDirect("West Germanic languages", "Which countries speak West Germanic languages?",3))