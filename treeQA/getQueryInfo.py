
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from entitylinking.ELModels import llmForEntityExtract,llmForEntityFilter, linkEntity
from treeQA.wikipediaUtills import getWikipediaResultByNV, getWikipediaResultDirect
from treeQA_Config import Chroma_store, RetrieveModelName, article_top_k, RLTop_k
from treeQA.wikidataUtills import relationLinking, get_wikipedia_title_from_qid, getWikidataEntity, HEADERS

import requests



def safe_request(url, params, my_proxies, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url,headers=HEADERS,params=params,proxies=my_proxies)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as e:
            if attempt < (max_retries - 1):  # i.e. if it's not the last retry
                print(f"Request failed: {e}. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2 ** attempt)
            else:
                raise
    return None



def fetch_wikipedia_text(QID, query, top_k):
    label = get_wikipedia_title_from_qid(QID)
    if Chroma_store and RetrieveModelName != "lexical":
        return getWikipediaResultByNV(label, query,top_k=top_k)
    return getWikipediaResultDirect(label, query,top_k=top_k)


def getQueryInfo(query, myInfoBox, logicTree, top_k=RLTop_k, path=None):

    with ThreadPoolExecutor() as executor:
        # Step 1: Parallel entity extraction and linking
        entity_extract_future = executor.submit(llmForEntityExtract, query,logicTree)

        entity_linking_future = executor.submit(linkEntity, query)

        entity_linking_result = entity_linking_future.result()

        json_data = json.loads(entity_linking_result)


        entities = entity_extract_future.result()


        # Step 2: Parallel fetching of Wikidata entities and relation generalization

        entity_results_future = executor.submit(getWikidataEntity, entities)
        entity_results = entity_results_future.result()
        #relaQuery = relation_generalization_future.result()

        # Merge entity linking results
        for entityLinkingItem in json_data:


            first_sentence_end = entityLinkingItem['definition'].find('.')


            first_sentence = entityLinkingItem['definition'][
                             :first_sentence_end + 1] if first_sentence_end != -1 else entityLinkingItem['definition']
            entity_results[entityLinkingItem['wikidata']] = {
                'text': entityLinkingItem['text'],
                'wikidata': entityLinkingItem['wikidata'],
                'definition': first_sentence
            }
        # Step 3: Filter entities
        entityIDs = llmForEntityFilter(entity_results, query,logicTree)
        # Save filtered results
        retrieve_QID = []
        itemInfo_filtered = []
        for entityID in entityIDs:
            if entityID:
                retrieve_QID.append(entityID)
                itemInfo_filtered.append(entity_results[entityID])


        # Step 4: Parallel retrieval of graph content and Wikipedia text

        #print(entityIDs, relaQuery, entity_results, query, top_k, myInfoBox)
        retrieve_relation_future = executor.submit(relationLinking, entityIDs, query,entity_results,myInfoBox, top_k,logicTree)

        # Parallel fetching of Wikipedia texts
        futures = [executor.submit(fetch_wikipedia_text, QID, query, top_k=article_top_k) for QID in retrieve_QID]

        for future in as_completed(futures):
            text = future.result()
            myInfoBox.addText(text)

        retrieve_relation_List = retrieve_relation_future.result()
    trace_text = []
    for group in myInfoBox.textInfo:
        entries = group if isinstance(group, list) else [group]
        for entry in entries:
            if isinstance(entry, dict):
                trace_text.append({
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "content": entry.get("content"),
                    "similarity": entry.get("similarity"),
                })
    logicTree.record_trace_event({
        "type": "retrieval",
        "path": list(path) if path is not None else None,
        "query": query,
        "extracted_entities": entities,
        "wikidata_candidates": list(entity_results.values())[:20],
        "selected_entity_ids": retrieve_QID,
        "graph_facts": myInfoBox.graphInfo,
        "wikipedia_passages": trace_text[:20],
    })
    return retrieve_QID, list(retrieve_relation_List)
