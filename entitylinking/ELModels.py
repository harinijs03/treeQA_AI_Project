
import re
from LLMs.models import getModelResponse
from treeQA_Config import ELTop_k, el_model
import aiohttp
import asyncio
from aiohttp import ClientTimeout
from treeQA_Config import Azure_key, Azure_endpoint, proxies,relik_server_url
import json
import requests


async def get_wikidata_id_with_proxy(wikipedia_url, retries=3, delay=5):











    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(
                    timeout=ClientTimeout(total=30),
                    connector=aiohttp.TCPConnector(ssl=False)
            ) as session:
                async with session.get(
                        f"https://www.wikidata.org/w/api.php",
                        params={
                            "action": "wbgetentities",
                            "sites": "enwiki",
                            "titles": wikipedia_url.split("/")[-1],
                            "format": "json",
                        },
                        proxy=proxies.get('http') if proxies else None
                ) as response:
                    data = await response.json()
                    entities = data.get("entities", {})
                    return next(iter(entities.keys()), None)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {wikipedia_url}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    return None



async def azureEntityLinking(text):
    """
    Performs Azure Entity Linking and fetches Wikidata IDs, returning results as a JSON string.
    """
    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        client = TextAnalyticsClient(
            endpoint=Azure_endpoint,
            credential=AzureKeyCredential(Azure_key),
        )
        documents = [text]

        results = client.recognize_linked_entities(documents=documents)
        result = None

        for doc_result in results:
            if not doc_result.is_error:
                result = doc_result
                break
            else:

                print(f"Error processing document: {doc_result.id}, Error: {doc_result.error}")

                error_info = {"error": f"Document level error: {doc_result.error.code}", "message": doc_result.error.message}
                return json.dumps(error_info, ensure_ascii=False)

        if result is None:

            return json.dumps({"error": "No document processed successfully", "entities": []}, ensure_ascii=False)


        entities_data = []
        tasks = []

        for entity in result.entities:

            entity_info = {
                "text": entity.name,
                "Wikipedia URL": entity.url,
                "definition": entity.name,
                "wikidata": None
            }
            entities_data.append(entity_info)

            tasks.append(get_wikidata_id_with_proxy(entity.url))



        wikidata_ids = await asyncio.gather(*tasks, return_exceptions=True)


        for i, wikidata_id_or_exception in enumerate(wikidata_ids):
            if isinstance(wikidata_id_or_exception, Exception):

                print(f"Error fetching Wikidata ID for {entities_data[i]['Wikipedia URL']}: {wikidata_id_or_exception}")
                entities_data[i]["wikidata"] = f"Error: {wikidata_id_or_exception}"
            else:
                entities_data[i]["wikidata"] = wikidata_id_or_exception




        return json.dumps(entities_data, indent=4, ensure_ascii=False)

    except Exception as e:

        print(f"An error occurred in azureEntityLinking: {e}")

        error_info = {"error": str(e), "entities": []}
        return json.dumps(error_info, ensure_ascii=False)

    except Exception as err:
        raise RuntimeError(f"Error extracting entities: {err}")



def relikEntityLinking(query):

    url = relik_server_url
    params = {
        'text': query,
        'is_split_into_words': 'false',
        'retriever_batch_size': '32',
        'reader_batch_size': '32',
        'return_windows': 'false',
        'use_doc_topic': 'false',
        'annotation_type': 'char',
        'relation_threshold': '0.5'
    }


    headers = {
        'accept': 'application/json'
    }


    response = requests.get(url, params=params, headers=headers)


    #print(response.json())


    candidates_info=[]
    for item in response.json():

        if 'candidates' in item and isinstance(item['candidates'], dict) and 'span' in item['candidates']:

            for candidate in item['candidates']['span'][0][0]:

                text = candidate.get('text', 'N/A')
                wikidata = candidate.get('metadata', {}).get('wikidata', 'N/A')
                definition = candidate.get('metadata', {}).get('definition', 'N/A')


                candidates_info.append({
                    'text': text,
                    'wikidata': wikidata,
                    'definition': definition
                })


    return json.dumps(candidates_info, ensure_ascii=False, indent=4)







def llmForEntityExtract(query,logicTree):
    prompt = """ I need you to help me understand the user's request and identify the entities involved in the user's query. Your task is simple: recognize the user's request and generate the parameters accordingly. Please note that when extracting, only the noun form is required! No plural forms or other forms of nouns, etc.
    Here's an example:
    Input: Who composed the music for Manru? The music for Manru was composed by Ignacy Jan Paderewski.
    Output: "Manru", "Ignacy Jan Paderewski"
    Input:Find information about Kirill Eskov's biography or personal details to determine his country of citizenship.
    Output:"Kirill Eskov"
    Now the user input is: """

    item_string,tokenCount = getModelResponse(prompt, query)
    logicTree.tokenCount +=tokenCount
    print("Entity extract complete!")
    # currentCount = currentCount + tokenCount


    pattern = r'"(.*?)"'

    matches = re.findall(pattern, item_string)
    return matches



def llmForEntityFilter(itemInfo,query,logicTree):
    itemSelectPrompt = f"""Now I need you to select the entity that is truly relevant to the query.  
    I need you to select the most relevant entity ID based on the following information, 
    Output the related IDs no more than {ELTop_k} without outputting any other content.
    format: ['','','']
    entity info: {itemInfo}
    The user's query is:{query}
    """
    top1_item_string,tokenCount = getModelResponse(itemSelectPrompt, "Please begin to choose.")
    logicTree.tokenCount+=tokenCount
    # currentCount = currentCount + tokenCount


    pattern = r"'(.*?)'"

    matches = re.findall(pattern, top1_item_string)
    print(f"Selected {len(matches)} entities!")
    return matches




def linkEntity(query, model_name=el_model):
    if model_name == "wikidata":
        return json.dumps([])
    if model_name =="relik":
        return relikEntityLinking(query)
    if model_name == "azure":
        return asyncio.run(azureEntityLinking(query))
    raise ValueError(f"Invalid entity-linking model: {model_name}")



if __name__ == "__main__":
    input_text = """To Catch a Predator was devoted to impersonating people below the age of consent for which in North America varies by what?"""
    print(linkEntity(input_text))
    print(relikEntityLinking(input_text))