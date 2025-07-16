from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

wrapper = DuckDuckGoSearchAPIWrapper(
    # region="de-de",
    # time="d",
    # backend="lite",
    max_results=30,
    safesearch="off",
    source="text",
    region="us-US",
)

tool = DuckDuckGoSearchResults(
    api_wrapper=wrapper,
    response_format="content_and_artifact",
    output_format="list",
    max_results=30,
    region="us-US",
)
tool.max_results = 30

o: list = tool.invoke("SB1047")

# result: o is a list of such dicts:
# {'link': 'https://www.rfi.fr/fr/technologies/20240826-régulation-de-l-ia-sb-1047-ce-projet-de-loi-qui-secoue-le-secteur-en-californie', 'snippet': 'Aug 26, 2024 · Régulation de l’IA: «SB-1047», ce projet de loi ' 'qui secoue le secteur en Californie Aux États-Unis, l’État de ' 'Californie, berceau mondial du développement de la tech, pourrait ' 'voter …', 'title': 'Régulation de l’IA: «SB-1047», ce projet de loi qui secoue ... - ' 'RFI'}

print(o)
print(len(o))

breakpoint()
