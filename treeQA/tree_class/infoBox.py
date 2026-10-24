class infoBox:
    def __init__(self):

        self.graphInfo = []
        self.textInfo = []
        self.sparqlInfo = []

        self.seen_graphs = set()
        self.seen_texts = set()

    def addGraph(self, json_array):
        for items in json_array:

            jsonItems = json_array[items]
            for item in jsonItems:
                key = (item.get('head', ''), item.get('relation', ''), item.get('tail'))
                if key not in self.seen_graphs:
                    self.graphInfo.append([item])
                    self.seen_graphs.add(key)

    def addText(self, new_info):

        if new_info:
            for json_array in new_info:
                key = json_array.get('id', '')
                if key and key not in self.seen_texts:
                    self.textInfo.append([json_array])
                    self.seen_texts.add(key)
