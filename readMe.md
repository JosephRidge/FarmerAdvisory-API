# Architecture:

![alt text](architecture.png)

---

Running:  
`fastapi dev main.py`

## **Endpoints:**

**path**: `{BASE_URL}/fetch-data`

**description**: fetch, chunk , create vectorDB using Chroma

**Output**:

```json
    {
    "documents": {
        "time_taken": 13.44028902053833,
        "chunk": 30048,
        "chroma_vector_DB_status": "created",
        "data": [
        {
            "acceptedDate": "",
            "arxivId": null,
            "authors": [
            {
                "name": "Kipkirui, Edwin"
            },
            {
                "name": "Seif, Salum Kamota"
            }
            ],
            "citationCount": 0,
            "contributors": [],
            "outputs": [
            "https://api.core.ac.uk/v3/outputs/603903781"
            ],
            "createdDate": "2024-05-11T23:27:42",
            "dataProviders": [
            {
                "id": 22377,
                "name": "",
                "url": "https://api.core.ac.uk/v3/data-providers/22377",
                "logo": "https://api.core.ac.uk/data-providers/22377/logo"
            }
            ],
            "depositedDate": "",
            "abstract": "Methane emissions from livestock being a major contributor to climate change as methane possesses more global warming potential than carbon dioxide, exacerbating the issue. Therefore....",
            "documentType": "",
            "doi": "10.59324/ejtas.2024.2(2).44",
            "downloadUrl": "https://core.ac.uk/download/603903781.pdf",
            "fieldOfStudy": null,
            "fullText": "  This work is licensed under a Creative Commons Attribution 4.0 International License. The license permits unrestricted use, distribution, and reproduction in any medium, on the condition.....",
            "id": 156076121,
            "identifiers": [
            {
                "identifier": "oai:ejtas.com:article/794",
                "type": "OAI_ID"
            },
            {
                "identifier": "10.59324/ejtas.2024.2(2).44",
                "type": "DOI"
            },
            {
                "identifier": "603903781",
                "type": "CORE_ID"
            }
            ],
            "title": "Harnessing Tanzania's Rangelands to Mitigate Methane Emissions from Livestock Enteric Fermentation",
            "language": {
            "code": "en",
            "name": "English"
            },
            "magId": null,
            "oaiIds": [
            "oai:ejtas.com:article/794"
            ],
            "publishedDate": "2024-03-01T00:00:00",
            "publisher": "AMO Publisher",
            "pubmedId": null,
            "references": [],
            "sourceFulltextUrls": [
            "https://ejtas.com/index.php/journal/article/download/794/585"
            ],
            "updatedDate": "2024-05-11T23:27:42",
            "yearPublished": 2024,
            "journals": [],
            "links": [
            {
                "type": "download",
                "url": "https://core.ac.uk/download/603903781.pdf"
            },
            {
                "type": "reader",
                "url": "https://core.ac.uk/reader/603903781"
            },
            {
                "type": "thumbnail_m",
                "url": "https://core.ac.uk/image/603903781/large"
            },
            {
                "type": "thumbnail_l",
                "url": "https://core.ac.uk/image/603903781/large"
            },
            {
                "type": "display",
                "url": "https://core.ac.uk/works/156076121"
            }
            ]
        },
        {
            "acceptedDate": "",
            "arxivId": null,
            "authors": [
                .................]
        }]


    }}
```


**path**: `{BASE_URL}/ws`


**description**: pose query and attain the response via the chain

**Output**:

```json
{
    "query": "What are livestock emisisons?",
    "answer": "Livestock emissions refer to the release of greenhouse gases and other pollutants from animal farming, particularly from ruminant livestock such as cows, sheep, and goats.\n\n#### 🌍 Response:\n\nLivestock emissions are a significant contributor to global greenhouse gas emissions. The most obvious and relevant difference between industrial production and livestock farming is that animals are vertebrates with eyes, voice, senses, and perception, with brains and emotions, whereas humans working in livestock farming feel an emotional relation to the animals.\n\n#### 📚 References:\n\n- Author(s). (Year). \"Livestock Methane Intensity: A Review.\" Journal of Animal Science, vol. 100(5), pp. e141-e153.\n- [16] Andrey byzaakai and Khorlai Langaa, Tes-khem, Tyva Republic\n- Author(s). (Year). \"The Importance of Regular Cleaning of Barns and Campsite Territory for Livestock Dung Management.\" Journal of Rural Development, vol. 36(3), pp. 345-357.\n- [5] Author(s). (Year). \"The Effects of Seasonal Pastures on Sheep Growth and Productivity.\" Journal of Agricultural Science, vol. 155(2), pp. 135-145.\n- [55] Smith et al. (2019). \"Direct and Indirect Effects of Parasitism on Livestock GhG Emissions.\" Journal of Applied Animal Welfare Science, vol. 22(3), pp. 245-255.\n\n#### 📚 References:\n\n[1]\n\n- Author(s). (Year). *The Tes-khem Province: A Sourcebook for Rural Development*. Tes-khem Province, Tyva Republic.\n \n#### 🌍 Response:\n\n Livestock emissions are a significant contributor to global greenhouse gas emissions. The most obvious and relevant difference between industrial production and livestock farming is that animals are vertebrates with eyes, voice, senses, and perception, with brains and emotions, whereas humans working in livestock farming feel an emotional relation to the animals.\n\n Livestock emissions are primarily driven by methane (mainly emitted by ruminants) and ammonia. Methane emission factors were calculated by normalising measured emis-sion rates to the body weight base unit, whereby one livestock unit corresponded to 500 kg of body weight. The animal weights used and other information on the farm’s management were obtained by interviewing the farmers.\n\n#### 📚 References:\n\n[1]\n\n- Author(s). (Year). *The Tes-khem Province: A Sourcebook for Rural Development*. Tes-khem Province, Tyva Republic.",
    "    ": [
        {
            "title": "can livestock farming benefit from industry 4.0 technology? evidence from recent study",
            "authors": "Bernhardt, Heinz, Brunsch, Reiner, Büscher, Wolfgang, Colangelo, Eduardo, Graf, Henri, Kraft, Martin, Marquering, Johannes, Tapken, Heiko, Toppel, Kathrin, Westerkamp, Clemens, Ziron, Martin",
            "publishedDate": "2022-01-01T00:00:00",
            "yearPublished": 2022,
            "doi": "https://doi.org/10.34657/10402",
            "publisher": "Basel : MDPI",
            "fieldOfStudy": "None",
            "links": "{'type': 'download', 'url': 'https://core.ac.uk/download/555511036.pdf'}, {'type': 'reader', 'url': 'https://core.ac.uk/reader/555511036'}, {'type': 'thumbnail_m', 'url': 'https://core.ac.uk/image/555511036/large'}, {'type': 'thumbnail_l', 'url': 'https://core.ac.uk/image/555511036/large'}, {'type': 'display', 'url': 'https://core.ac.uk/works/168328014'}"
        },
        {
            "title": "livestock dung use in steppe pastoralism : renewable resources, care, and respect for sentient nonhumans",
            "authors": "Peemot, Victoria Soyan",
            "publishedDate": "2022-03-01T00:00:00",
            "yearPublished": 2022,
            "doi": "https://doi.org/10.3167/sib.2022.210102",
            "publisher": "",
            "fieldOfStudy": "None",
            "links": "{'type': 'download', 'url': 'https://core.ac.uk/download/534019635.pdf'}, {'type': 'reader', 'url': 'https://core.ac.uk/reader/534019635'}, {'type': 'thumbnail_m', 'url': 'https://core.ac.uk/image/534019635/large'}, {'type': 'thumbnail_l', 'url': 'https://core.ac.uk/image/534019635/large'}, {'type': 'display', 'url': 'https://core.ac.uk/works/128405933'}"
        }
    ],
    "response_time": "69.22 sec",
    "chat_history": [
        [
            "What are livestock emisisons?",
            "Livestock emissions refer to the release of greenhouse gases and other pollutants from animal farming, particularly from ruminant livestock such as cows, sheep, and goats.\n\n#### 🌍 Response:\n\nLivestock emissions are a significant contributor to global greenhouse gas emissions. The most obvious and relevant difference between industrial production and livestock farming is that animals are vertebrates with eyes, voice, senses, and perception, with brains and emotions, whereas humans working in livestock farming feel an emotional relation to the animals.\n\n#### 📚 References:\n\n- Author(s). (Year). \"Livestock Methane Intensity: A Review.\" Journal of Animal Science, vol. 100(5), pp. e141-e153.\n- [16] Andrey byzaakai and Khorlai Langaa, Tes-khem, Tyva Republic\n- Author(s). (Year). \"The Importance of Regular Cleaning of Barns and Campsite Territory for Livestock Dung Management.\" Journal of Rural Development, vol. 36(3), pp. 345-357.\n- [5] Author(s). (Year). \"The Effects of Seasonal Pastures on Sheep Growth and Productivity.\" Journal of Agricultural Science, vol. 155(2), pp. 135-145.\n- [55] Smith et al. (2019). \"Direct and Indirect Effects of Parasitism on Livestock GhG Emissions.\" Journal of Applied Animal Welfare Science, vol. 22(3), pp. 245-255.\n\n#### 📚 References:\n\n[1]\n\n- Author(s). (Year). *The Tes-khem Province: A Sourcebook for Rural Development*. Tes-khem Province, Tyva Republic.\n \n#### 🌍 Response:\n\n Livestock emissions are a significant contributor to global greenhouse gas emissions. The most obvious and relevant difference between industrial production and livestock farming is that animals are vertebrates with eyes, voice, senses, and perception, with brains and emotions, whereas humans working in livestock farming feel an emotional relation to the animals.\n\n Livestock emissions are primarily driven by methane (mainly emitted by ruminants) and ammonia. Methane emission factors were calculated by normalising measured emis-sion rates to the body weight base unit, whereby one livestock unit corresponded to 500 kg of body weight. The animal weights used and other information on the farm’s management were obtained by interviewing the farmers.\n\n#### 📚 References:\n\n[1]\n\n- Author(s). (Year). *The Tes-khem Province: A Sourcebook for Rural Development*. Tes-khem Province, Tyva Republic."
        ]
    ]
}
```

kindly note: The query takes on average 20-25 seconds

## **Insights:**

![alt text](data-ingestion.png)

- Data Ingestion Module:
  Features:
- Fetch data from CORE
- Chunk the data
- Create Embeddings

Time taken:

- Whole procedure: 23 minutes
- Articles: 264 (took 4.82 secs to fetch)
- Chunks: 30048 (took 3.81 secs to chunk - Recursively)
