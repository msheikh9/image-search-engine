#  Image Search Engine using CLIP


This project implements an **AI-powered image search engine** that retrieves similar images based on either a **text description** or an **image query**.  
It uses the **CLIP (ViT-B/32)** model from OpenAI through the Sentence-Transformers library.




##  Objectives
- Develop a multimodal AI system that connects **vision and language**.
- Use pretrained models (CLIP) to perform **semantic image retrieval**.
- Demonstrate real-time image search through a **Streamlit web app**.


##  How It Works
1. **Dataset Preparation:**  
   - The dataset (101 Object Categories from Kaggle) is stored under `data/images/`.
2. **Index Building:**  
   - `build_index.py` uses CLIP to encode all dataset images into 512-D embeddings.  
   - These embeddings and image paths are saved in the `index/` folder.
3. **Search Engine:**  
   - `search_core.py` loads the embeddings and uses **cosine similarity** to find nearest matches.  
   - Queries (text or image) are encoded using CLIP in the same embedding space.
4. **User Interface:**  
   - `app.py` provides a **Streamlit** interface to search via text or image upload.

