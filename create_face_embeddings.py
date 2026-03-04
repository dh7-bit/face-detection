import os
import cv2
import numpy as np
from crop import crop
from align import align_face
from embedding import get_embedding
import faiss
import pickle
personaldetails = []
embeddingslist2 = []
ids = 1
for item in os.listdir('rawimages'):
    pathing = os.path.join('rawimages', item)
    img=cv2.imread(pathing)
    crops = crop(img)
    if crops is None:
        print(f"Skipping {pathing}, image not loaded")
        continue
    aligns = align_face(crops)
    embeddings = get_embedding(aligns)
    embeddingslist2.append(embeddings.reshape(1, -1)) 
    personaldetails.append({"id": ids, "name": item.split('.')[0]})
    ids += 1
embeddings = np.vstack(embeddingslist2).astype("float32")
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
dimension = 128
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("Number of vectors in index:", index.ntotal)
faiss.write_index(index, "faiss_index.index")
with open("personaldetails.pkl", "wb") as f:
    pickle.dump(personaldetails, f)

  