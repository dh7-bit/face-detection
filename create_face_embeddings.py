import numpy as np
from crop import crop
from align import align_face
from embedding import get_embedding
from loader import load_images
import faiss
import pickle
def generateembeddings(images):
    personal_details = []
    embeddings_list = []
    person_id= 1
    for file_name,img in images:
        crops = crop(img)
        aligns = align_face(crops)
        embedding = get_embedding(aligns)
        embeddings_list.append(embedding.reshape(1, -1)) 
        personal_details.append({"id": person_id, "name": file_name.split('.')[0]})
        person_id += 1
    return embeddings_list,personal_details    
def built_faiss_index(embeddings_list):
    embeddings=np.vstack(embeddings_list).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = 128
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("Number of vectors in index:", index.ntotal)
    faiss.write_index(index, "faiss_index.index")

def generating_pickle_file(personal_details):
    with open("personal_details.pkl", "wb") as f:
        pickle.dump(personal_details, f)
def main():
    images=load_images()
    embeddings_list,personal_details=generateembeddings(images)
    built_faiss_index(embeddings_list)
    generating_pickle_file(personal_details)
    print(embeddings_list,personal_details)
if __name__=="__main__":
    main()    
  