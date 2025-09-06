# test script for embeddings
from src.embed.embedder import Embedder
import numpy as np

def main():
    e = Embedder()
    vecs = e.embed_texts(["hello world", "hello world", "goodbye"])
    assert vecs.shape[1] == e.dim              # expect 384
    assert vecs.shape[0] == 3
    # cosine similarity check
    sim_same = float(np.dot(vecs[0], vecs[1])) # normalized -> cosine
    sim_diff = float(np.dot(vecs[0], vecs[2]))
    assert sim_same > 0.95 and sim_diff < 0.9
    print("OK embedder:", e.dim, vecs.shape)

if __name__ == "__main__":
    main()
