from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


def similarity(a, b):

    emb = model.encode([a, b])

    score = cosine_similarity([emb[0]], [emb[1]])[0][0]

    return score