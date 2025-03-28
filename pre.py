from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba
import re
import numpy as np
import torch

with open("modern_mnrs.txt", "r", encoding="utf-8", errors="ignore") as file_:
    file_data = []
    labels = []
    for line in file_.readlines():
        fields = re.split(r'\s+', line.strip())
        if len(fields) >= 2:
            label, text = fields[:2]
            labels.append(label)
            text_seg = jieba.lcut(text)
            file_data.append(TaggedDocument(text_seg, [label]))

corpus = [' '.join(doc.words) for doc in file_data]
vectorizer = TfidfVectorizer(max_features=300, min_df=0.001, max_df=0.5)
tfidf_matrix = vectorizer.fit_transform(corpus)
tfidf_values = tfidf_matrix.toarray()
print(tfidf_matrix.shape[1])

model = Doc2Vec(vector_size=300, window=5, min_count=1, epochs=20, dm=0)
tagged_documents = [TaggedDocument(doc.words, [i]) for i, doc in enumerate(file_data)]
model.build_vocab(tagged_documents)
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

combined_vectors = []
for i, doc in enumerate(file_data):
    doc_vector = model.infer_vector(doc.words)
    combined_vector = np.concatenate([doc_vector, tfidf_values[i]])
    combined_vectors.append(combined_vector)

X = torch.tensor(np.array(combined_vectors), dtype=torch.float)

label_to_id = {label: idx for idx, label in enumerate(set(labels))}

y = torch.tensor([label_to_id[label] for label in labels], dtype=torch.long)

np.savetxt("tfidf_doc2vec_features.csv", X.numpy(), delimiter=",")
assert X.shape[1] == 600, f"Combined vector dimension is not 200, actual dimension is {X.shape[1]}"

np.savetxt("labels.csv", y.numpy(), delimiter=",")
