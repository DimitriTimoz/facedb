# FaceDB — ThisPersonDoesNotExist database with vector search (Meilisearch)

A tiny Rust ingestor that continuously fetches faces from thispersondoesnotexist.com, embeds them with an ArcFace ONNX model, and indexes them into Meilisearch with user-provided vectors for fast nearest-neighbor search.

It’s a simple playground to explore image embeddings and vector search, fully self-hosted.

## What it does

- Pulls one random synthetic face ~every 1 second from thispersondoesnotexist.com
- Preprocesses to 112×112 RGB and runs ArcFace (ONNX Runtime) to get a 512‑dim embedding
- Stores each face as a document in Meilisearch with two vector fields: `embedding` and `default` (both 512‑dim)

## Searching

You can use Meilisearch’s vector search API to find the nearest neighbors given a 512‑dim query vector. With multiple vector fields configured, pass the field name:

```zsh
# Replace … with a 512‑float array you computed using the same model
curl \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MEILI_MASTER_KEY" \
  -X POST http://localhost:7700/indexes/faces/search \
  -d '{
    "vector": { "embedding": [ … 512 floats … ] },
    "limit": 10
  }'
```

Tip: you can also use `"vector": { "default": [...] }` — both fields contain the same embedding in this project.


## How it works (under the hood)

- The model is downloaded from Hugging Face if `model.onnx` is missing:
  - `https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx`
- ONNX Runtime is initialized with optimization level 3 and 6 intra threads
- Each image is fetched via a pooled `reqwest` client, decoded with `image`, preprocessed, run through the model, and the resulting embedding is inserted twice (as `embedding` and `default`) into the `_vectors` field so either name can be used in queries
- Primary key is `id` (random `u32`)

## Notes

- The faces are synthetic and generated on‑the‑fly by thispersondoesnotexist.com
- This project is for experimentation and demos; no persistence besides Meilisearch’s data directory (in the Docker volume)

