# Pipeline RAG de ContextVault

## 1. Resumen

ContextVault es un backend FastAPI para una aplicacion RAG. Permite registrar usuarios, crear workspaces, subir documentos, indexarlos y hacer preguntas sobre su contenido.

Un RAG, o Retrieval-Augmented Generation, resuelve un problema comun de los LLMs: el modelo no conoce automaticamente los documentos privados de una aplicacion. En lugar de pedirle al LLM que responda solo con su conocimiento interno, el backend primero busca fragmentos relevantes en los documentos del usuario y luego le entrega ese contexto al modelo para generar una respuesta fundamentada.

En palabras simples, el flujo completo es:

- El usuario sube documentos a un workspace.
- El backend guarda el archivo y su metadata.
- El usuario pide indexar el documento.
- El backend extrae texto, lo divide en chunks y genera embeddings.
- Los chunks se guardan en PostgreSQL con pgvector.
- Cuando el usuario pregunta algo, el backend recupera chunks relevantes.
- Gemini genera una respuesta usando solo ese contexto recuperado.
- El backend devuelve la respuesta, sus fuentes y metadata de retrieval.

## 2. Flujo general

```text
Usuario sube documento
-> backend guarda metadata y archivo
-> usuario indexa documento
-> extractor obtiene texto
-> splitter divide en chunks
-> embedding provider genera vectores
-> chunks se guardan en PostgreSQL + pgvector
-> usuario pregunta
-> retrieval busca chunks relevantes
-> LLM genera respuesta usando contexto
-> backend devuelve answer + sources + metadata
```

Este flujo separa dos momentos importantes: primero preparar los documentos para busqueda y luego usarlos como contexto para responder preguntas.

## 3. Upload de documentos

El upload de documentos pertenece al modulo `documents`. Cuando un usuario sube un archivo, el backend guarda dos cosas:

- Metadata en base de datos: workspace, nombre original, ruta de storage, content type, estado y timestamps.
- Archivo en storage local: el contenido real del documento queda guardado bajo `LOCAL_STORAGE_PATH`.

Los estados actuales del documento son:

- `uploaded`: el archivo fue subido, pero todavia no fue indexado.
- `indexing`: el documento esta en proceso de indexacion.
- `indexed`: la indexacion termino correctamente y existen chunks persistidos.
- `index_failed`: ocurrio un error durante la indexacion.

El upload no genera embeddings inmediatamente. Solo registra y guarda el documento para que pueda indexarse despues.

## 4. Indexacion

La indexacion convierte un documento subido en chunks buscables. En ContextVault esto vive en `IndexDocumentService`.

El proceso actual es:

1. Validar que el documento pertenezca al usuario y workspace correctos.
2. Cambiar el estado del documento a `indexing`.
3. Leer el archivo desde storage local.
4. Extraer texto segun el tipo de archivo.
5. Dividir el texto en chunks.
6. Generar embeddings para cada chunk.
7. Reemplazar los chunks anteriores del documento, si existian.
8. Guardar los nuevos chunks en `document_chunks`.
9. Marcar el documento como `indexed`.

Los extractores actuales soportan:

- `.txt`
- `.md`
- `.pdf`

El splitter actual usa `RecursiveCharacterTextSplitter` de LangChain, configurado con `RAG_CHUNK_SIZE` y `RAG_CHUNK_OVERLAP`.

Separar upload e indexacion tiene ventajas practicas:

- El upload puede ser rapido y simple.
- La indexacion puede fallar sin perder el documento original.
- En el futuro seria posible mover la indexacion a workers/background jobs.

## 5. Embeddings y pgvector

Un embedding es una representacion numerica de un texto. En vez de comparar palabras exactas, el sistema convierte una frase o chunk en un vector de numeros que representa su significado aproximado.

Ejemplo simple:

- Pregunta: `como configuro la clave secreta del JWT?`
- Chunk: `JWT_SECRET_KEY configures signing secrets`

Aunque no usan exactamente las mismas palabras, sus embeddings pueden quedar cerca en el espacio vectorial porque hablan de conceptos relacionados.

ContextVault usa Gemini para generar embeddings y PostgreSQL con pgvector para guardarlos y buscarlos. pgvector permite ordenar chunks por cercania vectorial, lo que habilita busqueda semantica dentro del workspace del usuario.

La tabla `document_chunks` guarda, entre otros campos:

- `workspace_id`
- `document_id`
- `content`
- `chunk_index`
- `embedding`
- `metadata`

## 6. Busqueda vectorial

La busqueda vectorial se activa con:

```json
{
  "retrieval_mode": "vector"
}
```

En este modo, el backend:

1. Genera un embedding para la pregunta.
2. Busca chunks cercanos usando pgvector.
3. Devuelve los chunks semanticamente mas parecidos.

Sirve muy bien cuando el usuario pregunta con palabras distintas a las del documento, pero con la misma intencion.

Fortalezas:

- Encuentra similitud semantica.
- Tolera sinonimos y formulaciones distintas.
- Es util para preguntas naturales.

Debilidades:

- Puede no priorizar terminos tecnicos exactos.
- Identificadores como nombres de variables, tablas o clases pueden quedar subrepresentados.
- Depende de la calidad del embedding.

## 7. Busqueda keyword/full-text

La busqueda keyword se activa con:

```json
{
  "retrieval_mode": "keyword"
}
```

En este modo, el backend no genera embedding para la pregunta. Busca coincidencias lexicales en el contenido de los chunks.

En PostgreSQL, la busqueda usa full-text search sobre `document_chunks.content` con configuracion `simple`. Esta configuracion es adecuada para terminos tecnicos porque evita que reglas de stemming o stopwords de un idioma transformen demasiado los identificadores.

Sirve especialmente para buscar terminos exactos como:

- `pgvector`
- `JWT_SECRET_KEY`
- `RAG_MIN_RELEVANCE_SCORE`
- `workspace_id`
- `document_chunks`

Fortalezas:

- Encuentra identificadores tecnicos exactos.
- No requiere llamada al proveedor de embeddings.
- Es facil de depurar cuando se busca una palabra concreta.

Debilidades:

- No entiende sinonimos ni reformulaciones tan bien como vector search.
- Si el usuario pregunta de forma muy distinta al texto original, puede no encontrar buenos resultados.

## 8. RAG hibrido

El modo hibrido se activa con:

```json
{
  "retrieval_mode": "hybrid"
}
```

Tambien puede ser el modo por defecto con `RAG_RETRIEVAL_MODE=hybrid`.

En este modo, ContextVault combina dos estrategias:

- Busqueda vectorial: buena para significado e intencion.
- Busqueda keyword: buena para terminos exactos.

El resultado es mas robusto. Por ejemplo, una pregunta puede tener intencion semantica general y al mismo tiempo incluir un identificador tecnico. El modo hibrido permite que ambos caminos aporten candidatos.

El problema que soluciona es que ningun metodo por separado es perfecto:

- Solo vector puede perder exactitud tecnica.
- Solo keyword puede perder significado semantico.
- Hybrid intenta recuperar lo mejor de ambos.

## 9. Weighted Reciprocal Rank Fusion

El RAG hibrido necesita combinar resultados de dos busquedas con scores diferentes. El score de pgvector y el score de full-text search no significan exactamente lo mismo, asi que no conviene sumarlos directamente como si fueran comparables.

ContextVault usa weighted Reciprocal Rank Fusion, o weighted RRF. La idea simple es usar posiciones de ranking:

- Si un chunk aparece muy arriba en la busqueda vectorial, gana puntos.
- Si aparece muy arriba en la busqueda keyword, gana puntos.
- Si aparece bien rankeado en ambas, sube mas.
- Si aparece solo en una busqueda, igual puede competir.

La formula conceptual es:

```text
score = vector_weight  * 1 / (RRF_K + vector_rank)
      + keyword_weight * 1 / (RRF_K + keyword_rank)
```

Variables configurables actualmente por settings/env:

- `RAG_RETRIEVAL_MODE`: modo por defecto (`vector`, `keyword` o `hybrid`).
- `RAG_VECTOR_WEIGHT`: peso de la busqueda vectorial en hybrid.
- `RAG_KEYWORD_WEIGHT`: peso de la busqueda keyword en hybrid.
- `RAG_VECTOR_CANDIDATES`: cantidad de candidatos vectoriales antes de fusionar.
- `RAG_KEYWORD_CANDIDATES`: cantidad de candidatos keyword antes de fusionar.
- `RAG_RERANKING_ENABLED`: activa o desactiva reranking por defecto.
- `RAG_RERANKING_PROVIDER`: proveedor de reranking; en esta fase solo existe `noop`.
- `RAG_RERANKING_CANDIDATES`: cantidad maxima de candidatos que se pasan al reranker.

Parametro interno actual:

- `RRF_K = 60`: constante interna usada para suavizar el impacto del ranking.

Actualmente `RRF_K` no esta expuesto como variable de entorno. No existe `RAG_RRF_K` en esta fase.

## 10. Reranking opcional

Reranking es un paso opcional posterior al retrieval inicial. Toma los candidatos recuperados por vector, keyword o hybrid y puede reordenarlos antes de seleccionar los chunks finales.

El flujo con reranking habilitado queda asi:

```text
pregunta
-> retrieval obtiene candidatos iniciales
-> reranker reordena candidatos
-> se seleccionan los mejores chunks
-> se construye contexto
-> LLM genera respuesta
```

En esta fase, el backend agrega la arquitectura de reranking sin depender de un proveedor externo real. El proveedor por defecto es `noop`, que conserva el orden actual. Esto mantiene el pipeline testeable y permite activar/desactivar el paso sin llamadas adicionales a Gemini.

Reranking puede controlarse por settings o por request:

```json
{
  "query": "JWT_SECRET_KEY",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "reranking_enabled": true
}
```

Si `reranking_enabled` no viene en el request, el backend usa `RAG_RERANKING_ENABLED`.

La metadata de retrieval incluye diagnosticos como:

- `reranking_enabled`
- `reranking_provider`
- `reranking_applied`
- `reranking_candidates`
- `candidates_before_rerank`

Los resultados pueden incluir campos opcionales:

- `rerank_score`
- `original_rank`
- `reranked_rank`

El campo `score` conserva el score de retrieval o fusion. No se reemplaza por `rerank_score`, porque ambos pueden tener escalas distintas.

## 11. Query answering

ContextVault tiene dos endpoints RAG principales:

- `/rag/search`
- `/rag/query`

`/rag/search` es un endpoint de debug/retrieval. Sirve para ver que chunks recupera el sistema, con que modo, scores y metadata. No genera una respuesta con LLM.

`/rag/query` es el endpoint de pregunta-respuesta. Hace retrieval igual que search, selecciona chunks como contexto, construye un prompt con `RagPromptBuilder` y llama al LLM configurado. Actualmente el LLM provider usa Gemini.

El flujo de `/rag/query` es:

1. Validar pregunta y workspace.
2. Recuperar chunks con `RetrievalService`.
3. Si el modo es `vector`, aplicar `RAG_MIN_RELEVANCE_SCORE` para filtrar contexto.
4. Si el modo es `keyword` o `hybrid`, usar los resultados recuperados sin aplicar ese threshold vectorial.
5. Si no hay contexto suficiente, devolver una respuesta controlada sin llamar al LLM.
6. Construir prompt con contexto recuperado.
7. Llamar a Gemini mediante `LLMProviderPort`.
8. Devolver `answer`, `sources` y `metadata`.

## 12. Sources y metadata

Las fuentes permiten explicar de donde salio la respuesta. Esto es clave para un sistema RAG porque ayuda a revisar si el modelo respondio usando documentos reales.

Campos importantes:

- `score`: score final usado para ordenar el resultado. En hybrid es el score fusionado normalizado.
- `vector_score`: score original de la busqueda vectorial, si el chunk vino por ese camino.
- `keyword_score`: score original de la busqueda keyword, si el chunk vino por ese camino.
- `rerank_score`: score opcional asignado por el reranker cuando aplica.
- `original_rank`: posicion opcional antes de reranking.
- `reranked_rank`: posicion opcional despues de reranking.
- `retrieval_source`: indica si el chunk vino de `vector`, `keyword` o de ambos (`hybrid`).
- `fusion_algorithm`: indica el algoritmo usado para fusionar resultados; actualmente `weighted_rrf` cuando aplica.

En `/rag/search`, los resultados incluyen el contenido completo del chunk para debug. En `/rag/query`, las sources incluyen un `content_preview` para mostrar evidencia sin devolver todo el contexto completo.

La metadata tambien incluye datos utiles como:

- modo de retrieval usado
- cantidad de candidatos vectoriales y keyword
- cantidad de resultados por estrategia
- cantidad de chunks deduplicados
- cantidad final de resultados
- modelo LLM usado en query answering
- cantidad de chunks usados como contexto

## 13. Arquitectura del modulo RAG

El modulo RAG sigue una estructura de capas:

- `domain`: entidades como `DocumentChunk`, `SimilarChunk`, `RetrievalMode` y `RetrievalSource`.
- `application`: servicios, DTOs y puertos.
- `infrastructure`: adapters concretos para SQLAlchemy, Gemini, LangChain, extractores y repositorios.
- `interfaces`: routers FastAPI y schemas HTTP.

Piezas principales:

- `IndexDocumentService`: orquesta la indexacion de un documento.
- `RetrievalService`: punto central para vector, keyword e hybrid retrieval.
- `SearchSimilarChunksService`: caso de uso del endpoint de debug `/rag/search`.
- `QueryRagService`: caso de uso para responder preguntas con contexto recuperado.
- `RagPromptBuilder`: construye prompts para el LLM a partir de pregunta y chunks.
- Ports: abstracciones como `EmbeddingProviderPort`, `LLMProviderPort`, `RerankerPort`, `ChunkRepositoryPort`, `TextExtractorPort` y `TextSplitterPort`.
- Adapters: implementaciones concretas, por ejemplo Gemini embeddings, Gemini LLM y LangChain splitter.
- Rerankers: implementaciones detras de `RerankerPort`; en esta fase el default es `noop`.
- Repositories: persistencia y busqueda de chunks en SQLAlchemy/PostgreSQL.

La regla importante es que la logica RAG no vive en los routers. Los routers componen dependencias, validan HTTP, llaman servicios y convierten DTOs a schemas de respuesta.

## 14. Que no hace todavia

El pipeline actual todavia no incluye:

- Un proveedor real externo de reranking.
- Parent-child chunks.
- Historial conversacional.
- Streaming de respuestas.
- Observabilidad avanzada.

Estas capacidades estan fuera del alcance actual y no deben asumirse como implementadas.

## 15. Proxima fase: provider real de reranking

La siguiente fase natural despues de la arquitectura de reranking es agregar un adapter real.

Ese adapter podria usar un modelo dedicado de reranking o un LLM con salida estructurada. Debe integrarse detras de `RerankerPort` y no llamarse directamente desde routers ni desde `QueryRagService`.

El objetivo futuro seria reemplazar el comportamiento `noop` por una estrategia mas precisa:

```text
pregunta
-> hybrid retrieval obtiene candidatos
-> provider real de reranking reordena candidatos
-> se construye contexto con los mejores
-> LLM genera respuesta
```

El objetivo seria mejorar la calidad del contexto final. No cambia la idea de RAG; mejora que fragmentos llegan al prompt.

Segun `AGENTS.md`, los nuevos providers deben agregarse como port/adapter y deben integrarse alrededor de `RetrievalService`. No deben llamarse directamente desde routers ni mezclarse con detalles de proveedor en la capa de interfaces.

## 16. Comandos utiles

Validar linting:

```bash
ruff check .
```

Ejecutar tests:

```bash
pytest
```

Ejecutar migraciones:

```bash
alembic upgrade head
```

Ejemplo basico para `/rag/search`:

```bash
curl -X POST "http://localhost:8000/api/v1/workspaces/$WORKSPACE_ID/rag/search" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "JWT_SECRET_KEY",
    "top_k": 5,
    "retrieval_mode": "hybrid",
    "reranking_enabled": true
  }'
```

Ejemplo basico para `/rag/query`:

```bash
curl -X POST "http://localhost:8000/api/v1/workspaces/$WORKSPACE_ID/rag/query" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Para que sirve JWT_SECRET_KEY?",
    "top_k": 5,
    "retrieval_mode": "hybrid",
    "reranking_enabled": true
  }'
```

Variables usadas en los ejemplos:

- `ACCESS_TOKEN`: token JWT obtenido con `/api/v1/auth/login`.
- `WORKSPACE_ID`: identificador del workspace donde estan los documentos.
