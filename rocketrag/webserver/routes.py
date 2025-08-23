from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import json
import os

from .models import QuestionRequest, QuestionResponse
from ..visualization import VectorVisualizer


def setup_routes(app, rocket_rag):
    """Setup all API routes for the FastAPI app"""

    # Use the RocketRAG instance components
    db = rocket_rag.db
    vectorizer = rocket_rag.vectorizer

    # Setup Jinja2 templates
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    templates = Jinja2Templates(directory=templates_dir)

    @app.get("/", response_class=HTMLResponse)
    async def index_page(request: Request):
        return templates.TemplateResponse(
            request=request, name="index.html", context={}
        )

    @app.post("/ask", response_model=QuestionResponse)
    async def ask_question(request: QuestionRequest):
        try:
            answer, sources = rocket_rag.ask(request.question)
            # Convert SearchResult objects to SourceInfo format
            serializable_sources = [
                {
                    "text": source.chunk,
                    "filename": source.filename,
                    "score": source.score,
                }
                for source in sources
            ]
            return QuestionResponse(answer=answer, sources=serializable_sources)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ask/stream")
    async def ask_question_stream(request: QuestionRequest):
        try:

            def generate_stream():
                stream, sources = rocket_rag.stream_ask(request.question)
                # Convert SearchResult objects to dictionaries for JSON serialization
                serializable_sources = [
                    {
                        "text": source.chunk,
                        "filename": source.filename,
                        "score": source.score,
                    }
                    for source in sources
                ]
                for output in stream:
                    delta = output["choices"][0]["delta"]
                    if "content" in delta:
                        yield f"data: {json.dumps({'content': delta['content'], 'sources': serializable_sources})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': serializable_sources})}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/visualize", response_class=HTMLResponse)
    async def visualize_page(request: Request):
        return templates.TemplateResponse(
            request=request, name="visualize.html", context={}
        )

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page(request: Request):
        return templates.TemplateResponse(request=request, name="chat.html", context={})

    @app.get("/browse", response_class=HTMLResponse)
    async def browse_documents_page(request: Request):
        filenames = db.get_unique_filenames()
        return templates.TemplateResponse(
            request=request, name="browse.html", context={"filenames": filenames}
        )

    @app.get("/browse/document/{filename}")
    async def get_document_chunks(filename: str, page: int = 1, per_page: int = 10):
        try:
            all_chunks = db.get_vectors_by_filename(filename)
            total_chunks = len(all_chunks)
            total_pages = (total_chunks + per_page - 1) // per_page

            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            chunks = all_chunks[start_idx:end_idx]

            return {
                "filename": filename,
                "chunks": chunks,
                "pagination": {
                    "current_page": page,
                    "per_page": per_page,
                    "total_chunks": total_chunks,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/visualize/data")
    async def visualize_data(
        limit: int = 200,
        question: Optional[str] = None,
        filename_filter: Optional[str] = None,
    ):
        try:
            records = (
                db.get_vectors_by_filename(filename_filter)
                if filename_filter
                else db.get_vectors_with_metadata(limit=limit)
            )

            if not records:
                return {"error": "No vectors found"}

            vectors = [record["vector"] for record in records]
            metadata = [
                {"id": r["id"], "text": r["text"], "filename": r["filename"]}
                for r in records
            ]

            visualizer = VectorVisualizer()
            analysis_result = visualizer.analyze_vectors(
                vectors, metadata, color_by="filename"
            )

            if "error" in analysis_result:
                raise HTTPException(status_code=500, detail=analysis_result["error"])

            points = [
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "text": meta["text"],
                    "filename": meta["filename"],
                    "id": meta["id"],
                    "is_question": False,
                }
                for point, meta in zip(analysis_result["reduced_vectors"], metadata)
            ]

            response_data = {"points": points, "question_text": None}

            if question:
                question_result = visualizer.process_question(
                    question, vectors, metadata, analysis_result, vectorizer
                )

                if "error" not in question_result:
                    question_point = question_result.get("question_point")
                    if question_point is not None:
                        points.append(
                            {
                                "x": float(question_point[0]),
                                "y": float(question_point[1]),
                                "text": f"Question: {question}",
                                "filename": "Question",
                                "id": "question",
                                "is_question": True,
                            }
                        )
                        response_data["question_text"] = question

                    for similarity, idx, _ in question_result.get("similar_chunks", []):
                        if idx < len(points) - 1:
                            points[idx]["similarity"] = similarity

            return response_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
