import numpy as np
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class PCAReducer:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, : self.n_components]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (
            eigenvalues[: self.n_components] / total_variance
        )

        return X_centered @ self.components_


class KMeansClusterer:
    def __init__(
        self, n_clusters: int = 3, max_iters: int = 100, random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)

            new_centroids = np.array(
                [X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)]
            )

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def fit_predict(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        if n_clusters is not None:
            self.n_clusters = n_clusters
        self.fit(X)
        return self.labels_


class ASCIIPlotter:
    def __init__(self, width: int = 80, height: int = 24):
        self.width = width
        self.height = height
        self.symbols = ["●", "○", "◆", "◇", "▲", "△", "■", "□", "★", "☆"]
        self.colors = ["red", "blue", "green", "yellow", "magenta", "cyan", "white"]

    def plot(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        filenames: Optional[List[str]] = None,
        color_by: str = "cluster",
        question_point=None,
    ) -> str:
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D for plotting")

        all_points = points
        if question_point is not None:
            all_points = np.vstack([points, question_point.reshape(1, -1)])

        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1

        x_scaled = ((points[:, 0] - x_min) / (x_max - x_min) * (self.width - 1)).astype(
            int
        )
        y_scaled = (
            (points[:, 1] - y_min) / (y_max - y_min) * (self.height - 1)
        ).astype(int)

        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        if color_by == "filename" and filenames:
            unique_filenames = list(set(filenames))
            filename_to_idx = {
                filename: i for i, filename in enumerate(unique_filenames)
            }

            for i, (x, y) in enumerate(zip(x_scaled, y_scaled)):
                if 0 <= x < self.width and 0 <= y < self.height:
                    filename = filenames[i] if i < len(filenames) else "unknown"
                    symbol_idx = filename_to_idx.get(filename, 0) % len(self.symbols)
                    grid[self.height - 1 - y][x] = self.symbols[symbol_idx]
        else:
            for i, (x, y) in enumerate(zip(x_scaled, y_scaled)):
                if 0 <= x < self.width and 0 <= y < self.height:
                    symbol_idx = 0 if labels is None else labels[i] % len(self.symbols)
                    grid[self.height - 1 - y][x] = self.symbols[symbol_idx]

        if question_point is not None:
            q_x = int((question_point[0] - x_min) / (x_max - x_min) * (self.width - 1))
            q_y = int((question_point[1] - y_min) / (y_max - y_min) * (self.height - 1))
            if 0 <= q_x < self.width and 0 <= q_y < self.height:
                grid[self.height - 1 - q_y][q_x] = "✦"

        plot_str = "\n".join("".join(row) for row in grid)
        return plot_str

    def create_rich_plot(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        filenames: Optional[List[str]] = None,
        title: str = "Vector Visualization",
        color_by: str = "cluster",
        question_point=None,
        question_text=None,
    ) -> Panel:
        plot_str = self.plot(points, labels, filenames, color_by, question_point)

        legend_text = ""
        if color_by == "filename" and filenames:
            unique_filenames = list(set(filenames))
            filename_to_idx = {
                filename: i for i, filename in enumerate(unique_filenames)
            }
            legend_lines = []
            for filename in unique_filenames:
                symbol_idx = filename_to_idx[filename] % len(self.symbols)
                symbol = self.symbols[symbol_idx]
                count = filenames.count(filename)
                display_name = (
                    filename if len(filename) <= 30 else filename[:27] + "..."
                )
                legend_lines.append(f"{symbol} {display_name} ({count} points)")
            legend_text = "\n" + "\n".join(legend_lines)
        elif labels is not None:
            unique_labels = np.unique(labels)
            legend_lines = []
            for i, label in enumerate(unique_labels):
                symbol = self.symbols[i % len(self.symbols)]
                count = np.sum(labels == label)
                legend_lines.append(f"{symbol} Cluster {label} ({count} points)")
            legend_text = "\n" + "\n".join(legend_lines)

        if question_point is not None:
            legend_text += "\n✦ Question" + (
                f": {question_text[:30]}..."
                if question_text and len(question_text) > 30
                else f": {question_text}"
                if question_text
                else ""
            )

        return Panel(
            plot_str + legend_text, title=title, border_style="blue", padding=(1, 2)
        )


class VectorVisualizer:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.pca = PCAReducer(n_components=2)
        self.kmeans = KMeansClusterer()
        self.plotter = ASCIIPlotter()

    def process_question(
        self,
        question_text: str,
        vectors: List[List[float]],
        metadata: List[Dict],
        analysis_result: Dict,
        vectorizer,
    ) -> Dict:
        try:
            question_vector = vectorizer.vectorize(question_text)

            question_array = np.array(question_vector).reshape(1, -1)
            question_centered = question_array - self.pca.mean_
            question_reduced = question_centered @ self.pca.components_

            similarities = []
            for i, vector in enumerate(vectors):
                dot_product = np.dot(question_vector, vector)
                norm_q = np.linalg.norm(question_vector)
                norm_v = np.linalg.norm(vector)
                similarity = dot_product / (norm_q * norm_v)
                similarities.append((similarity, i, metadata[i]))

            similarities.sort(key=lambda x: x[0], reverse=True)
            top_5 = similarities[:5]

            return {"question_point": question_reduced[0], "similar_chunks": top_5}

        except Exception as e:
            return {"error": f"Failed to process question: {str(e)}"}

    def analyze_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict],
        n_clusters: int = 5,
        color_by: str = "cluster",
    ) -> Dict:
        if not vectors:
            return {"error": "No vectors provided"}

        try:
            vector_array = np.array(vectors)
            reduced_vectors = self.pca.fit_transform(vector_array)

            labels = None
            if color_by == "cluster":
                labels = self.kmeans.fit_predict(reduced_vectors, n_clusters)

            filenames = [meta.get("filename", "unknown") for meta in metadata]

            return {
                "reduced_vectors": reduced_vectors,
                "labels": labels,
                "filenames": filenames,
                "metadata": metadata,
            }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def create_visualization_panel(
        self,
        analysis_result: Dict,
        title: str = "Vector Space Visualization",
        color_by: str = "cluster",
        question_point=None,
        question_text=None,
    ) -> Panel:
        if "error" in analysis_result:
            return Panel(
                f"Error: {analysis_result['error']}",
                title="Visualization Error",
                border_style="red",
            )

        reduced_vectors = analysis_result["reduced_vectors"]
        labels = analysis_result.get("labels")
        filenames = analysis_result.get("filenames")

        return self.plotter.create_rich_plot(
            reduced_vectors,
            labels,
            filenames,
            title,
            color_by,
            question_point,
            question_text,
        )

    def create_similar_chunks_panel(
        self, similar_chunks: List[tuple], question: str
    ) -> Panel:
        if not similar_chunks:
            return Panel(
                "No similar chunks found", title="Similar Chunks", border_style="yellow"
            )

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Similarity", justify="right", width=10)
        table.add_column("File", style="cyan", width=25)
        table.add_column("Text Preview", width=50)

        for i, (similarity, idx, metadata) in enumerate(similar_chunks, 1):
            sim_str = f"{similarity:.3f}"

            filename = metadata.get("filename", "unknown")
            if len(filename) > 22:
                filename = filename[:19] + "..."

            text = metadata.get("text", "No text available")
            if len(text) > 47:
                text = text[:44] + "..."

            table.add_row(str(i), sim_str, filename, text)

        title = "Similar Chunks" + (
            f" for: {question[:30]}..."
            if question and len(question) > 30
            else f" for: {question}"
            if question
            else ""
        )
        return Panel(table, title=title, border_style="green")
