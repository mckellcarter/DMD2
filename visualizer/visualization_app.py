"""
Interactive Dash application for DMD2 activation visualization.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import json
import argparse
from sklearn.neighbors import NearestNeighbors

# For dynamic UMAP recalculation
from process_embeddings import compute_umap, load_dataset_activations


class DMD2Visualizer:
    """Main visualizer application."""

    def __init__(self, data_dir: Path, embeddings_path: Path = None):
        """
        Args:
            data_dir: Root data directory
            embeddings_path: Optional path to precomputed embeddings CSV
        """
        self.data_dir = Path(data_dir)
        self.embeddings_path = embeddings_path
        self.df = None
        self.umap_params = None
        self.activations = None
        self.metadata_df = None
        self.nn_model = None  # Nearest neighbors model
        self.selected_point = None  # Currently selected point
        self.neighbor_indices = None  # Indices of neighbors
        self.class_labels = {}  # ImageNet class labels

        # Load class labels
        self.load_class_labels()

        # Load data
        self.load_data()

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.app.title = "DMD2 Activation Visualizer"
        self.build_layout()
        self.register_callbacks()

    def load_class_labels(self):
        """Load ImageNet class labels."""
        label_path = self.data_dir / "imagenet_class_labels.json"
        if label_path.exists():
            with open(label_path, 'r') as f:
                raw_labels = json.load(f)
                # Convert to dict with integer keys
                self.class_labels = {int(k): v[1] for k, v in raw_labels.items()}
            print(f"Loaded {len(self.class_labels)} ImageNet class labels")
        else:
            print(f"Warning: Class labels not found at {label_path}")
            self.class_labels = {}

    def get_class_name(self, class_id):
        """Get human-readable class name for a class ID."""
        if class_id in self.class_labels:
            return self.class_labels[class_id]
        return f"Unknown class {class_id}"

    def load_data(self):
        """Load embeddings or prepare for generation."""
        if self.embeddings_path and Path(self.embeddings_path).exists():
            print(f"Loading embeddings from {self.embeddings_path}")
            self.df = pd.read_csv(self.embeddings_path)

            # Load UMAP params
            param_path = Path(self.embeddings_path).with_suffix('.json')
            if param_path.exists():
                with open(param_path, 'r') as f:
                    self.umap_params = json.load(f)
            else:
                self.umap_params = {}

            print(f"Loaded {len(self.df)} samples")
        else:
            print("No embeddings found. Will load activations for dynamic UMAP.")
            self.df = pd.DataFrame()

    def load_activations_for_model(self, model_type: str):
        """Load raw activations for dynamic UMAP computation."""
        activation_dir = self.data_dir / "activations" / model_type
        metadata_path = self.data_dir / "metadata" / model_type / "dataset_info.json"

        if not metadata_path.exists():
            print(f"Warning: No metadata found for {model_type}")
            return None, None

        activations, metadata_df = load_dataset_activations(
            activation_dir,
            metadata_path
        )
        return activations, metadata_df

    def get_image_base64(self, image_path: str, size: tuple = (256, 256)):
        """Convert image to base64 for hover display."""
        try:
            full_path = self.data_dir / image_path
            img = Image.open(full_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def build_layout(self):
        """Build Dash layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("DMD2 Activation Visualizer", className="mb-4")
                ])
            ]),

            dbc.Row([
                # Left sidebar - controls + hover preview
                dbc.Col([
                    # Controls card
                    dbc.Card([
                        dbc.CardHeader("Controls"),
                        dbc.CardBody([
                            # Model selector
                            html.Label("Model Type"),
                            dcc.Dropdown(
                                id="model-selector",
                                options=[
                                    {"label": "ImageNet-64x64", "value": "imagenet"},
                                    {"label": "SDXL-1024", "value": "sdxl"},
                                    {"label": "SDv1.5-512", "value": "sdv1.5"}
                                ],
                                value=self.umap_params.get("model", "imagenet") if self.umap_params else "imagenet",
                                className="mb-3"
                            ),

                            # UMAP parameters
                            html.Hr(),
                            html.Label("UMAP n_neighbors"),
                            dcc.Slider(
                                id="n-neighbors-slider",
                                min=5,
                                max=100,
                                step=5,
                                value=self.umap_params.get("n_neighbors", 15) if self.umap_params else 15,
                                marks={5: "5", 25: "25", 50: "50", 75: "75", 100: "100"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            html.Label("UMAP min_dist", className="mt-3"),
                            dcc.Slider(
                                id="min-dist-slider",
                                min=0.0,
                                max=1.0,
                                step=0.05,
                                value=self.umap_params.get("min_dist", 0.1) if self.umap_params else 0.1,
                                marks={0.0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            # Recalculate button
                            html.Hr(),
                            dbc.Button(
                                "Recalculate UMAP",
                                id="recalculate-btn",
                                color="primary",
                                className="w-100 mb-2"
                            ),

                            # Export button
                            dbc.Button(
                                "Export Data",
                                id="export-btn",
                                color="secondary",
                                className="w-100"
                            ),
                            dcc.Download(id="download-data"),

                            # Status
                            html.Hr(),
                            html.Div(id="status-text", className="text-muted small")
                        ])
                    ], className="mb-3"),

                    # Hover preview card
                    dbc.Card([
                        dbc.CardHeader("Hover Preview"),
                        dbc.CardBody([
                            html.Div(id="hover-image"),
                            html.Div(id="hover-details", className="mt-2")
                        ])
                    ])
                ], width=3),

                # Main visualization
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-plot",
                                children=[
                                    dcc.Graph(
                                        id="umap-scatter",
                                        style={"height": "70vh"}
                                    )
                                ],
                                type="default"
                            )
                        ])
                    ])
                ], width=6),

                # Right sidebar - selected sample + neighbors
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Selected Sample"),
                            dbc.Button(
                                "✕",
                                id="clear-selection-btn",
                                color="link",
                                size="sm",
                                className="float-end p-0",
                                style={"fontSize": "20px", "lineHeight": "1", "display": "none"}
                            )
                        ]),
                        dbc.CardBody([
                            html.Div(id="selected-image"),
                            html.Div(id="selected-details", className="mt-2"),
                            html.Hr(),

                            # Neighbor search
                            html.Label("Find Similar Samples"),
                            dcc.Slider(
                                id="k-neighbors-slider",
                                min=1,
                                max=20,
                                step=1,
                                value=5,
                                marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            dbc.Button(
                                "Find Neighbors",
                                id="find-neighbors-btn",
                                color="info",
                                size="sm",
                                className="w-100 mt-2 mb-2",
                                disabled=True
                            ),

                            html.Hr(),
                            html.Label("Manual Neighbor Selection"),
                            html.Div(
                                "Click on other points to add/remove neighbors",
                                className="text-muted small mb-2"
                            ),
                            html.Div(id="neighbor-list", className="small", style={"maxHeight": "400px", "overflowY": "auto"})
                        ])
                    ])
                ], width=3)
            ]),

            # Hidden stores for state
            dcc.Store(id="selected-point-store", data=None),
            dcc.Store(id="neighbor-indices-store", data=None),
            dcc.Store(id="manual-neighbors-store", data=[])
        ], fluid=True, className="p-4")

    def fit_nearest_neighbors(self):
        """Fit KNN model on UMAP coordinates."""
        if self.df.empty or 'umap_x' not in self.df.columns:
            return

        coords = self.df[['umap_x', 'umap_y']].values
        self.nn_model = NearestNeighbors(n_neighbors=21, metric='euclidean')
        self.nn_model.fit(coords)

    def register_callbacks(self):
        """Register Dash callbacks."""

        @self.app.callback(
            Output("umap-scatter", "figure"),
            Output("status-text", "children"),
            Input("recalculate-btn", "n_clicks"),
            State("model-selector", "value"),
            State("n-neighbors-slider", "value"),
            State("min-dist-slider", "value"),
            prevent_initial_call=False
        )
        def update_plot(n_clicks, model, n_neighbors, min_dist):
            """Update UMAP plot."""
            ctx = callback_context
            triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else None

            # If recalculate button clicked, recompute UMAP
            if triggered == "recalculate-btn.n_clicks":
                status = f"Recalculating UMAP for {model}..."
                print(status)

                # Load activations if not cached
                if self.activations is None or self.umap_params.get("model") != model:
                    self.activations, self.metadata_df = self.load_activations_for_model(model)

                if self.activations is None:
                    return go.Figure(), f"Error: No data for {model}"

                # Compute UMAP
                embeddings = compute_umap(
                    self.activations,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42
                )

                # Update dataframe
                self.df = self.metadata_df.copy()
                self.df['umap_x'] = embeddings[:, 0]
                self.df['umap_y'] = embeddings[:, 1]

                # Fit nearest neighbors
                self.fit_nearest_neighbors()

                status = f"UMAP computed: {len(self.df)} samples"

            # Fit NN model if not already done
            if self.nn_model is None and not self.df.empty:
                self.fit_nearest_neighbors()

            # Create plot
            if self.df.empty:
                return go.Figure(), "No data loaded"

            # Determine color column
            if 'class_label' in self.df.columns:
                color_col = 'class_label'
                hover_data = ['class_label']
            else:
                color_col = None
                hover_data = []

            fig = px.scatter(
                self.df,
                x='umap_x',
                y='umap_y',
                color=color_col,
                hover_data=hover_data + ['sample_id'],
                title="DMD2 Activation UMAP",
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2'}
            )

            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(
                hovermode='closest',
                template='plotly_white'
            )

            status = f"Showing {len(self.df)} samples | n_neighbors={n_neighbors}, min_dist={min_dist}"
            return fig, status

        @self.app.callback(
            Output("hover-image", "children"),
            Output("hover-details", "children"),
            Input("umap-scatter", "hoverData")
        )
        def display_hover(hoverData):
            """Display image and info on hover."""
            if not hoverData or self.df.empty:
                return "Hover over a point", html.Div("No point hovered", className="text-muted small")

            point_idx = hoverData['points'][0]['pointIndex']
            sample = self.df.iloc[point_idx]

            # Get image thumbnail for hover
            img_b64 = self.get_image_base64(sample['image_path'], size=(200, 200))
            img_element = html.Img(
                src=img_b64,
                style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "4px"}
            ) if img_b64 else html.Div("Image not found")

            # Compact details for hover
            details = []
            details.append(html.Div([
                html.Strong("ID: "),
                html.Span(sample['sample_id'], className="small")
            ]))

            if 'class_label' in sample:
                class_id = int(sample['class_label'])
                class_name = self.get_class_name(class_id)
                details.append(html.Div([
                    html.Strong("Class: "),
                    html.Span(f"{class_id}: {class_name}", className="small")
                ]))

            details.append(html.Div([
                html.Strong("Coords: "),
                html.Span(f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})", className="small")
            ]))

            return img_element, html.Div(details)

        @self.app.callback(
            Output("selected-image", "children"),
            Output("selected-details", "children"),
            Output("find-neighbors-btn", "disabled"),
            Output("selected-point-store", "data"),
            Output("manual-neighbors-store", "data"),
            Output("clear-selection-btn", "style"),
            Output("neighbor-indices-store", "data", allow_duplicate=True),
            Input("umap-scatter", "clickData"),
            Input("clear-selection-btn", "n_clicks"),
            State("selected-point-store", "data"),
            State("manual-neighbors-store", "data"),
            State("neighbor-indices-store", "data"),
            prevent_initial_call=True
        )
        def display_selected(clickData, clear_clicks, current_selected, manual_neighbors, knn_neighbors):
            """Handle point selection and neighbor toggling."""
            ctx = callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Handle clear button
            if trigger_id == "clear-selection-btn":
                return (
                    "Click a point to select",
                    html.Div("No point selected", className="text-muted small"),
                    True,  # Disable find neighbors
                    None,  # Clear selected point
                    [],    # Clear manual neighbors
                    {"fontSize": "20px", "lineHeight": "1", "display": "none"},  # Hide clear button
                    None   # Clear KNN neighbors
                )

            # Handle click on plot
            if not clickData or self.df.empty:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            point_idx = clickData['points'][0]['pointIndex']

            # Ensure lists are initialized
            if manual_neighbors is None:
                manual_neighbors = []
            if knn_neighbors is None:
                knn_neighbors = []

            # If no point currently selected, select this one
            if current_selected is None:
                sample = self.df.iloc[point_idx]
                img_b64 = self.get_image_base64(sample['image_path'])
                img_element = html.Img(
                    src=img_b64,
                    style={"width": "100%", "border": "2px solid #0d6efd", "borderRadius": "4px"}
                ) if img_b64 else html.Div("Image not found")

                details = []
                details.append(html.P([html.Strong("Sample ID: "), sample['sample_id']]))
                if 'class_label' in sample:
                    class_id = int(sample['class_label'])
                    class_name = self.get_class_name(class_id)
                    details.append(html.P([html.Strong("Class: "), f"{class_id}: {class_name}"]))
                details.append(html.P([
                    html.Strong("UMAP Coords: "),
                    f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"
                ]))
                details.append(html.P(
                    "Click points to add or remove neighbors",
                    className="text-info small"
                ))

                return (
                    img_element,
                    html.Div(details),
                    False,  # Enable find neighbors
                    point_idx,
                    [],     # Reset manual neighbors
                    {"fontSize": "20px", "lineHeight": "1", "display": "inline"},  # Show clear button
                    None    # Clear KNN neighbors
                )

            # If clicking the same point, do nothing
            if point_idx == current_selected:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            # Toggle neighbor: check if in manual or KNN list
            # Priority: if in manual list, remove from manual; if in KNN list, move to manual (for removal); if in neither, add to manual
            if point_idx in manual_neighbors:
                # Remove from manual list
                manual_neighbors.remove(point_idx)
                new_knn = knn_neighbors  # Keep KNN unchanged
            elif point_idx in knn_neighbors:
                # If clicking a KNN neighbor, remove it from KNN list and add to manual remove list
                # This effectively removes it since we filter KNN against manual in display
                new_knn = [idx for idx in knn_neighbors if idx != point_idx]
                manual_neighbors = manual_neighbors  # Keep manual unchanged
            else:
                # Add to manual list
                manual_neighbors.append(point_idx)
                new_knn = knn_neighbors  # Keep KNN unchanged

            # Keep the current display but return updated neighbor lists
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                manual_neighbors,
                dash.no_update,
                new_knn
            )

        @self.app.callback(
            Output("neighbor-indices-store", "data"),
            Input("find-neighbors-btn", "n_clicks"),
            State("selected-point-store", "data"),
            State("k-neighbors-slider", "value"),
            State("manual-neighbors-store", "data"),
            prevent_initial_call=True
        )
        def find_neighbors(n_clicks, selected_idx, k, manual_neighbors):
            """Find k nearest neighbors and merge with manual neighbors."""
            if selected_idx is None or self.nn_model is None:
                return None

            # Get coordinates of selected point
            selected_coords = self.df.iloc[selected_idx][['umap_x', 'umap_y']].values.reshape(1, -1)

            # Find neighbors (k+1 to exclude the point itself)
            distances, indices = self.nn_model.kneighbors(selected_coords, n_neighbors=k+1)

            # Remove the point itself (first result)
            neighbor_indices = indices[0][1:].tolist()

            # Merge with manual neighbors (manual neighbors take priority)
            if manual_neighbors:
                # Add manual neighbors that aren't already in the KNN list
                for manual_idx in manual_neighbors:
                    if manual_idx not in neighbor_indices:
                        neighbor_indices.append(manual_idx)

            return neighbor_indices

        @self.app.callback(
            Output("neighbor-list", "children"),
            Input("manual-neighbors-store", "data"),
            Input("neighbor-indices-store", "data"),
            State("selected-point-store", "data"),
            prevent_initial_call=False
        )
        def display_neighbor_list(manual_neighbors, knn_neighbors, selected_idx):
            """Display combined neighbor list with remove buttons for manual neighbors."""
            if self.df.empty:
                return html.Div("No data loaded", className="text-muted")

            if not manual_neighbors and not knn_neighbors:
                return html.Div("No neighbors selected", className="text-muted small")

            # Ensure lists are not None
            manual_neighbors = manual_neighbors or []
            knn_neighbors = knn_neighbors or []

            # Build combined list (KNN first, then manual additions at bottom)
            all_neighbors = []
            # Add KNN neighbors that aren't in manual list
            for idx in knn_neighbors:
                if idx not in manual_neighbors:
                    all_neighbors.append(idx)
            # Add manual neighbors at the end (most recently added at bottom)
            all_neighbors.extend(manual_neighbors)

            # Build neighbor cards
            neighbor_items = []
            for i, idx in enumerate(all_neighbors):
                neighbor_sample = self.df.iloc[idx]
                is_manual = idx in manual_neighbors

                # Get image thumbnail
                img_b64 = self.get_image_base64(neighbor_sample['image_path'], size=(64, 64))

                # Calculate distance if selected point exists
                dist_text = ""
                if selected_idx is not None:
                    selected_coords = self.df.iloc[selected_idx][['umap_x', 'umap_y']].values
                    neighbor_coords = self.df.iloc[idx][['umap_x', 'umap_y']].values
                    dist = np.linalg.norm(selected_coords - neighbor_coords)
                    dist_text = f"(dist: {dist:.2f})"

                # Build neighbor card
                neighbor_card = dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Img(
                                    src=img_b64,
                                    style={"width": "64px", "height": "64px"}
                                ) if img_b64 else html.Div("No img", style={"width": "64px"})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.Strong(f"#{i+1} "),
                                    html.Span("✓ " if is_manual else "", className="text-success"),
                                    html.Span(dist_text, className="text-muted small"),
                                    html.Br(),
                                    html.Span(
                                        f"{int(neighbor_sample['class_label'])}: {self.get_class_name(int(neighbor_sample['class_label']))}",
                                        className="small"
                                    ) if 'class_label' in neighbor_sample else None,
                                ])
                            ], width=8)
                        ])
                    ], className="p-2")
                ], className="mb-2", style={"border": "2px solid green" if is_manual else "1px solid #dee2e6"})

                neighbor_items.append(neighbor_card)

            return html.Div(neighbor_items)

        @self.app.callback(
            Output("umap-scatter", "figure", allow_duplicate=True),
            Input("selected-point-store", "data"),
            Input("manual-neighbors-store", "data"),
            Input("neighbor-indices-store", "data"),
            State("umap-scatter", "figure"),
            prevent_initial_call=True
        )
        def highlight_neighbors(selected_idx, manual_neighbors, neighbor_indices, current_figure):
            """Highlight selected point and neighbors on plot."""
            if current_figure is None:
                return current_figure

            fig = go.Figure(current_figure)

            # Remove any existing highlight traces (keep only the first trace which is the main scatter)
            # Keep only trace 0 (the main UMAP scatter plot)
            fig.data = [fig.data[0]]

            # If no point selected, just return with only main scatter
            if selected_idx is None:
                return fig

            # Always highlight selected point in blue
            selected_coords = self.df.iloc[[selected_idx]][['umap_x', 'umap_y']]

            fig.add_trace(go.Scatter(
                x=selected_coords['umap_x'],
                y=selected_coords['umap_y'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='blue',
                    symbol='circle-open',
                    line=dict(width=3, color='blue')
                ),
                name='Selected Point',
                hoverinfo='skip',
                showlegend=True
            ))

            # Add KNN neighbors in red with thin line if any
            if neighbor_indices:
                knn_coords = self.df.iloc[neighbor_indices][['umap_x', 'umap_y']]

                fig.add_trace(go.Scatter(
                    x=knn_coords['umap_x'],
                    y=knn_coords['umap_y'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle-open',
                        line=dict(width=1, color='red')
                    ),
                    name='KNN Neighbors',
                    hoverinfo='skip',
                    showlegend=True
                ))

            # Add manual neighbors in red with thicker line if any
            if manual_neighbors:
                manual_coords = self.df.iloc[manual_neighbors][['umap_x', 'umap_y']]

                fig.add_trace(go.Scatter(
                    x=manual_coords['umap_x'],
                    y=manual_coords['umap_y'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle-open',
                        line=dict(width=2, color='red')
                    ),
                    name='Manual Neighbors',
                    hoverinfo='skip',
                    showlegend=True
                ))

            return fig

        @self.app.callback(
            Output("download-data", "data"),
            Input("export-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def export_data(n_clicks):
            """Export current embeddings to CSV."""
            if self.df.empty:
                return None

            return dcc.send_data_frame(
                self.df.to_csv,
                "dmd2_embeddings_export.csv",
                index=False
            )

    def run(self, debug: bool = False, port: int = 8050):
        """Run the Dash app."""
        print(f"\nStarting DMD2 Visualizer on http://localhost:{port}")
        self.app.run(debug=debug, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="DMD2 Activation Visualizer"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to precomputed embeddings CSV (optional)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run server on"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    args = parser.parse_args()

    visualizer = DMD2Visualizer(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings
    )

    visualizer.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
