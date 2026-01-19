from typing import Annotated, List, Literal, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

DType = TypeVar("DType", bound=np.generic)
IntType = TypeVar("IntType", bound=np.int32)
Array2xN = Annotated[npt.NDArray[DType], Literal[2, "N"]]
Array2 = Annotated[npt.NDArray[DType], Literal[2]]
ArrayNx2 = Annotated[npt.NDArray[DType], Literal["N", 2]]
ArrayNxM = Annotated[npt.NDArray[DType], Literal["N", "M"]]
ArrayNxN = Annotated[npt.NDArray[DType], Literal["N", "N"]]
IntArrayNx2 = Annotated[npt.NDArray[IntType], Literal["N", 2]]
IntArray2 = Annotated[npt.NDArray[IntType], Literal[2]]
IntArrayNxM = Annotated[npt.NDArray[IntType], Literal["N", "M"]]


class MultiDijkstra:
    def __init__(
        self,
        occupancy_data: ArrayNxM,
        resolution: float,
        origin: Array2,
        approx_downsample_resolution: Optional[float] = 0.2,
        return_predecessors: bool = True,
    ):
        """
        Initialize the MultiDijkstra object.

        Args:
            occupancy_data (ArrayNxM): The occupancy data (2D array). Indexed as (y, x).
            resolution (float): The resolution of the occupancy data.
            origin (Array2): The origin point (x, y) in the occupancy grid.
            approx_downsample_resolution (Optional[float]): The approximate downsample resolution.
            return_predecessors (bool): Whether to return the predecessors in the graph. Required for path reconstruction and for path length computation.
        """
        occupancy_data[occupancy_data < 0] = 0
        occupancy_data = occupancy_data > 0

        self.original_resolution = resolution
        self.origin = origin
        if approx_downsample_resolution is not None:
            occupancy_data, self.downsample_factor = self._downsample_occupancy(
                occupancy_data, resolution, approx_downsample_resolution
            )
            self.occupancy_data = occupancy_data > 0
            self.resolution = self.original_resolution * self.downsample_factor
        else:
            self.occupancy_data = occupancy_data
            self.resolution = self.original_resolution
            self.downsample_factor = 1

        self.occupancy_data = self.occupancy_data.astype(np.uint8)
        self.connectivity = 8
        self.graph, self.node_index_map, self.node_coords = self._occupancy_to_sparse_graph(
            self.occupancy_data, connectivity=self.connectivity
        )
        self.return_predecessors = return_predecessors

    @staticmethod
    def _downsample_occupancy(
        occupancy_data: ArrayNxM, resolution: float, approx_downsample_resolution: float
    ) -> Tuple[ArrayNxM, int]:
        factor = int(approx_downsample_resolution / resolution)
        if factor <= 1:
            return occupancy_data, 1
        h, w = occupancy_data.shape
        h_trim, w_trim = h - h % factor, w - w % factor
        trimmed = occupancy_data[:h_trim, :w_trim]
        downsampled = trimmed.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(axis=(1, 3))
        return downsampled, factor

    @staticmethod
    def _occupancy_to_sparse_graph(
        occupancy: ArrayNxM, connectivity: int = 4, include_occupied: bool = True, occupied_cost=100
    ) -> Tuple[csr_matrix, IntArrayNxM, IntArrayNxM]:
        """
        Convert a 2D occupancy grid to a sparse graph representation.

        Args:
            occupancy (ArrayNxM): 2D occupancy grid (0 = free, 1 = occupied).
            connectivity (int): Connectivity type (4 or 8).
            include_occupied (bool): Whether to include edges to occupied cells.
            occupied_cost (float): Cost for edges to occupied cells.

        Returns:
            Tuple[coo_matrix, ArrayNxM, ArrayNx2]: Sparse graph, node index map, and node coordinates.
        """
        assert occupancy.ndim == 2, "Occupancy must be a 2D array"
        assert occupancy.dtype in [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64], (
            f"Occupancy must be an integer type, is {occupancy.dtype}"
        )

        H, W = occupancy.shape
        assert connectivity in [4, 8], "Only 4 or 8 connectivity supported"

        free_mask = occupancy == 0
        num_nodes = H * W  # all cells have a node index now

        # Map from grid (y,x) to node index (flattened index)
        def grid_to_index(y, x):
            return y * W + x

        node_index_map = np.arange(num_nodes).reshape(H, W)
        node_coords = np.column_stack(np.unravel_index(np.arange(num_nodes), (H, W)))

        # Directions: (dy, dx, weight)
        directions = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]
        if connectivity == 8:
            directions += [(-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]

        rows = []
        cols = []
        weights = []

        y_indices, x_indices = np.indices((H, W)).reshape(2, -1)

        for dy, dx, weight in directions:
            ny = y_indices + dy
            nx = x_indices + dx

            # Filter valid bounds
            valid = (0 <= ny) & (ny < H) & (0 <= nx) & (nx < W)
            y_from = y_indices[valid]
            x_from = x_indices[valid]
            y_to = ny[valid]
            x_to = nx[valid]

            from_ids = grid_to_index(y_from, x_from)
            to_ids = grid_to_index(y_to, x_to)

            from_free = free_mask[y_from, x_from]
            to_free = free_mask[y_to, x_to]

            if include_occupied:
                costs = np.where(from_free & to_free, weight, occupied_cost)
                rows.append(from_ids)
                cols.append(to_ids)
                weights.append(costs)
            else:
                # only include edges between free cells
                mask = from_free & to_free
                rows.append(from_ids[mask])
                cols.append(to_ids[mask])
                weights.append(np.full(np.sum(mask), weight, dtype=float))

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)

        graph = coo_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

        return graph, node_index_map, node_coords  # type: ignore

    def _reconstruct_path_celled(self, predecessors: np.ndarray, start_node: int, goal_node: int) -> IntArrayNx2:
        path = []
        current = goal_node
        while current != -9999:
            path.append(current)
            if current == start_node:
                break
            current = predecessors[current]
        path = path[::-1] if path and path[-1] == start_node else []
        path = self.node_coords[path] if path else np.array([], dtype=int).reshape(0, 2)
        return path

    def _world_to_cell(self, points: ArrayNx2) -> IntArrayNx2:
        xy_coords = np.floor((points - self.origin) / self.resolution).astype(np.int32)
        return (
            np.vstack((xy_coords[:, 1], xy_coords[:, 0])).T
            if xy_coords.ndim == 2
            else np.array((xy_coords[1], xy_coords[0]))
        )  # (x, y) -> (y, x)

    def _cell_to_world(self, points: IntArrayNx2) -> ArrayNx2:
        yx_coords = points * self.resolution + np.array((self.origin[1], self.origin[0])) + self.resolution / 2
        return (
            np.vstack((yx_coords[:, 1], yx_coords[:, 0])).T
            if yx_coords.ndim == 2
            else np.array((yx_coords[1], yx_coords[0]))
        )  # (y, x) -> (x, y)

    def _compute_path_length(self, path: ArrayNx2) -> float:
        if path.shape[0] < 2:
            return 0.0
        if self.connectivity == 4:
            return np.sum(np.abs(np.diff(path, axis=0)))
        else:
            return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    def _project_outlier_nodes_to_border(self, nodes: IntArrayNx2) -> Tuple[IntArrayNx2, np.ndarray]:
        H, W = self.occupancy_data.shape
        projected_nodes = np.copy(nodes)
        projected_nodes[:, 0] = np.clip(nodes[:, 0], 0, H - 1)
        projected_nodes[:, 1] = np.clip(nodes[:, 1], 0, W - 1)
        if self.connectivity == 4:
            projection_distances = np.sum(np.abs(projected_nodes - nodes), axis=1)
        else:
            projection_distances = np.linalg.norm(projected_nodes - nodes, axis=1)
        return projected_nodes, projection_distances

    def _get_min_distance_to_goals_celled(
        self, start_node: IntArray2, goal_nodes: IntArrayNx2
    ) -> Tuple[np.ndarray, Optional[List[IntArrayNx2]]]:
        projected_start_node, projection_distance_start = self._project_outlier_nodes_to_border(
            start_node.reshape(1, 2)
        )
        projected_start_node = projected_start_node[0]  # Convert to 1D array
        projection_distance_start = projection_distance_start[0]  # Convert to scalar
        projected_goal_nodes, projection_distance_goals = self._project_outlier_nodes_to_border(goal_nodes)

        start_idx: int = self.node_index_map[projected_start_node[0], projected_start_node[1]]  # type: ignore
        goal_idx = self.node_index_map[projected_goal_nodes[:, 0], projected_goal_nodes[:, 1]]

        results = dijkstra(self.graph, directed=False, indices=start_idx, return_predecessors=self.return_predecessors)
        if self.return_predecessors:
            costs, predecessors = results
            paths = [self._reconstruct_path_celled(predecessors, start_idx, g) for g in goal_idx]
            if projection_distance_start > 0:
                paths = [np.vstack((start_node, p)) for p in paths]
            paths = [
                (np.vstack((p, original_goal)) if projected_distance > 0 else p)
                for p, projected_distance, original_goal in zip(paths, projection_distance_goals, goal_nodes)
            ]
        else:
            costs = results
            predecessors = None
            paths = None

        return costs[goal_idx], paths

    def get_min_distance_to_goals(
        self, start_node: Array2, goal_nodes: ArrayNx2
    ) -> Tuple[np.ndarray, Optional[list[float]], Optional[List[ArrayNx2]]]:
        """
        Get the minimum distance from a start node to each goal inside goal_nodes. Nodes outside the occupancy grid are projected to the nearest border and the returned path includes this projection.

        Args:
            start_node (Array2): Start node in world coordinates.
            goal_nodes (ArrayNx2): Goal nodes in world coordinates.

        Returns:
            Tuple[np.ndarray, Optional[List[ArrayNx2]]]: Distances and paths to each goal node.
        """
        start_node_cell = self._world_to_cell(start_node)
        goal_nodes_cell = self._world_to_cell(goal_nodes)

        costs, paths = self._get_min_distance_to_goals_celled(start_node_cell, goal_nodes_cell)

        costs *= self.resolution
        if paths is not None:
            paths = [self._cell_to_world(path) for path in paths]
            distances = [self._compute_path_length(path) for path in paths]
        else:
            distances = None
        return costs, distances, paths

    def _get_min_distance_to_goalset_celled(
        self, start_node: Array2, goal_nodes: ArrayNx2
    ) -> Tuple[np.ndarray, Optional[ArrayNx2]]:
        projected_start_node, projection_distance_start = self._project_outlier_nodes_to_border(
            start_node.reshape(1, 2)
        )
        projected_start_node = projected_start_node[0]  # Convert to 1D array
        projection_distance_start = projection_distance_start[0]  # Convert to scalar
        projected_goal_nodes, projection_distance_goals = self._project_outlier_nodes_to_border(goal_nodes)

        start_idx: int = self.node_index_map[projected_start_node[0], projected_start_node[1]]  # type: ignore
        goal_idx = self.node_index_map[projected_goal_nodes[:, 0], projected_goal_nodes[:, 1]]

        results = dijkstra(
            self.graph, directed=False, indices=goal_idx, return_predecessors=self.return_predecessors, min_only=True
        )
        if self.return_predecessors:
            costs, predecessors, sources = results
            source_idx = sources[start_idx]
            cost = costs[start_idx]
            path = np.flip(self._reconstruct_path_celled(predecessors, source_idx, start_idx), axis=0)

            if projection_distance_start > 0:
                path = np.vstack((start_node, path))

            goal_index_in_set = np.where(goal_idx == source_idx)[0][0]
            if projection_distance_goals[goal_index_in_set] > 0:
                original_goal = goal_nodes[goal_index_in_set]
                path = np.vstack((path, original_goal))
        else:
            costs, sources = results
            source_idx = sources[start_idx]
            cost = costs[start_idx]
            predecessors = None
            path = None

        return cost, path

    def get_min_distance_to_goalset(
        self, start_node: Array2, goal_nodes: ArrayNx2
    ) -> Tuple[float, Optional[float], Optional[ArrayNx2]]:
        """
        Get the minimum distance from a start node to any node in goal_nodes. Nodes outside the occupancy grid are projected to the nearest border and the returned path includes this projection.

        Args:
            start_node (Array2): Start node in world coordinates.
            goal_nodes (ArrayNx2): Goal nodes in world coordinates.

        Returns:
            Tuple[float, Optional[float], Optional[ArrayNx2]]: Minimum cost, distance, and path to the closest goal node.
        """
        start_node_cell = self._world_to_cell(start_node)
        goal_nodes_cell = self._world_to_cell(goal_nodes)
        cost, path = self._get_min_distance_to_goalset_celled(start_node_cell, goal_nodes_cell)
        cost *= self.resolution
        if path is not None:
            path = self._cell_to_world(path)
            distance = self._compute_path_length(path)
        else:
            distance = None
        return cost, distance, path

    def _get_pairwise_shortest_distances_celled(
        self, start_nodes: IntArrayNx2
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if np.shape(start_nodes)[0] < 2:
            return np.zeros((1, 1)), (np.zeros((1, 1)) if self.return_predecessors else None)
        projected_start_nodes, projection_distances = self._project_outlier_nodes_to_border(start_nodes)
        nodes_indices = self.node_index_map[projected_start_nodes[:, 0], projected_start_nodes[:, 1]]
        results = dijkstra(
            self.graph, directed=False, indices=nodes_indices, return_predecessors=self.return_predecessors
        )
        if self.return_predecessors:
            costs, predecessors = results
            path_lengths = np.full((len(nodes_indices), len(nodes_indices)), np.inf, dtype=float)
            for i, src_id in enumerate(nodes_indices):
                for j, dst_id in enumerate(nodes_indices):
                    path = self._reconstruct_path_celled(predecessors[i], src_id, dst_id)
                    length = self._compute_path_length(path) + projection_distances[i] + projection_distances[j]
                    path_lengths[i][j] = length
                    path_lengths[j][i] = length
        else:
            costs = results
            path_lengths = None

        costs = costs[np.reshape(np.arange(len(start_nodes)), (-1, 1)), np.reshape(nodes_indices, (1, -1))]
        return costs, path_lengths

    def get_pairwise_shortest_distances(self, start_nodes: ArrayNx2) -> Tuple[ArrayNxN, Optional[ArrayNxN]]:
        """
        Get pairwise shortest distances between a set of start nodes. Nodes outside the occupancy grid are projected to the nearest border and the distance to the border is added to the path length.

        Args:
            start_nodes (ArrayNx2): Array of start nodes in world coordinates.

        Returns:
            Tuple[ArrayNxN, Optional[ArrayNxN]]: Costs and path lengths between start nodes.
        """
        start_nodes_cell = self._world_to_cell(start_nodes)
        costs, path_lengths = self._get_pairwise_shortest_distances_celled(start_nodes_cell)

        costs *= self.resolution
        if path_lengths is not None:
            path_lengths *= self.resolution

        return costs, path_lengths
